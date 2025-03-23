import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from tqdm import tqdm
import random

class DualInputManipulationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, orig_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.orig_dir = orig_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        orig_path = os.path.join(self.orig_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
        
        mask_sum = np.sum(mask)
        is_fully_fake = mask_sum == 1.0 * mask.size  # Image is over 95% fake
        is_fully_real = mask_sum == 0.0 * mask.size  # Image is over 95% real
        
        if os.path.exists(orig_path):
            orig_image = cv2.imread(orig_path)
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            has_original = True
            image_type = 1
        else:
            # no original available - either Type 2 or Type 3
            if is_fully_real:
                # type 2: Real
                orig_image = image.copy()  # Use the image itself as original
                has_original = False  # Still mark as no original available
                image_type = 2
            else:
                # Type 3: fake or heavily manipulated image
                orig_image = np.zeros_like(image)  # no original reference
                has_original = False
                image_type = 3
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask, orig_image=orig_image)
            image = transformed['image']
            mask = transformed['mask'].unsqueeze(0)  # Add channel dimension
            orig_image = transformed['orig_image']
        else:
            # default normalization if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            orig_image = torch.from_numpy(orig_image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
            
        return {
            'image': image,
            'orig_image': orig_image,
            'mask': mask,
            'image_id': img_name.split('.')[0],
            'has_original': has_original,
            'image_type': image_type  
        }

def get_dual_transforms():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'orig_image': 'image'})


class BoundaryLoss(nn.Module):
    def __init__(self, theta=1.5):
        super(BoundaryLoss, self).__init__()
        self.theta = theta
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # bound
        target = torch.clamp(target, 0, 1)
        
        # calculate gradients using Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(pred.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        
        # filter
        edge_pred_x = F.conv2d(pred, sobel_x, padding=1)
        edge_pred_y = F.conv2d(pred, sobel_y, padding=1)
        edge_target_x = F.conv2d(target, sobel_x, padding=1)
        edge_target_y = F.conv2d(target, sobel_y, padding=1)
        
        epsilon = 1e-6
        edge_pred = torch.sqrt(edge_pred_x**2 + edge_pred_y**2 + epsilon)
        edge_target = torch.sqrt(edge_target_x**2 + edge_target_y**2 + epsilon)
        
        edge_pred = torch.clamp(edge_pred, 0, 10)
        edge_target = torch.clamp(edge_target, 0, 10)
        
        boundary_loss = F.l1_loss(edge_pred, edge_target)
        
        return boundary_loss



class TeacherModel(nn.Module):
    def __init__(self, encoder_name="mit_b3"):
        super(TeacherModel, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=6,  # 3 for manipulated + 3 for original
            classes=1,
        )
        
    def forward(self, x, orig):
        x_combined = torch.cat([x, orig], dim=1)
        return self.model(x_combined)

class StudentModel(nn.Module):
    def __init__(self, encoder_name="mit_b5"): # larger
        super(StudentModel, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        
    def forward(self, x):
        return self.model(x)

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.dice_bce_loss = DiceBCEBoundaryLoss()
        
    def forward(self, student_logits, teacher_logits, targets):
        hard_loss = self.dice_bce_loss(student_logits, targets)
        
        soft_targets = torch.sigmoid(teacher_logits.detach() / self.temperature)
        soft_student = torch.sigmoid(student_logits / self.temperature)
        soft_loss = F.binary_cross_entropy(soft_student, soft_targets)
        
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss

class DiceBCEBoundaryLoss(nn.Module):
    def __init__(self, dice_weight=0.5, boundary_weight=0.05):
        super(DiceBCEBoundaryLoss, self).__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.boundary_loss = BoundaryLoss()
        
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets)
        
        inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs_sigmoid * targets).sum()
        dice = 1 - (2. * intersection + 1e-6) / (inputs_sigmoid.sum() + targets.sum() + 1e-6)
        
        try:
            boundary = self.boundary_loss(inputs, targets)
            if torch.isnan(boundary) or torch.isinf(boundary):
                print("Warning: NaN or Inf in boundary loss, setting to zero")
                boundary = torch.tensor(0.0, device=inputs.device)
        except Exception as e:
            print(f"Error in boundary loss: {e}")
            boundary = torch.tensor(0.0, device=inputs.device)
        
        # combined loss with safety check
        total_loss = (1-self.dice_weight-self.boundary_weight) * bce + self.dice_weight * dice + self.boundary_weight * boundary
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN in final loss, falling back to BCE+Dice only")
            total_loss = 0.5 * bce + 0.5 * dice
            
        return total_loss

def train_teacher_with_mixup_cutmix(model, dataloader, optimizer, criterion, device, 
                                    use_mixup=True, mixup_alpha=0.2, 
                                    use_cutmix=True, cutmix_alpha=1.0,
                                    scheduler=None):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    for batch in tqdm(dataloader, desc="Training Teacher"):
        images = batch['image'].to(device)
        orig_images = batch['orig_image'].to(device)
        masks = batch['mask'].to(device)
        has_original = batch['has_original']
        image_types = batch['image_type']
        
        type1_samples = has_original.bool()
        
        type2_samples = (image_types == 2)
        
        usable_samples = type1_samples | type2_samples
        
        if not usable_samples.any():
            continue
            
        images = images[usable_samples]
        orig_images = orig_images[usable_samples]
        masks = masks[usable_samples]
        
        batch_size = images.size(0)
        if batch_size <= 1:  # Skip batches with just 1 sample (can't mix with others)
            continue
            
        do_mixup = use_mixup and (not use_cutmix or random.random() < 0.5)
        do_cutmix = use_cutmix and not do_mixup
        
        if do_mixup:
            indices = torch.randperm(batch_size).to(device)
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            
            mixed_images = lam * images + (1 - lam) * images[indices]
            mixed_orig_images = lam * orig_images + (1 - lam) * orig_images[indices]
            mixed_masks = torch.max(masks, masks[indices])
            
            images = mixed_images
            orig_images = mixed_orig_images
            masks = mixed_masks
            
        elif do_cutmix:
            indices = torch.randperm(batch_size).to(device)
            lam = np.random.beta(cutmix_alpha, cutmix_alpha)
            
            _, _, height, width = images.shape
            cut_ratio = np.sqrt(1.0 - lam)
            cut_h = int(height * cut_ratio)
            cut_w = int(width * cut_ratio)
            
            cx = np.random.randint(width)
            cy = np.random.randint(height)
            
            x1 = max(cx - cut_w // 2, 0)
            y1 = max(cy - cut_h // 2, 0)
            x2 = min(cx + cut_w // 2, width)
            y2 = min(cy + cut_h // 2, height)
            
            mixed_images = images.clone()
            mixed_orig_images = orig_images.clone()
            mixed_masks = masks.clone()
            
            mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
            mixed_orig_images[:, :, y1:y2, x1:x2] = orig_images[indices, :, y1:y2, x1:x2]
            mixed_masks[:, :, y1:y2, x1:x2] = masks[indices, :, y1:y2, x1:x2]
            
            images = mixed_images
            orig_images = mixed_orig_images
            masks = mixed_masks
        
        outputs = model(images, orig_images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
    return epoch_loss / max(1, batch_count)  

def train_student_with_mixup_cutmix(student_model, teacher_model, dataloader, optimizer, criterion, device,
                                    use_mixup=True, mixup_alpha=0.2, 
                                    use_cutmix=True, cutmix_alpha=1.0,
                                    scheduler=None):
    student_model.train()
    teacher_model.eval()
    epoch_loss = 0
    batch_count = 0
    
    for batch in tqdm(dataloader, desc="Training Student"):
        images = batch['image'].to(device)
        orig_images = batch['orig_image'].to(device)
        masks = batch['mask'].to(device)
        has_original = batch['has_original'].to(device)
        image_types = batch['image_type'].to(device)
        
        type1_samples = has_original.bool()
        type2_samples = (image_types == 2)
        teacher_usable_samples = type1_samples | type2_samples
        
        batch_size = images.size(0)
        if batch_size <= 1:  
            continue
            
        do_mixup = use_mixup and (not use_cutmix or random.random() < 0.5)
        do_cutmix = use_cutmix and not do_mixup
        
        if do_mixup:
            indices = torch.randperm(batch_size).to(device)
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            
            mixed_images = lam * images + (1 - lam) * images[indices]
            mixed_orig_images = lam * orig_images + (1 - lam) * orig_images[indices]
            mixed_masks = torch.max(masks, masks[indices])
            
            mixed_teacher_usable = teacher_usable_samples & teacher_usable_samples[indices]
            
            images = mixed_images
            orig_images = mixed_orig_images
            masks = mixed_masks
            teacher_usable_samples = mixed_teacher_usable
            
        elif do_cutmix:
            indices = torch.randperm(batch_size).to(device)
            lam = np.random.beta(cutmix_alpha, cutmix_alpha)
            
            _, _, height, width = images.shape
            cut_ratio = np.sqrt(1.0 - lam)
            cut_h = int(height * cut_ratio)
            cut_w = int(width * cut_ratio)
            
            # Get random center point
            cx = np.random.randint(width)
            cy = np.random.randint(height)
            
            # Calculate box boundaries
            x1 = max(cx - cut_w // 2, 0)
            y1 = max(cy - cut_h // 2, 0)
            x2 = min(cx + cut_w // 2, width)
            y2 = min(cy + cut_h // 2, height)
            
            # Apply cutmix to all inputs
            mixed_images = images.clone()
            mixed_orig_images = orig_images.clone()
            mixed_masks = masks.clone()
            
            mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
            mixed_orig_images[:, :, y1:y2, x1:x2] = orig_images[indices, :, y1:y2, x1:x2]
            mixed_masks[:, :, y1:y2, x1:x2] = masks[indices, :, y1:y2, x1:x2]
            
            mixed_teacher_usable = teacher_usable_samples.clone()
            
            images = mixed_images
            orig_images = mixed_orig_images
            masks = mixed_masks
            teacher_usable_samples = mixed_teacher_usable
        
        # Forward pass for student on all samples
        student_outputs = student_model(images)
        
        # Calculate loss based on which samples can use teacher
        if teacher_usable_samples.any():
            with torch.no_grad():
                usable_images = images[teacher_usable_samples]
                usable_orig_images = orig_images[teacher_usable_samples]
                teacher_outputs = teacher_model(usable_images, usable_orig_images)
            
            full_teacher_outputs = torch.zeros_like(student_outputs).detach()
            
            full_teacher_outputs[teacher_usable_samples] = teacher_outputs
            
            loss = criterion(student_outputs, full_teacher_outputs, masks)
        else:
            loss = F.binary_cross_entropy_with_logits(student_outputs, masks) + \
                  dice_loss(torch.sigmoid(student_outputs), masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        epoch_loss += loss.item()
        batch_count += 1
    
    return epoch_loss / max(1, batch_count)

def dice_loss(pred, target):
    intersection = (pred * target).sum()
    dice = 1 - (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return dice

def train_teacher_student_models(config):
    device = config['DEVICE']
    
    print("Creating datasets...")
    train_transform = get_dual_transforms()
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'orig_image': 'image'})
    
    train_dataset = DualInputManipulationDataset(
        img_dir=config['TRAIN_IMG_DIR'],
        mask_dir=config['TRAIN_MASK_DIR'],
        orig_dir=config['TRAIN_ORIG_DIR'],
        transform=train_transform
    )
    
    val_dataset = DualInputManipulationDataset(
        img_dir=config['TRAIN_IMG_DIR'],
        mask_dir=config['TRAIN_MASK_DIR'],
        orig_dir=config['TRAIN_ORIG_DIR'],
        transform=val_transform
    )
    
    val_size = int(config['VALIDATION_SPLIT'] * len(train_dataset))
    train_size = len(train_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=config['NUM_WORKERS']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=config['NUM_WORKERS']
    )
    
    print("Initializing teacher and student models...")
    teacher_model = TeacherModel(encoder_name='mit_b3').to(device)
    student_model = StudentModel(encoder_name='mit_b5').to(device)
    
    teacher_criterion = DiceBCEBoundaryLoss(dice_weight=0.5, boundary_weight=0.2)
    student_criterion = DistillationLoss(alpha=0.7, temperature=2.0)
    
    teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=config['LEARNING_RATE'])
    student_optimizer = torch.optim.Adam(student_model.parameters(), lr=config['LEARNING_RATE'])
    
    # Define schedulers (cosine annealing)
    teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        teacher_optimizer, 
        T_max=config['TEACHER_EPOCHS'],
        eta_min=config['MIN_LR']
    )
    
    student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        student_optimizer, 
        T_max=config['STUDENT_EPOCHS'],
        eta_min=config['MIN_LR']
    )
    
    # Train teacher model
    print(f"Training teacher model for {config['TEACHER_EPOCHS']} epochs...")
    best_teacher_dice = 0
    
    for epoch in range(config['TEACHER_EPOCHS']):
        print(f"\nTeacher Epoch {epoch+1}/{config['TEACHER_EPOCHS']}")
        
        train_loss = train_teacher_with_mixup_cutmix(
            teacher_model, 
            train_loader, 
            teacher_optimizer, 
            teacher_criterion, 
            device,
            use_mixup=True,
            mixup_alpha=0.2,
            use_cutmix=True, 
            cutmix_alpha=1.0,
            scheduler=None
        )
        
        # Validate teacher
        teacher_model.eval()
        val_loss = 0
        val_dice_scores = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating Teacher"):
                images = batch['image'].to(device)
                orig_images = batch['orig_image'].to(device)
                masks = batch['mask'].to(device)
                has_original = batch['has_original']
                image_types = batch['image_type']
                
                # Include Type 1 and Type 2 images for validation
                valid_samples = has_original.bool() | (image_types == 2)
                
                if not valid_samples.any():
                    continue
                    
                # Filter batch
                images = images[valid_samples]
                orig_images = orig_images[valid_samples]
                masks = masks[valid_samples]
                
                # Forward pass
                outputs = teacher_model(images, orig_images)
                loss = teacher_criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate Dice
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                batch_dice = (2 * (pred_masks * masks).sum()) / ((pred_masks + masks).sum() + 1e-6)
                val_dice_scores.append(batch_dice.item())
        
        if len(val_dice_scores) > 0:
            avg_val_loss = val_loss / len(val_dice_scores)
            avg_val_dice = sum(val_dice_scores) / len(val_dice_scores)
            print(f"Teacher Train Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
            
            teacher_scheduler.step()
            
            # Save best teacher model
            if avg_val_dice > best_teacher_dice:
                best_teacher_dice = avg_val_dice
                torch.save(teacher_model.state_dict(), config['TEACHER_MODEL_PATH'])
                print(f"Teacher model saved with Dice: {best_teacher_dice:.4f}")
        else:
            print("No valid samples for validation")
    
    # Load best teacher model
    print("Loading best teacher model...")
    teacher_model.load_state_dict(torch.load(config['TEACHER_MODEL_PATH']))
    
    # Train student model with knowledge distillation
    print(f"\nTraining student model for {config['STUDENT_EPOCHS']} epochs...")
    best_student_dice = 0
    
    for epoch in range(config['STUDENT_EPOCHS']):
        print(f"\nStudent Epoch {epoch+1}/{config['STUDENT_EPOCHS']}")
        
        train_loss = train_student_with_mixup_cutmix(
            student_model, 
            teacher_model, 
            train_loader, 
            student_optimizer, 
            student_criterion, 
            device,
            use_mixup=True,
            mixup_alpha=0.2,
            use_cutmix=True, 
            cutmix_alpha=1.0,
            scheduler=None
        )
        
        # Validate student
        student_model.eval()
        val_loss = 0
        val_dice_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating Student"):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = student_model(images)
                loss = F.binary_cross_entropy_with_logits(outputs, masks) + dice_loss(torch.sigmoid(outputs), masks)
                val_loss += loss.item()
                
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                batch_dice = (2 * (pred_masks * masks).sum()) / ((pred_masks + masks).sum() + 1e-6)
                val_dice_scores.append(batch_dice.item())
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = sum(val_dice_scores) / len(val_dice_scores)
        print(f"Student Train Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
        
        student_scheduler.step()
        
        if avg_val_dice > best_student_dice:
            best_student_dice = avg_val_dice
            torch.save(student_model.state_dict(), config['STUDENT_MODEL_PATH'])
            print(f"Student model saved with Dice: {best_student_dice:.4f}")
    
    print("Loading best student model...")
    student_model.load_state_dict(torch.load(config['STUDENT_MODEL_PATH']))
    
    return student_model
    
# Example usage:
# CONFIG = {
#     'TRAIN_IMG_DIR': '/home/cv-hacker/competition-data/train/train/images',
#     'TRAIN_MASK_DIR': '/home/cv-hacker/competition-data/train/train/masks',
#     'TRAIN_ORIG_DIR': '/home/cv-hacker/competition-data/train/train/originals',
#     'TEST_IMG_DIR': '/home/cv-hacker/competition-data/test/test/images',
#     'BATCH_SIZE': 32,
#     'TEACHER_EPOCHS': 10,
#     'STUDENT_EPOCHS': 20,
#     'LEARNING_RATE': 1e-4,
#     'MIN_LR': 1e-6,
#     'NUM_WORKERS': 4,
#     'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     'VALIDATION_SPLIT': 0.1,
#     'TEACHER_MODEL_PATH': 'best_teacher_model.pth',
#     'STUDENT_MODEL_PATH': 'best_student_model.pth',
#     'ENCODER': 'mit_b3',
# }
# student_model = train_teacher_student_models(CONFIG)
