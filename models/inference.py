## Inference
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
#from teacher_student import StudentModel 
import random

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

CONFIG = {
    'TRAIN_IMG_DIR': '/home/cv-hacker/competition-data/train/train/images',
    'TRAIN_MASK_DIR': '/home/cv-hacker/competition-data/train/train/masks',
    'TEST_IMG_DIR': '/home/cv-hacker/competition-data/test/test/images',
    'BATCH_SIZE': 128,
    'EPOCHS': 20,
    'LEARNING_RATE': 1e-4,
    'NUM_WORKERS': 4,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'VALIDATION_SPLIT': 0.1,
    'MODEL_PATH': 'swa_model.pth',
    'SUBMISSION_CSV': 'swa_model.csv',
    # Advanced augmentation parameters
    'USE_MIXUP': True,
    'MIXUP_ALPHA': 0.2,
    'USE_CUTMIX': True,
    'CUTMIX_ALPHA': 1.0,
    'USE_TTA': True,
    'TTA_AUGMENTATIONS': 4,  # Number of augmentations to use in TTA
    # Learning rate scheduler parameters
    'LR_SCHEDULER': 'cosine',  # Options: 'cosine', 'reduce_on_plateau', 'one_cycle'
    'MIN_LR': 1e-6,
    'WARMUP_EPOCHS': 2,
}

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Dataset class
class ManipulationDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, is_test=False, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_test = is_test
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform
        
    def __len__(self):  
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare mask if available (training/validation)
        if not self.is_test and self.mask_dir:
            mask_path = os.path.join(self.mask_dir, img_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.float32)
            
            # Apply augmentations if specified
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                
                # Add channel dimension to mask if it doesn't have one
                # With ToTensorV2, mask should already be [H, W] tensor
                mask = mask.unsqueeze(0)  # Add channel dimension to make it [1, H, W]
            else:
                # Default normalization if no transform
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image.transpose(2, 0, 1))
                mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        else:
            # Test set processing
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image.transpose(2, 0, 1))
            
            # Dummy mask for test set
            mask = torch.zeros((1, 256, 256))
            
        return {
            'image': image,
            'mask': mask,
            'image_id': img_name.split('.')[0]
        }

# Student model (takes only manipulated images)
class StudentModel(nn.Module):
    def __init__(self, encoder_name="mit_b3"):
        super(StudentModel, self).__init__()
        # Standard model with single image input
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        
    def forward(self, x):
        return self.model(x)

# RLE encoding function for submission
def mask2rle(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)



# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    dice_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            # Calculate Dice coefficient
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (pred_masks * masks).sum()
            dice = (2. * intersection) / (pred_masks.sum() + masks.sum() + 1e-6)
            dice_scores.append(dice.item())
    
    return val_loss / len(dataloader), sum(dice_scores) / len(dice_scores)

# Standard inference function
def inference(model, dataloader, device):
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images = batch['image'].to(device)
            image_ids = batch['image_id']
            
            # Forward pass
            outputs = model(images)
            outputs = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(np.uint8)
            
            # Convert predictions to RLE format
            for i, image_id in enumerate(image_ids):
                rle = mask2rle(outputs[i][0])
                results.append([image_id, rle])
    
    return results

# Test Time Augmentation inference function
def inference_with_tta(model, dataloader, device, num_aug=4):
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference with TTA"):
            images = batch['image'].to(device)
            image_ids = batch['image_id']
            
            # Original prediction
            outputs_orig = model(images)
            
            # Horizontal flip
            images_hflip = torch.flip(images, dims=[-1])
            outputs_hflip = model(images_hflip)
            outputs_hflip = torch.flip(outputs_hflip, dims=[-1])
            
            # Vertical flip
            images_vflip = torch.flip(images, dims=[-2])
            outputs_vflip = model(images_vflip)
            outputs_vflip = torch.flip(outputs_vflip, dims=[-2])
            
            # 90-degree rotation
            images_rot90 = torch.rot90(images, k=1, dims=[-2, -1])
            outputs_rot90 = model(images_rot90)
            outputs_rot90 = torch.rot90(outputs_rot90, k=3, dims=[-2, -1])
            
            # Average predictions based on how many augmentations were requested
            if num_aug == 1:
                # Only use original
                outputs_combined = outputs_orig
            elif num_aug == 2:
                # Original + Horizontal flip
                outputs_combined = (outputs_orig + outputs_hflip) / 2.0
            elif num_aug == 3:
                # Original + Horizontal flip + Vertical flip
                outputs_combined = (outputs_orig + outputs_hflip + outputs_vflip) / 3.0
            else:
                # All four augmentations
                outputs_combined = (outputs_orig + outputs_hflip + outputs_vflip + outputs_rot90) / 4.0
            
            # Apply threshold
            outputs = (torch.sigmoid(outputs_combined) > 0.5).cpu().numpy().astype(np.uint8)
            
            # Convert predictions to RLE format
            for i, image_id in enumerate(image_ids):
                rle = mask2rle(outputs[i][0])
                results.append([image_id, rle])
    
    return results

def main(model_path, use_tta=True):
    print(f"Using device: {CONFIG['DEVICE']}")
    
    # Create test dataset
    test_dataset = ManipulationDataset(
        img_dir=CONFIG['TEST_IMG_DIR'],
        is_test=True,
        transform=val_transform
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=CONFIG['NUM_WORKERS']
    )
    
    # Initialize student model instead of the original model
    model = StudentModel(encoder_name="mit_b3")
    
    # Load the trained student model
    model.load_state_dict(torch.load("swa_model.pth"))
    model = model.to(CONFIG['DEVICE'])
    
    # Generate predictions with or without TTA
    if use_tta or CONFIG['USE_TTA']:
        print("Generating predictions with TTA...")
        submission_csv = CONFIG['SUBMISSION_CSV'].replace('.csv', '_tta.csv')
        submission_results = inference_with_tta(
            model, 
            test_loader, 
            CONFIG['DEVICE'],
            num_aug=CONFIG['TTA_AUGMENTATIONS']
        )
    else:
        print("Generating predictions without TTA...")
        submission_csv = CONFIG['SUBMISSION_CSV']
        submission_results = inference(model, test_loader, CONFIG['DEVICE'])
    
    # Create submission file
    print(f"Creating submission file: {submission_csv}")
    submission_df = pd.DataFrame(submission_results, columns=['ImageId', 'EncodedPixels'])
    submission_df.to_csv(submission_csv, index=False)
    
    print("Done!")

if __name__ == "__main__":
    main(model_path='swa_model.pth')