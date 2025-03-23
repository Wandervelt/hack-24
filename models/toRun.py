import torch
#from teacher_student import train_teacher_student_models
from teacher_student_boundary_loss import train_teacher_student_models
#from teacher_student_convnext import train_teacher_student_models

# Configuration
CONFIG = {
    'TRAIN_IMG_DIR': '/home/cv-hacker/competition-data/train/train/images',
    'TRAIN_MASK_DIR': '/home/cv-hacker/competition-data/train/train/masks',
    'TRAIN_ORIG_DIR': '/home/cv-hacker/competition-data/train/train/originals',
    'TEST_IMG_DIR': '/home/cv-hacker/competition-data/test/test/images',
    'BATCH_SIZE': 128,  # Adjust based on your GPU memory
    'TEACHER_EPOCHS': 30,  # You can use fewer epochs since you have limited time
    'STUDENT_EPOCHS': 40,
    'LEARNING_RATE': 1e-4,
    'MIN_LR': 1e-6,
    'NUM_WORKERS': 6,
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'VALIDATION_SPLIT': 0.2,
    'TEACHER_MODEL_PATH': 'best_teacher_model_bound.pth',
    'STUDENT_MODEL_PATH': 'best_student_model_bound.pth',
    'ENCODER': 'mit_b3',  # Use the same encoder as your original model
    #'ENCODER': 'convnext_base',  # Use the same encoder as your original model

}

# Train the teacher and student models
student_model = train_teacher_student_models(CONFIG)

print("Training complete! You can now use the student model for inference.")