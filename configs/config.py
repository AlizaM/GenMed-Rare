# config.py
import torch
from pathlib import Path

class Config:
    # Paths
    DATA_DIR = Path("data/interim")
    CSV_PATH = DATA_DIR / "filtered_data_entry.csv"
    TRAIN_VAL_DIR = DATA_DIR / "train_val"
    TEST_DIR = DATA_DIR / "test"
    OUTPUT_DIR = Path("outputs")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"
    
    # Classes for binary classification
    CLASS_COMMON = "Effusion"
    CLASS_RARE = "Fibrosis"
    
    # Model configuration
    MODEL_BACKBONE = "resnet50"  # Options: resnet18, resnet34, resnet50, resnet101
    PRETRAINED = True
    NUM_CLASSES = 2
    
    # Training configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Data split
    TRAIN_VAL_SPLIT = 0.8
    STRATIFIED = True
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Image configuration
    IMG_SIZE = (1024, 1024)
    
    # Augmentation parameters
    ROTATION_DEGREES = 10
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    GAUSSIAN_NOISE_STD = 0.01
    
    # Normalization (ImageNet stats)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Learning rate scheduler
    LR_SCHEDULER = "ReduceLROnPlateau"
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    
    # Checkpoint
    SAVE_BEST_ONLY = True
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        cls.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        cls.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        cls.LOG_DIR.mkdir(exist_ok=True, parents=True)