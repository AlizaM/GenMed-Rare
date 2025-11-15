#!/usr/bin/env python3
"""
Main training script for Effusion vs Fibrosis classification.

This script orchestrates the complete training pipeline:
1. Load configuration
2. Process data (if needed)
3. Create datasets and dataloaders
4. Initialize model
5. Setup training components (optimizer, scheduler, loss)
6. Train the model
"""

import sys
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import ConfigManager
from src.data.data_handler import DataHandler
from src.data.dataset import MedicalImageDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.classifier import create_model, count_parameters
from src.train.trainer import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def create_dataloaders(config):
    """
    Create training and validation dataloaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    
    # Create datasets
    train_dataset = MedicalImageDataset(
        csv_path=config.data.processed_csv,
        split='train',
        config=config,
        transform=train_transform
    )
    
    val_dataset = MedicalImageDataset(
        csv_path=config.data.processed_csv,
        split='val',
        config=config,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Log class distribution
    logger.info("Train class distribution:")
    for class_name, count in train_dataset.get_class_distribution().items():
        logger.info(f"  {class_name}: {count}")
    
    logger.info("Val class distribution:")
    for class_name, count in val_dataset.get_class_distribution().items():
        logger.info(f"  {class_name}: {count}")
    
    return train_loader, val_loader, train_dataset


def create_optimizer(model, config):
    """
    Create optimizer.
    
    Args:
        model: Model to optimize
        config: Configuration object
        
    Returns:
        Optimizer
    """
    optimizer_name = config.training.optimizer.lower()
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    logger.info(f"Optimizer: {optimizer_name}")
    logger.info(f"Learning rate: {config.training.lr}")
    logger.info(f"Weight decay: {config.training.weight_decay}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        config: Configuration object
        
    Returns:
        Scheduler or None
    """
    scheduler_name = config.training.scheduler.lower()
    
    if scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.training.scheduler_params.mode,
            factor=config.training.scheduler_params.factor,
            patience=config.training.scheduler_params.patience,
            min_lr=config.training.scheduler_params.min_lr,
            verbose=True
        )
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs
        )
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    logger.info(f"Scheduler: {scheduler_name}")
    
    return scheduler


def create_loss_function(config, train_dataset=None):
    """
    Create loss function.
    
    Args:
        config: Configuration object
        train_dataset: Training dataset (for class weights)
        
    Returns:
        Loss function
    """
    # For imbalanced datasets, use weighted cross entropy
    if train_dataset is not None:
        class_weights = train_dataset.get_class_weights()
        logger.info(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(get_device()))
    else:
        criterion = nn.CrossEntropyLoss()
    
    return criterion


def get_device():
    """Get device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train medical image classifier')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_effusion_fibrosis.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-processing',
        action='store_true',
        help='Skip data processing (use existing processed CSV)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Medical Image Classification Training")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = ConfigManager.from_yaml(args.config)
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Classes: {config.data.class_1} vs {config.data.class_2}")
    
    # Set random seed
    set_seed(config.experiment.seed)
    
    # Process data if needed
    if not args.skip_processing:
        logger.info("\n" + "=" * 80)
        logger.info("Data Processing")
        logger.info("=" * 80)
        handler = DataHandler(config)
        handler.process_data()
    else:
        logger.info("Skipping data processing (using existing processed CSV)")
    
    # Create dataloaders
    logger.info("\n" + "=" * 80)
    logger.info("Creating Dataloaders")
    logger.info("=" * 80)
    train_loader, val_loader, train_dataset = create_dataloaders(config)
    
    # Get device
    device = get_device()
    
    # Create model
    logger.info("\n" + "=" * 80)
    logger.info("Creating Model")
    logger.info("=" * 80)
    model = create_model(config)
    model = model.to(device)
    
    n_params = count_parameters(model)
    logger.info(f"Model has {n_params:,} trainable parameters")
    
    # Create optimizer
    logger.info("\n" + "=" * 80)
    logger.info("Creating Optimizer and Scheduler")
    logger.info("=" * 80)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss function
    criterion = create_loss_function(config, train_dataset)
    
    # Create trainer
    logger.info("\n" + "=" * 80)
    logger.info("Initializing Trainer")
    logger.info("=" * 80)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Training")
    logger.info("=" * 80)
    trainer.train()
    
    logger.info("\n" + "=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
