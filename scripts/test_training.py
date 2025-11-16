#!/usr/bin/env python3
"""
Quick training test script using small dataset.

Usage:
    python scripts/test_training.py
"""
import sys
import argparse
import logging
import torch
import numpy as np
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data.dataset import ChestXrayDataset
from src.models import create_model
from src.train import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def create_test_dataloaders(config, test_csv_path):
    """Create dataloaders using test dataset."""
    # Create datasets for train and val splits from test CSV
    train_dataset = ChestXrayDataset(test_csv_path, config, split='train')
    val_dataset = ChestXrayDataset(test_csv_path, config, split='val')
    
    logger.info(f"Train dataset: {len(train_dataset)} images")
    logger.info(f"Val dataset: {len(val_dataset)} images")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.hardware.pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.hardware.pin_memory
    )
    
    return train_loader, val_loader


def main():
    """Main testing pipeline."""
    parser = argparse.ArgumentParser(description='Test training pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_test.yaml',
        help='Path to test configuration YAML file'
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default='data/processed/effusion_fibrosis/dataset_test.csv',
        help='Path to test dataset CSV'
    )
    
    args = parser.parse_args()
    
    # Check if test dataset exists
    test_csv_path = Path(args.test_csv)
    if not test_csv_path.exists():
        logger.error("=" * 80)
        logger.error("TEST DATASET NOT FOUND")
        logger.error("=" * 80)
        logger.error(f"Test dataset CSV not found: {test_csv_path}")
        logger.error("Please create it first:")
        logger.error("  python scripts/create_test_dataset.py")
        return
    
    # Load configuration
    logger.info("=" * 80)
    logger.info("TEST TRAINING PIPELINE")
    logger.info("=" * 80)
    config = load_config(args.config)
    
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Description: {config.experiment.description}")
    logger.info(f"Using test dataset: {test_csv_path}")
    
    # Set random seed
    set_seed(config.experiment.seed)
    
    # Create output directories
    config.create_dirs()
    
    # Create dataloaders with test dataset
    logger.info("=" * 80)
    logger.info("LOADING TEST DATA")
    logger.info("=" * 80)
    train_loader, val_loader = create_test_dataloaders(config, test_csv_path)
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("=" * 80)
    logger.info("CREATING MODEL")
    logger.info("=" * 80)
    model = create_model(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Train
    logger.info("=" * 80)
    logger.info("STARTING TRAINING TEST")
    logger.info("=" * 80)
    trainer.train()
    
    # Verify checkpoints were saved
    logger.info("=" * 80)
    logger.info("VERIFICATION")
    logger.info("=" * 80)
    
    checkpoint_dir = Path(config.training.checkpoint_dir)
    best_checkpoint = checkpoint_dir / "best_checkpoint.pth"
    latest_checkpoint = checkpoint_dir / "latest_checkpoint.pth"
    
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Best checkpoint exists: {best_checkpoint.exists()}")
    logger.info(f"Latest checkpoint exists: {latest_checkpoint.exists()}")
    
    if best_checkpoint.exists() and latest_checkpoint.exists():
        logger.info("\n✓ Training test PASSED - All checkpoints saved successfully!")
    else:
        logger.error("\n✗ Training test FAILED - Missing checkpoints!")
    
    logger.info("=" * 80)
    logger.info(f"TensorBoard logs: {config.training.log_dir}")
    logger.info("\nTo view TensorBoard:")
    logger.info(f"  tensorboard --logdir={config.training.log_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
