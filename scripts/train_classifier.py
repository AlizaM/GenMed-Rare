#!/usr/bin/env python3
"""
Main training script for binary classification.

Usage:
    python scripts/train_classifier.py --config configs/config.yaml
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
from src.data.dataset import create_dataloaders
from src.models import create_model
from src.train import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def main():
    """Main training pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train binary classification model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info("=" * 80)
    logger.info("LOADING CONFIGURATION")
    logger.info("=" * 80)
    config = load_config(args.config)
    
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Description: {config.experiment.description}")
    logger.info(f"Binary classification: {config.data.class_negative} (0) vs {config.data.class_positive} (1)")
    logger.info(f"Model: {config.model.variant}")
    
    # Set random seed
    set_seed(config.experiment.seed)
    
    # Create output directories
    config.create_dirs()
    
    # Check if preprocessing has been done
    dataset_csv = config.data.processed_dir / 'dataset.csv'
    
    if not dataset_csv.exists():
        logger.error("=" * 80)
        logger.error("PREPROCESSING REQUIRED")
        logger.error("=" * 80)
        logger.error("Unified dataset CSV not found. Please run preprocessing first:")
        logger.error(f"  python src/data/preprocess.py --config {args.config}")
        return
    
    # Create dataloaders
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Log class distribution
    import pandas as pd
    df = pd.read_csv(dataset_csv)
    logger.info("-" * 40)
    logger.info("CLASS DISTRIBUTION:")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if len(split_df) == 0:
            continue
        label_counts = split_df['label'].value_counts().sort_index()
        neg_count = label_counts.get(0, 0)
        pos_count = label_counts.get(1, 0)
        logger.info(f"  {split.upper():5s}: {config.data.class_negative}={neg_count}, {config.data.class_positive}={pos_count}, Total={len(split_df)}")

    # Log synthetic breakdown if available
    train_df = df[df['split'] == 'train']
    if 'is_synthetic' in train_df.columns:
        synthetic_count = train_df['is_synthetic'].sum()
        real_count = len(train_df) - synthetic_count
        logger.info(f"  TRAIN source: Real={real_count}, Synthetic={synthetic_count}")
    elif 'source' in train_df.columns:
        synthetic_count = (train_df['source'] == 'synthetic').sum()
        real_count = len(train_df) - synthetic_count
        logger.info(f"  TRAIN source: Real={real_count}, Synthetic={synthetic_count}")
    logger.info("-" * 40)
    
    # Create model
    logger.info("=" * 80)
    logger.info("CREATING MODEL")
    logger.info("=" * 80)
    model = create_model(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        from src.models import load_checkpoint
        device = torch.device(config.hardware.device)
        checkpoint = load_checkpoint(model, args.resume, device, load_optimizer=False)
        logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Train
    trainer.train()
    
    # Print final information
    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    logger.info(f"TensorBoard logs: {trainer.log_dir}")
    logger.info("\nTo view TensorBoard:")
    logger.info(f"  tensorboard --logdir={config.training.log_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
