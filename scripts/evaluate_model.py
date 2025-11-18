#!/usr/bin/env python3
"""
Evaluation script for trained models.

Usage:
    python scripts/evaluate_model.py --config configs/config.yaml --checkpoint outputs/experiment/checkpoints/checkpoint_21.pth
"""
import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.eval.evaluator import evaluate_checkpoint

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--test-data', 
        type=str, 
        default=None,
        help='Path to test dataset CSV (defaults to config processed_dir/dataset.csv with test split)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None,
        help='Output directory for evaluation results (defaults to outputs/evaluation/checkpoint_name)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default=None,
        help='Device override (cuda/cpu, defaults to config device)'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Adjust logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Test data: {args.test_data or 'Default (dataset.csv with test split)'}")
    logger.info(f"Output dir: {args.output_dir or 'Auto-generated'}")
    logger.info(f"Device: {args.device or 'From config'}")
    logger.info("=" * 60)
    
    try:
        # Run evaluation
        metrics = evaluate_checkpoint(
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            test_data_path=args.test_data,
            output_dir=args.output_dir,
            device=args.device
        )
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Final Metrics:")
        logger.info(f"  Accuracy:   {metrics['accuracy']:.4f}")
        logger.info(f"  Precision:  {metrics['precision']:.4f}")
        logger.info(f"  Recall:     {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:   {metrics['f1']:.4f}")
        logger.info(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
        logger.info(f"  Specificity: {metrics.get('specificity', 'N/A'):.4f}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()