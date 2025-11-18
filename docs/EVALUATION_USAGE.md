"""
Evaluation Module Usage Examples

This module provides comprehensive evaluation capabilities for trained models.
It can run inference on test data, compute metrics, and generate visualizations.
"""

# Example 1: Basic evaluation using the convenience function
from src.eval.evaluator import evaluate_checkpoint

def example_basic_evaluation():
    """Basic evaluation example."""
    
    # Evaluate a specific checkpoint
    metrics = evaluate_checkpoint(
        config_path="configs/config.yaml",
        checkpoint_path="outputs/effusion_vs_fibrosis_baseline/checkpoints/effusion_vs_fibrosis_50_epochs_baseline/checkpoint_21.pth"
    )
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")


# Example 2: Advanced evaluation with custom parameters  
from src.eval.evaluator import ModelEvaluator
from src.config import load_config

def example_advanced_evaluation():
    """Advanced evaluation with more control."""
    
    # Load config
    config = load_config("configs/config.yaml")
    
    # Create evaluator with custom settings
    evaluator = ModelEvaluator(
        config=config,
        checkpoint_path="outputs/effusion_vs_fibrosis_baseline/checkpoints/effusion_vs_fibrosis_50_epochs_baseline/checkpoint_21.pth",
        output_dir="custom_evaluation_results",  # Custom output directory
        device="cuda"  # Force GPU usage
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Access detailed results
    print("Confusion Matrix:", evaluator.results['confusion_matrix'])
    print("Per-class metrics:", {k: v for k, v in metrics.items() if 'class' in k})


# Command Line Usage Examples:

"""
1. Basic evaluation:
python scripts/evaluate_model.py \
    --config configs/config.yaml \
    --checkpoint outputs/effusion_vs_fibrosis_baseline/checkpoints/effusion_vs_fibrosis_50_epochs_baseline/checkpoint_21.pth

2. Evaluation with custom output directory:
python scripts/evaluate_model.py \
    --config configs/config.yaml \
    --checkpoint outputs/effusion_vs_fibrosis_baseline/checkpoints/effusion_vs_fibrosis_50_epochs_baseline/checkpoint_21.pth \
    --output-dir my_evaluation_results

3. Evaluation with CPU override:
python scripts/evaluate_model.py \
    --config configs/config.yaml \
    --checkpoint outputs/effusion_vs_fibrosis_baseline/checkpoints/effusion_vs_fibrosis_50_epochs_baseline/checkpoint_21.pth \
    --device cpu

4. Evaluation with custom test data:
python scripts/evaluate_model.py \
    --config configs/config.yaml \
    --checkpoint outputs/effusion_vs_fibrosis_baseline/checkpoints/effusion_vs_fibrosis_50_epochs_baseline/checkpoint_21.pth \
    --test-data path/to/custom_test_data.csv

5. Verbose evaluation:
python scripts/evaluate_model.py \
    --config configs/config.yaml \
    --checkpoint outputs/effusion_vs_fibrosis_baseline/checkpoints/effusion_vs_fibrosis_50_epochs_baseline/checkpoint_21.pth \
    --verbose
"""

# Output Files Generated:
"""
The evaluation module generates the following files:

1. metrics.json - Complete metrics, confusion matrix, and configuration
2. predictions.csv - Per-sample predictions with probabilities
3. confusion_matrix.png - Confusion matrix heatmap visualization  
4. roc_curve.png - ROC curve plot with AUC score
5. precision_recall_curve.png - Precision-Recall curve
6. evaluation_summary.txt - Human-readable summary report

All files are saved to: data/processed/evaluation/{checkpoint_name}/
(or custom output directory if specified)
"""