"""Evaluation module for model inference and metrics computation."""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union
from tqdm import tqdm
import logging
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

from src.config import load_config, Config
from src.models.classifier import SwinClassifier
from src.data.dataset import ChestXrayDataset

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluation orchestrator."""
    
    def __init__(
        self,
        config: Config,
        checkpoint_path: Union[str, Path],
        test_data_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            config: Configuration object
            checkpoint_path: Path to model checkpoint
            test_data_path: Optional path to test dataset CSV (defaults to config dataset)
            output_dir: Directory to save evaluation results (defaults to evaluation subdir)
            device: Device override (defaults to config device)
        """
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        
        # Set device (allow override)
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(config.hardware.device)
        
        # Set test data path
        if test_data_path is None:
            self.test_data_path = config.data.processed_dir / "dataset.csv"
        else:
            self.test_data_path = Path(test_data_path)
        
        # Set output directory
        if output_dir is None:
            self.output_dir = Path("outputs") / config.experiment.name / "evaluation" / self.checkpoint_path.stem
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and data
        self.model = self._load_model()
        self.test_loader = self._create_test_loader()
        
        # Results storage
        self.results = {
            'predictions': [],
            'labels': [],
            'probabilities': [],
            'metrics': {},
            'confusion_matrix': None,
            'classification_report': None
        }
        
        logger.info(f"ModelEvaluator initialized")
        logger.info(f"Checkpoint: {self.checkpoint_path}")
        logger.info(f"Test data: {self.test_data_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Test samples: {len(self.test_loader.dataset)}")
    
    def _load_model(self) -> nn.Module:
        """Load model with checkpoint weights."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")
        
        # Initialize model
        model = SwinClassifier(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Log checkpoint info
        if 'epoch' in checkpoint:
            logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            logger.info(f"Checkpoint validation metrics: {checkpoint['metrics']}")
        
        return model
    
    def _create_test_loader(self) -> torch.utils.data.DataLoader:
        """Create test data loader."""
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.test_data_path}")
        
        logger.info(f"Creating test dataset from: {self.test_data_path}")
        
        test_dataset = ChestXrayDataset(
            csv_path=self.test_data_path,
            config=self.config,
            split='test'
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.hardware.pin_memory
        )
        
        logger.info(f"Test dataset created with {len(test_dataset)} samples")
        return test_loader
    
    @torch.no_grad()
    def run_inference(self) -> Dict[str, np.ndarray]:
        """
        Run inference on test data.
        
        Returns:
            Dictionary containing predictions, labels, and probabilities
        """
        logger.info("Starting inference on test data...")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        # Progress bar
        pbar = tqdm(self.test_loader, desc='Inference')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            with torch.cuda.amp.autocast() if self.config.training.use_amp else torch.no_grad():
                outputs = self.model(images)
            
            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update progress bar
            batch_acc = (predictions == labels).float().mean().item()
            pbar.set_postfix({'batch_acc': f'{batch_acc:.3f}'})
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)
        
        # Store results
        self.results['predictions'] = predictions
        self.results['labels'] = labels
        self.results['probabilities'] = probabilities
        
        logger.info(f"Inference completed. Total samples: {len(labels)}")
        
        return {
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Returns:
            Dictionary of metric name -> value
        """
        preds = self.results['predictions']
        labels = self.results['labels']
        probs = self.results['probabilities']
        
        logger.info("Computing evaluation metrics...")
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='binary', zero_division=0),
            'recall': recall_score(labels, preds, average='binary', zero_division=0),
            'f1': f1_score(labels, preds, average='binary', zero_division=0),
        }
        
        # AUC-ROC
        try:
            metrics['auc_roc'] = roc_auc_score(labels, probs[:, 1])
        except ValueError as e:
            logger.warning(f"Could not compute AUC-ROC: {e}")
            metrics['auc_roc'] = 0.0
        
        # Confusion matrix and derived metrics
        cm = confusion_matrix(labels, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
        
        # Per-class metrics
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        
        for i, (prec, rec, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            class_name = self.config.data.class_negative if i == 0 else self.config.data.class_positive
            metrics[f'precision_{class_name}'] = prec
            metrics[f'recall_{class_name}'] = rec
            metrics[f'f1_{class_name}'] = f1
        
        # Store metrics and confusion matrix
        self.results['metrics'] = metrics
        self.results['confusion_matrix'] = cm.tolist()
        
        # Classification report
        class_names = [self.config.data.class_negative, self.config.data.class_positive]
        self.results['classification_report'] = classification_report(
            labels, preds, target_names=class_names, output_dict=True
        )
        
        logger.info("Metrics computed successfully")
        return metrics
    
    def save_results(self):
        """Save evaluation results to files."""
        logger.info("Saving evaluation results...")
        
        # Save metrics as JSON
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': self.results['metrics'],
                'confusion_matrix': self.results['confusion_matrix'],
                'classification_report': self.results['classification_report'],
                'config': {
                    'experiment_name': self.config.experiment.name,
                    'checkpoint_path': str(self.checkpoint_path),
                    'test_data_path': str(self.test_data_path),
                    'class_positive': self.config.data.class_positive,
                    'class_negative': self.config.data.class_negative,
                    'device': str(self.device),
                    'test_samples': len(self.results['labels'])
                }
            }, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'true_label': self.results['labels'],
            'predicted_label': self.results['predictions'],
            'prob_negative': self.results['probabilities'][:, 0],
            'prob_positive': self.results['probabilities'][:, 1],
            'correct': self.results['labels'] == self.results['predictions']
        })
        
        predictions_path = self.output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")
    
    def plot_confusion_matrix(self):
        """Create and save confusion matrix plot."""
        cm = np.array(self.results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.config.experiment.name}')
        plt.colorbar(im)
        
        # Set labels
        classes = [self.config.data.class_negative, self.config.data.class_positive]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        cm_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {cm_path}")
    
    def plot_roc_curve(self):
        """Create and save ROC curve plot."""
        if self.results['metrics']['auc_roc'] == 0.0:
            logger.warning("Skipping ROC curve due to invalid AUC-ROC")
            return
        
        fpr, tpr, _ = roc_curve(self.results['labels'], self.results['probabilities'][:, 1])
        auc_score = self.results['metrics']['auc_roc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.config.experiment.name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        roc_path = self.output_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {roc_path}")
    
    def plot_precision_recall_curve(self):
        """Create and save Precision-Recall curve plot."""
        precision, recall, _ = precision_recall_curve(
            self.results['labels'], 
            self.results['probabilities'][:, 1]
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.config.experiment.name}')
        plt.grid(True, alpha=0.3)
        
        # Add metrics annotations
        f1_score_val = self.results['metrics']['f1']
        auc_score = self.results['metrics']['auc_roc']
        plt.text(0.05, 0.95, f'F1 Score: {f1_score_val:.3f}\nAUC-ROC: {auc_score:.3f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
                verticalalignment='top')
        
        plt.tight_layout()
        pr_path = self.output_dir / "precision_recall_curve.png"
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-Recall curve saved to {pr_path}")
    
    def create_summary_report(self):
        """Create a summary report with all metrics and visualizations."""
        logger.info("Creating evaluation summary report...")
        
        metrics = self.results['metrics']
        cm = np.array(self.results['confusion_matrix'])
        
        # Create summary text
        summary = f"""EVALUATION SUMMARY
==================

Experiment: {self.config.experiment.name}
Checkpoint: {self.checkpoint_path.name}
Test Data: {self.test_data_path}
Classes: {self.config.data.class_negative} (0) vs {self.config.data.class_positive} (1)
Test Samples: {len(self.results['labels'])}
Device: {self.device}

PERFORMANCE METRICS
===================
Accuracy:     {metrics['accuracy']:.4f}
Precision:    {metrics['precision']:.4f}
Recall:       {metrics['recall']:.4f}
F1 Score:     {metrics['f1']:.4f}
AUC-ROC:      {metrics['auc_roc']:.4f}
Specificity:  {metrics.get('specificity', 'N/A'):.4f}
Sensitivity:  {metrics.get('sensitivity', 'N/A'):.4f}

PER-CLASS METRICS
=================
{self.config.data.class_negative} (Class 0):
  Precision: {metrics.get(f'precision_{self.config.data.class_negative}', 'N/A'):.4f}
  Recall:    {metrics.get(f'recall_{self.config.data.class_negative}', 'N/A'):.4f}
  F1:        {metrics.get(f'f1_{self.config.data.class_negative}', 'N/A'):.4f}

{self.config.data.class_positive} (Class 1):
  Precision: {metrics.get(f'precision_{self.config.data.class_positive}', 'N/A'):.4f}
  Recall:    {metrics.get(f'recall_{self.config.data.class_positive}', 'N/A'):.4f}
  F1:        {metrics.get(f'f1_{self.config.data.class_positive}', 'N/A'):.4f}

CONFUSION MATRIX
================
                    Predicted
                {self.config.data.class_negative:>8} {self.config.data.class_positive:>8}
True {self.config.data.class_negative:>8}      {cm[0,0]:>4}      {cm[0,1]:>4}
     {self.config.data.class_positive:>8}      {cm[1,0]:>4}      {cm[1,1]:>4}

FILES GENERATED
===============
- metrics.json: Complete metrics and configuration
- predictions.csv: Per-sample predictions and probabilities  
- confusion_matrix.png: Confusion matrix visualization
- roc_curve.png: ROC curve plot
- precision_recall_curve.png: Precision-Recall curve
- evaluation_summary.txt: This summary report
"""
        
        # Save summary
        summary_path = self.output_dir / "evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Summary report saved to {summary_path}")
        
        # Print key metrics to console
        logger.info("EVALUATION COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        logger.info("=" * 50)
        logger.info(f"Results saved to: {self.output_dir}")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run complete evaluation pipeline.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting complete evaluation pipeline...")
        
        # Run inference
        self.run_inference()
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        # Save results
        self.save_results()
        
        # Create visualizations
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        
        # Create summary report
        self.create_summary_report()
        
        logger.info("Evaluation pipeline completed successfully!")
        
        return metrics


def evaluate_checkpoint(
    config_path: str,
    checkpoint_path: str,
    test_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    Convenience function to evaluate a checkpoint.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint
        test_data_path: Optional path to test dataset CSV
        output_dir: Optional output directory for results
        device: Optional device override
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        config=config,
        checkpoint_path=checkpoint_path,
        test_data_path=test_data_path,
        output_dir=output_dir,
        device=device
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    return metrics