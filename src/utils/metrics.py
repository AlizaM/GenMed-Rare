"""
Training utilities including metrics and evaluation functions.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate and track training metrics."""
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, probs: torch.Tensor = None):
        """
        Update metrics with batch results.
        
        Args:
            preds: Predicted class indices (B,)
            targets: Ground truth labels (B,)
            probs: Class probabilities (B, num_classes), optional
        """
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        if probs is not None:
            self.probabilities.extend(probs.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric names to values
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(targets, preds)
        
        # Precision, Recall, F1
        metrics['precision'] = precision_score(targets, preds, average='binary', zero_division=0)
        metrics['recall'] = recall_score(targets, preds, average='binary', zero_division=0)
        metrics['f1_score'] = f1_score(targets, preds, average='binary', zero_division=0)
        
        # AUC-ROC (if probabilities available)
        if len(self.probabilities) > 0:
            probs = np.array(self.probabilities)
            # For binary classification, use probability of positive class
            if self.num_classes == 2:
                probs_pos = probs[:, 1]
                metrics['auc_roc'] = roc_auc_score(targets, probs_pos)
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
            Confusion matrix
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        return confusion_matrix(targets, preds)


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self, name: str):
        """
        Initialize average meter.
        
        Args:
            name: Name of the metric
        """
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics.
        
        Args:
            val: New value
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}: {self.avg:.4f}"


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy from outputs and targets.
    
    Args:
        outputs: Model outputs (logits) of shape (B, num_classes)
        targets: Ground truth labels of shape (B,)
        
    Returns:
        Accuracy as float
    """
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy


def get_predictions_and_probabilities(
    outputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get predicted classes and probabilities from model outputs.
    
    Args:
        outputs: Model outputs (logits) of shape (B, num_classes)
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    # Apply softmax to get probabilities
    probs = torch.softmax(outputs, dim=1)
    
    # Get predicted classes
    preds = torch.argmax(outputs, dim=1)
    
    return preds, probs


if __name__ == '__main__':
    # Example usage
    import torch
    
    # Simulate some predictions
    batch_size = 32
    num_classes = 2
    
    # Random outputs and targets
    outputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Calculate accuracy
    acc = calculate_accuracy(outputs, targets)
    print(f"Accuracy: {acc:.4f}")
    
    # Get predictions and probabilities
    preds, probs = get_predictions_and_probabilities(outputs)
    print(f"Predictions shape: {preds.shape}")
    print(f"Probabilities shape: {probs.shape}")
    
    # Use metrics calculator
    metrics_calc = MetricsCalculator(num_classes=num_classes)
    metrics_calc.update(preds, targets, probs)
    
    # Compute metrics
    metrics = metrics_calc.compute()
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Confusion matrix
    cm = metrics_calc.get_confusion_matrix()
    print(f"\nConfusion Matrix:\n{cm}")
