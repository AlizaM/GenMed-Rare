"""Training utilities and trainer class."""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from src.config import Config

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and compute classification metrics."""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset metric accumulators."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.losses = []
    
    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        probs: torch.Tensor,
        loss: float
    ):
        """
        Update metrics with batch results.
        
        Args:
            preds: Predicted classes [batch]
            labels: True labels [batch]
            probs: Class probabilities [batch, num_classes]
            loss: Batch loss
        """
        self.predictions.extend(preds.detach().cpu().numpy())
        self.labels.extend(labels.detach().cpu().numpy())
        self.probabilities.extend(probs.detach().cpu().numpy())
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric name -> value
        """
        preds = np.array(self.predictions)
        labels = np.array(self.labels)
        probs = np.array(self.probabilities)
        
        metrics = {
            'loss': np.mean(self.losses),
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='binary', zero_division=0),
            'recall': recall_score(labels, preds, average='binary', zero_division=0),
            'f1': f1_score(labels, preds, average='binary', zero_division=0),
        }
        
        # AUC-ROC (requires probabilities)
        try:
            metrics['auc_roc'] = roc_auc_score(labels, probs[:, 1])
        except ValueError:
            metrics['auc_roc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        mode: str = 'min',
        delta: float = 0.0
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            mode: 'min' for loss, 'max' for accuracy/f1
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.delta)
        else:  # max
            improved = score > (self.best_score + self.delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
        
        return False


class Trainer:
    """Training orchestrator."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Config
    ):
        """
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Configuration object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device
        self.device = torch.device(config.hardware.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker(config)
        
        # Early stopping
        mode = 'min' if 'loss' in config.training.early_stopping_metric else 'max'
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode=mode
        ) if config.training.early_stopping else None
        
        # TensorBoard writer
        self.checkpoint_dir, self.log_dir = config.create_dirs()
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Best metric tracking
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.training.use_amp else None
        
        # Training history for plotting
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'train_f1': [],
            'train_recall': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'val_recall': []
        }
        
        logger.info("Trainer initialized")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.training.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.training.scheduler.lower() == 'reduce_lr_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.training.lr_patience,
                factor=self.config.training.lr_factor,
                min_lr=self.config.training.lr_min
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.metrics_tracker.reset()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                # Mixed precision training
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Compute predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Update metrics
            self.metrics_tracker.update(preds, labels, probs, loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to TensorBoard
            if batch_idx % self.config.training.log_interval == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        # Compute epoch metrics
        metrics = self.metrics_tracker.compute()
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.metrics_tracker.reset()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Compute predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Update metrics
                self.metrics_tracker.update(preds, labels, probs, loss.item())
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        metrics = self.metrics_tracker.compute()
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'experiment_name': self.config.experiment.name,
                'class_positive': self.config.data.class_positive,
                'class_negative': self.config.data.class_negative,
                'model_variant': self.config.model.variant,
            }
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / f'checkpoint_{epoch}.pth'
        torch.save(checkpoint, latest_path)
        logger.info(f"Saved latest checkpoint (epoch {epoch}) to {latest_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Saved BEST checkpoint (epoch {epoch}) to {best_path}")
    
    def save_training_graphs(self):
        """
        Save training graphs for loss, accuracy, and F1 score.
        """
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - {self.config.experiment.name}', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score plot
        axes[1, 0].plot(epochs, self.history['train_f1'], 'b-', label='Train F1', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_f1'], 'r-', label='Validation F1', linewidth=2)
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall plot
        axes[1, 1].plot(epochs, self.history['train_recall'], 'b-', label='Train Recall', linewidth=2)
        axes[1, 1].plot(epochs, self.history['val_recall'], 'r-', label='Validation Recall', linewidth=2)
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        graphs_path = self.checkpoint_dir / 'training_graphs.png'
        plt.savefig(graphs_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        logger.info(f"Training graphs saved to {graphs_path}")
        
        # Also save a summary plot with just the most important metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Training Summary - {self.config.experiment.name}', fontsize=16)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_accuracy'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[2].plot(epochs, self.history['train_f1'], 'b-', label='Train', linewidth=2)
        axes[2].plot(epochs, self.history['val_f1'], 'r-', label='Validation', linewidth=2)
        axes[2].set_title('F1 Score')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the summary plot
        summary_path = self.checkpoint_dir / 'training_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training summary saved to {summary_path}")

    def log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """
        Log metrics to TensorBoard and console.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Log to TensorBoard
        for metric_name, value in train_metrics.items():
            if metric_name != 'confusion_matrix':
                self.writer.add_scalar(f'train/{metric_name}', value, epoch)
        
        for metric_name, value in val_metrics.items():
            if metric_name != 'confusion_matrix':
                self.writer.add_scalar(f'val/{metric_name}', value, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', current_lr, epoch)
        
        # Console logging with all requested metrics
        logger.info(f"\nEpoch {epoch} Results:")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"F1: {train_metrics['f1']:.4f}, "
                   f"Recall: {train_metrics['recall']:.4f}, "
                   f"AUC: {train_metrics['auc_roc']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}, "
                   f"Recall: {val_metrics['recall']:.4f}, "
                   f"AUC: {val_metrics['auc_roc']:.4f}")
        
        # Store metrics in history for plotting
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_accuracy'].append(train_metrics['accuracy'])
        self.history['train_f1'].append(train_metrics['f1'])
        self.history['train_recall'].append(train_metrics['recall'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['val_f1'].append(val_metrics['f1'])
        self.history['val_recall'].append(val_metrics['recall'])
    
    def train(self):
        """Main training loop."""
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        logger.info(f"Experiment: {self.config.experiment.name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.config.training.num_epochs}")
        logger.info(f"Batch size: {self.config.training.batch_size}")
        logger.info(f"Learning rate: {self.config.training.learning_rate}")
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Log metrics
            self.log_metrics(epoch, train_metrics, val_metrics)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['loss'])
            
            # Get primary metric for model selection
            primary_metric = self.config.metrics.primary_metric
            primary_mode = self.config.metrics.primary_mode
            current_metric = val_metrics.get(primary_metric, val_metrics['loss'])
            
            # Check if best model
            is_best = False
            if primary_mode == 'max':
                is_best = current_metric > self.best_metric
            else:
                is_best = current_metric < self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
            
            # Save checkpoint
            if self.config.training.save_checkpoints:
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Early stopping
            if self.early_stopping is not None:
                stop_metric = val_metrics.get(
                    self.config.training.early_stopping_metric,
                    val_metrics['loss']
                )
                if self.early_stopping(stop_metric):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Save graphs every 10 epochs and at the end
            if epoch % 10 == 0 or epoch == self.config.training.num_epochs:
                self.save_training_graphs()
        
        # Final save of training graphs
        self.save_training_graphs()
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info(f"Best {self.config.metrics.primary_metric}: {self.best_metric:.4f} (epoch {self.best_epoch})")
        logger.info(f"Training graphs saved to: {self.checkpoint_dir}")
        logger.info("=" * 80)
        
        self.writer.close()
