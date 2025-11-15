"""
Training loop and trainer class.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Optional

from src.utils.metrics import MetricsCalculator, AverageMeter, calculate_accuracy, get_predictions_and_probabilities

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config,
        device: torch.device
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            config: Configuration object
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0
        
        # Setup logging
        self.setup_logging()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.experiment.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Trainer initialized")
        logger.info(f"Device: {device}")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
    
    def setup_logging(self):
        """Setup TensorBoard logging."""
        if self.config.logging.use_tensorboard:
            log_dir = Path(self.config.experiment.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard logging enabled: {log_dir}")
        else:
            self.writer = None
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Metrics tracking
        loss_meter = AverageMeter('loss')
        metrics_calc = MetricsCalculator(num_classes=self.config.model.num_classes)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            preds, probs = get_predictions_and_probabilities(outputs)
            
            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            metrics_calc.update(preds, targets, probs)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
            
            # Log to TensorBoard
            if self.writer and batch_idx % self.config.logging.log_interval == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss_batch', loss.item(), global_step)
        
        # Compute epoch metrics
        metrics = metrics_calc.compute()
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        # Metrics tracking
        loss_meter = AverageMeter('loss')
        metrics_calc = MetricsCalculator(num_classes=self.config.model.num_classes)
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
        
        with torch.no_grad():
            for images, targets in pbar:
                # Move to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Calculate metrics
                preds, probs = get_predictions_and_probabilities(outputs)
                
                # Update metrics
                loss_meter.update(loss.item(), images.size(0))
                metrics_calc.update(preds, targets, probs)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        # Compute epoch metrics
        metrics = metrics_calc.compute()
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def train(self):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.log_epoch_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            is_best = self.check_improvement(val_metrics)
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.config.training.early_stopping:
                if self.epochs_without_improvement >= self.config.training.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        logger.info("=" * 60)
        logger.info("Training completed")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
    
    def log_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics for the epoch."""
        # Console logging
        logger.info(f"\nEpoch {self.current_epoch + 1}/{self.config.training.epochs}")
        logger.info("Train metrics:")
        for name, value in train_metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        logger.info("Val metrics:")
        for name, value in val_metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr:.2e}")
        
        # TensorBoard logging
        if self.writer:
            for name, value in train_metrics.items():
                self.writer.add_scalar(f'train/{name}', value, self.current_epoch)
            for name, value in val_metrics.items():
                self.writer.add_scalar(f'val/{name}', value, self.current_epoch)
            self.writer.add_scalar('learning_rate', current_lr, self.current_epoch)
    
    def check_improvement(self, val_metrics: Dict[str, float]) -> bool:
        """
        Check if validation metrics improved.
        
        Args:
            val_metrics: Validation metrics
            
        Returns:
            True if metrics improved
        """
        val_loss = val_metrics['loss']
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        # Save best model always
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_loss': self.best_val_loss,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoints
        if (self.current_epoch + 1) % self.config.training.save_frequency == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch+1}.pth'
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_loss': self.best_val_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch + 1}")
