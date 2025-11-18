"""Unit tests for trainer metric logging functionality."""
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from io import StringIO

import sys
sys.path.append('/home/aliza/cv_project/GenMed-Rare')

from src.train.trainer import Trainer, MetricsTracker
from src.config import Config


class MockModel(nn.Module):
    """Mock model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class MockDataLoader:
    """Mock dataloader for testing."""
    def __init__(self, num_batches=2, batch_size=4):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.data = []
        for _ in range(num_batches):
            # Create mock batch: (images, labels)
            images = torch.randn(batch_size, 10)
            labels = torch.randint(0, 2, (batch_size,))
            self.data.append((images, labels))
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return self.num_batches


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.hardware = Mock()
        self.hardware.device = 'cpu'
        
        self.training = Mock()
        self.training.optimizer = 'adam'
        self.training.learning_rate = 0.001
        self.training.weight_decay = 0.01
        self.training.scheduler = 'none'
        self.training.early_stopping = False
        self.training.early_stopping_metric = 'loss'
        self.training.early_stopping_patience = 10
        self.training.use_amp = False
        self.training.log_interval = 1
        self.training.num_epochs = 3
        self.training.batch_size = 4
        self.training.save_checkpoints = True
        
        self.experiment = Mock()
        self.experiment.name = 'test_experiment'
        
        self.metrics = Mock()
        self.metrics.primary_metric = 'f1'
        self.metrics.primary_mode = 'max'
        
    def create_dirs(self):
        """Mock directory creation."""
        checkpoint_dir = Path(tempfile.mkdtemp()) / 'checkpoints'
        log_dir = Path(tempfile.mkdtemp()) / 'logs'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir, log_dir


class TestTrainerLogging(unittest.TestCase):
    """Test trainer logging functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.train_loader = MockDataLoader(num_batches=2)
        self.val_loader = MockDataLoader(num_batches=2)
        self.config = MockConfig()
        
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directories
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.train.trainer.SummaryWriter')
    def test_trainer_initialization_with_history(self, mock_writer):
        """Test that trainer initializes with empty history tracking."""
        trainer = Trainer(self.model, self.train_loader, self.val_loader, self.config)
        
        # Check that history is properly initialized
        expected_keys = [
            'train_loss', 'train_accuracy', 'train_f1', 'train_recall',
            'val_loss', 'val_accuracy', 'val_f1', 'val_recall'
        ]
        
        for key in expected_keys:
            self.assertIn(key, trainer.history)
            self.assertEqual(trainer.history[key], [])
    
    @patch('src.train.trainer.SummaryWriter')
    @patch('src.train.trainer.logger')
    def test_log_metrics_console_output(self, mock_logger, mock_writer):
        """Test that log_metrics logs all required metrics to console."""
        trainer = Trainer(self.model, self.train_loader, self.val_loader, self.config)
        
        # Mock metrics
        train_metrics = {
            'loss': 0.5234,
            'accuracy': 0.8567,
            'f1': 0.7890,
            'recall': 0.8123,
            'auc_roc': 0.9012
        }
        val_metrics = {
            'loss': 0.4567,
            'accuracy': 0.8901,
            'f1': 0.8234,
            'recall': 0.8456,
            'auc_roc': 0.9345
        }
        
        epoch = 5
        trainer.log_metrics(epoch, train_metrics, val_metrics)
        
        # Verify console logging calls
        expected_calls = [
            call(f"\nEpoch {epoch} Results:"),
            call(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                 f"Acc: {train_metrics['accuracy']:.4f}, "
                 f"F1: {train_metrics['f1']:.4f}, "
                 f"Recall: {train_metrics['recall']:.4f}, "
                 f"AUC: {train_metrics['auc_roc']:.4f}"),
            call(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                 f"Acc: {val_metrics['accuracy']:.4f}, "
                 f"F1: {val_metrics['f1']:.4f}, "
                 f"Recall: {val_metrics['recall']:.4f}, "
                 f"AUC: {val_metrics['auc_roc']:.4f}")
        ]
        
        mock_logger.info.assert_has_calls(expected_calls)
    
    @patch('src.train.trainer.SummaryWriter')
    def test_log_metrics_history_storage(self, mock_writer):
        """Test that log_metrics stores metrics in history for plotting."""
        trainer = Trainer(self.model, self.train_loader, self.val_loader, self.config)
        
        # Mock metrics for multiple epochs
        epochs_data = [
            {
                'train': {'loss': 0.8, 'accuracy': 0.7, 'f1': 0.6, 'recall': 0.65, 'auc_roc': 0.75},
                'val': {'loss': 0.7, 'accuracy': 0.75, 'f1': 0.65, 'recall': 0.7, 'auc_roc': 0.78}
            },
            {
                'train': {'loss': 0.6, 'accuracy': 0.8, 'f1': 0.75, 'recall': 0.78, 'auc_roc': 0.85},
                'val': {'loss': 0.5, 'accuracy': 0.85, 'f1': 0.8, 'recall': 0.82, 'auc_roc': 0.9}
            }
        ]
        
        # Log metrics for multiple epochs
        for i, epoch_data in enumerate(epochs_data, 1):
            trainer.log_metrics(i, epoch_data['train'], epoch_data['val'])
        
        # Verify history storage
        self.assertEqual(len(trainer.history['train_loss']), 2)
        self.assertEqual(len(trainer.history['val_accuracy']), 2)
        
        # Check specific values
        self.assertEqual(trainer.history['train_loss'][0], 0.8)
        self.assertEqual(trainer.history['train_loss'][1], 0.6)
        self.assertEqual(trainer.history['val_f1'][0], 0.65)
        self.assertEqual(trainer.history['val_f1'][1], 0.8)
    
    @patch('src.train.trainer.SummaryWriter')
    def test_log_metrics_tensorboard_logging(self, mock_writer):
        """Test that log_metrics logs to TensorBoard correctly."""
        trainer = Trainer(self.model, self.train_loader, self.val_loader, self.config)
        
        train_metrics = {
            'loss': 0.5, 'accuracy': 0.8, 'f1': 0.7, 'recall': 0.75, 'auc_roc': 0.85,
            'confusion_matrix': [[10, 2], [3, 15]]  # Should be excluded
        }
        val_metrics = {
            'loss': 0.4, 'accuracy': 0.85, 'f1': 0.8, 'recall': 0.82, 'auc_roc': 0.9,
            'confusion_matrix': [[12, 1], [2, 17]]  # Should be excluded
        }
        
        epoch = 3
        trainer.log_metrics(epoch, train_metrics, val_metrics)
        
        # Verify TensorBoard calls (excluding confusion_matrix)
        expected_train_calls = [
            call('train/loss', 0.5, 3),
            call('train/accuracy', 0.8, 3),
            call('train/f1', 0.7, 3),
            call('train/recall', 0.75, 3),
            call('train/auc_roc', 0.85, 3)
        ]
        
        expected_val_calls = [
            call('val/loss', 0.4, 3),
            call('val/accuracy', 0.85, 3),
            call('val/f1', 0.8, 3),
            call('val/recall', 0.82, 3),
            call('val/auc_roc', 0.9, 3)
        ]
        
        # Check that confusion_matrix is not logged
        all_calls = trainer.writer.add_scalar.call_args_list
        for call_args in all_calls:
            metric_name = call_args[0][0]
            self.assertNotIn('confusion_matrix', metric_name)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('src.train.trainer.SummaryWriter')
    @patch('src.train.trainer.logger')
    def test_save_training_graphs(self, mock_logger, mock_writer, mock_tight_layout, mock_close, mock_subplots, mock_savefig):
        """Test that save_training_graphs creates and saves plots correctly."""
        trainer = Trainer(self.model, self.train_loader, self.val_loader, self.config)
        
        # Populate history with some data
        trainer.history = {
            'train_loss': [0.8, 0.6, 0.4],
            'train_accuracy': [0.7, 0.8, 0.85],
            'train_f1': [0.6, 0.75, 0.8],
            'train_recall': [0.65, 0.78, 0.82],
            'val_loss': [0.7, 0.5, 0.35],
            'val_accuracy': [0.75, 0.85, 0.9],
            'val_f1': [0.65, 0.8, 0.85],
            'val_recall': [0.7, 0.82, 0.87]
        }
        
        # Mock matplotlib objects properly
        mock_fig = Mock()
        
        # Create a mock that behaves like a 2D numpy array for axes[row, col] indexing
        class MockAxes2D:
            def __init__(self):
                self.axes = {}
                for i in range(2):
                    for j in range(2):
                        self.axes[(i, j)] = Mock()
            
            def __getitem__(self, key):
                if isinstance(key, tuple):
                    return self.axes[key]
                else:
                    # For 1D indexing, convert to 2D
                    row, col = divmod(key, 2)
                    return self.axes[(row, col)]
        
        # Create mock for 1x3 axes (1D indexing)
        mock_axes_1x3 = [Mock(), Mock(), Mock()]
        
        # Set up the subplot mock to return different configurations
        mock_subplots.side_effect = [(mock_fig, MockAxes2D()), (mock_fig, mock_axes_1x3)]
        
        # Call the method
        trainer.save_training_graphs()
        
        # Verify that plots were created and saved
        self.assertEqual(mock_subplots.call_count, 2)  # Main plot + summary plot
        self.assertEqual(mock_savefig.call_count, 2)  # Two saves
        self.assertEqual(mock_close.call_count, 2)  # Two closes
        
        # Check that logger was called with save messages
        save_calls = [call for call in mock_logger.info.call_args_list 
                     if 'saved to' in str(call)]
        self.assertEqual(len(save_calls), 2)  # Two save messages
    
    def test_metrics_tracker_functionality(self):
        """Test MetricsTracker computes all required metrics."""
        config = self.config
        tracker = MetricsTracker(config)
        
        # Create mock data
        preds = torch.tensor([0, 1, 1, 0])
        labels = torch.tensor([0, 1, 0, 0])  # One false positive
        probs = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1]])
        loss = 0.5
        
        tracker.update(preds, labels, probs, loss)
        metrics = tracker.compute()
        
        # Verify all required metrics are present
        required_metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (float, int))
        
        # Verify specific calculations
        self.assertEqual(metrics['loss'], 0.5)
        self.assertEqual(metrics['accuracy'], 0.75)  # 3/4 correct
    
    @patch('src.train.trainer.SummaryWriter')
    def test_complete_logging_workflow(self, mock_writer):
        """Test complete logging workflow during training simulation."""
        trainer = Trainer(self.model, self.train_loader, self.val_loader, self.config)
        
        # Simulate logging for multiple epochs
        test_data = [
            {
                'epoch': 1,
                'train': {'loss': 0.8, 'accuracy': 0.7, 'f1': 0.6, 'recall': 0.65, 'auc_roc': 0.75},
                'val': {'loss': 0.75, 'accuracy': 0.72, 'f1': 0.62, 'recall': 0.68, 'auc_roc': 0.78}
            },
            {
                'epoch': 2,
                'train': {'loss': 0.6, 'accuracy': 0.8, 'f1': 0.75, 'recall': 0.78, 'auc_roc': 0.85},
                'val': {'loss': 0.55, 'accuracy': 0.82, 'f1': 0.77, 'recall': 0.8, 'auc_roc': 0.87}
            }
        ]
        
        # Log metrics for each epoch
        for data in test_data:
            trainer.log_metrics(data['epoch'], data['train'], data['val'])
        
        # Verify history has correct number of entries
        for key in trainer.history:
            self.assertEqual(len(trainer.history[key]), 2)
        
        # Verify progression (metrics should improve)
        self.assertLess(trainer.history['train_loss'][1], trainer.history['train_loss'][0])
        self.assertGreater(trainer.history['val_accuracy'][1], trainer.history['val_accuracy'][0])
        self.assertGreater(trainer.history['train_f1'][1], trainer.history['train_f1'][0])


if __name__ == '__main__':
    # Set up logging to capture log messages during tests
    logging.basicConfig(level=logging.INFO)
    
    # Run the tests
    unittest.main(verbosity=2)