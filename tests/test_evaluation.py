"""Unit tests for model evaluation module."""
import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import json
import pytest

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.eval.evaluator import ModelEvaluator, evaluate_checkpoint
from src.config import Config, load_config
from src.data.dataset import ChestXrayDataset


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Mock config
        self.mock_config = Mock()
        self.mock_config.experiment.name = "test_experiment"
        self.mock_config.hardware.device = "cpu"
        self.mock_config.training.use_amp = False
        self.mock_config.data.class_negative = "Normal"
        self.mock_config.data.class_positive = "Pneumonia"
        self.mock_config.data.processed_dir = self.test_dir  # Fix: use Path instead of Mock
        self.mock_config.training.batch_size = 32
        self.mock_config.training.num_workers = 0
        self.mock_config.hardware.pin_memory = False
        
        # Mock model
        self.mock_model = Mock()
        
        # Mock test loader
        self.mock_test_loader = Mock()
        
        # Create fake checkpoint
        self.checkpoint_path = self.test_dir / "test_checkpoint.pth"
        torch.save({
            'model_state_dict': {'layer.weight': torch.tensor([1.0, 2.0])},
            'epoch': 10,
            'metrics': {'accuracy': 0.85, 'f1': 0.82}
        }, self.checkpoint_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_evaluator_initialization(self):
        """Test ModelEvaluator initialization."""
        with patch.object(ModelEvaluator, '_load_model'), \
             patch.object(ModelEvaluator, '_create_test_loader'):
            evaluator = ModelEvaluator(
                config=self.mock_config,
                checkpoint_path=self.checkpoint_path
            )
            
            self.assertEqual(evaluator.config, self.mock_config)
            self.assertEqual(evaluator.checkpoint_path, self.checkpoint_path)
    
    def test_load_model(self):
        """Test model loading from checkpoint."""
        with patch('src.eval.evaluator.SwinClassifier') as mock_classifier, \
             patch('src.eval.evaluator.torch.load') as mock_torch_load:
            
            # Mock checkpoint data
            mock_torch_load.return_value = {
                'model_state_dict': {'layer.weight': torch.tensor([1.0])},
                'epoch': 10,
                'metrics': {'accuracy': 0.85}
            }
            
            # Mock model
            mock_model = Mock()
            mock_classifier.return_value = mock_model
            
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.checkpoint_path = self.checkpoint_path
            evaluator.device = torch.device('cpu')
            evaluator.config = self.mock_config
            
            model = evaluator._load_model()
            
            # Verify model loading was called
            mock_classifier.assert_called_once_with(self.mock_config)
            mock_model.load_state_dict.assert_called_once()
            mock_model.to.assert_called_once()
    
    def test_run_inference(self):
        """Test inference execution."""
        # Create sample data
        sample_images = torch.randn(2, 3, 224, 224)
        sample_labels = torch.tensor([0, 1])
        sample_outputs = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        
        # Mock test loader
        mock_test_loader = [(sample_images, sample_labels)]
        
        # Mock model output
        mock_model = Mock()
        mock_model.return_value = sample_outputs
        
        with patch.object(ModelEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.test_loader = mock_test_loader
            evaluator.model = mock_model
            evaluator.device = torch.device('cpu')
            evaluator.config = self.mock_config
            evaluator.results = {
                'predictions': [],
                'labels': [],
                'probabilities': []
            }
            
            result = evaluator.run_inference()
            
            # Check results
            self.assertEqual(len(result['predictions']), 2)
            self.assertEqual(len(result['labels']), 2)
            self.assertEqual(len(result['probabilities']), 2)
            
            # Verify predictions
            expected_preds = torch.argmax(torch.softmax(sample_outputs, dim=1), dim=1).numpy()
            np.testing.assert_array_equal(result['predictions'], expected_preds)
            np.testing.assert_array_equal(result['labels'], sample_labels.numpy())
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        with patch.object(ModelEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.config = self.mock_config
            
            # Mock results
            evaluator.results = {
                'predictions': np.array([0, 1, 1, 0]),
                'labels': np.array([0, 1, 0, 0]),
                'probabilities': np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1]])
            }
            
            metrics = evaluator.compute_metrics()
            
            # Check that all required metrics are computed
            required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
            for metric in required_metrics:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], float)
            
            # Check confusion matrix is stored
            self.assertIn('confusion_matrix', evaluator.results)
            self.assertIn('classification_report', evaluator.results)
    
    @patch('src.eval.evaluator.plt')
    @patch('src.eval.evaluator.json.dump')
    @patch('src.eval.evaluator.pd.DataFrame')
    def test_save_results(self, mock_df, mock_json_dump, mock_plt):
        """Test results saving."""
        with patch.object(ModelEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.output_dir = self.test_dir
            evaluator.config = self.mock_config
            evaluator.checkpoint_path = self.checkpoint_path
            evaluator.test_data_path = self.test_dir / "dataset.csv"  # Add missing attribute
            evaluator.device = torch.device('cpu')
            
            # Mock results
            evaluator.results = {
                'predictions': np.array([0, 1]),
                'labels': np.array([0, 1]),
                'probabilities': np.array([[0.8, 0.2], [0.3, 0.7]]),
                'metrics': {'accuracy': 0.85},
                'confusion_matrix': [[1, 0], [0, 1]],
                'classification_report': {'accuracy': 0.85}
            }
            
            # Mock DataFrame
            mock_df_instance = Mock()
            mock_df.return_value = mock_df_instance
            
            evaluator.save_results()
            
            # Verify JSON dump was called
            mock_json_dump.assert_called_once()
            
            # Verify DataFrame was created and saved
            mock_df.assert_called_once()
            mock_df_instance.to_csv.assert_called_once()
    
    @patch('src.eval.evaluator.plt')
    def test_plot_confusion_matrix(self, mock_plt):
        """Test confusion matrix plotting."""
        with patch.object(ModelEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.output_dir = self.test_dir
            evaluator.config = self.mock_config
            evaluator.results = {
                'confusion_matrix': [[10, 2], [3, 15]]
            }
            
            evaluator.plot_confusion_matrix()
            
            # Verify plotting functions were called
            mock_plt.figure.assert_called()
            mock_plt.savefig.assert_called()
            mock_plt.close.assert_called()
    
    @patch('src.eval.evaluator.plt')
    @patch('src.eval.evaluator.roc_curve')
    def test_plot_roc_curve(self, mock_roc_curve, mock_plt):
        """Test ROC curve plotting."""
        # Mock ROC curve data
        mock_roc_curve.return_value = ([0, 0.5, 1], [0, 0.8, 1], [0.5, 0.3, 0.1])
        
        with patch.object(ModelEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.output_dir = self.test_dir
            evaluator.config = self.mock_config
            evaluator.results = {
                'labels': np.array([0, 1, 0, 1]),
                'probabilities': np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]]),
                'metrics': {'auc_roc': 0.85}
            }
            
            evaluator.plot_roc_curve()
            
            # Verify plotting functions were called
            mock_plt.figure.assert_called()
            mock_plt.plot.assert_called()
            mock_plt.savefig.assert_called()
            mock_plt.close.assert_called()
    
    def test_plot_roc_curve_invalid_auc(self):
        """Test ROC curve plotting with invalid AUC."""
        with patch.object(ModelEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.results = {
                'metrics': {'auc_roc': 0.0}
            }
            
            # Should not raise an exception
            evaluator.plot_roc_curve()


class TestEvaluationHelpers(unittest.TestCase):
    """Test cases for evaluation helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_evaluate_checkpoint_function(self):
        """Test the evaluate_checkpoint helper function."""
        # This function exists in the evaluator module
        from src.eval.evaluator import evaluate_checkpoint
        
        # Test that the function is importable
        self.assertTrue(callable(evaluate_checkpoint))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)


# ============================================================================
# PYTEST-BASED INTEGRATION TESTS
# Covers functionality from test_evaluation_setup.py and test_evaluator_init.py
# ============================================================================

@pytest.fixture(scope="module")
def config():
    """Load configuration for integration tests."""
    return load_config('configs/config.yaml')


@pytest.fixture(scope="module")
def dataset_path(config):
    """Get path to test dataset CSV."""
    dataset_path = config.data.processed_dir / 'dataset.csv'
    if not dataset_path.exists():
        pytest.skip(f"Dataset CSV not found: {dataset_path}")
    return dataset_path


@pytest.fixture(scope="module")
def test_dataset(dataset_path, config):
    """Create test dataset."""
    return ChestXrayDataset(
        csv_path=dataset_path,
        config=config,
        split='test'
    )


@pytest.fixture(scope="module")
def checkpoint_path():
    """Get path to trained model checkpoint."""
    checkpoint_patterns = [
        "outputs/*/checkpoints/*/checkpoint_*.pth",
        "outputs/*/checkpoints/best_model.pth", 
        "outputs/*/checkpoints/latest_model.pth"
    ]
    
    for pattern in checkpoint_patterns:
        checkpoints = list(Path(".").glob(pattern))
        if checkpoints:
            return checkpoints[0]
    
    return None


class TestEvaluationDataSetup:
    """Integration tests for evaluation data loading (from test_evaluation_setup.py)."""
    
    def test_dataset_csv_exists(self, dataset_path):
        """Test that dataset CSV file exists."""
        assert dataset_path.exists(), "Dataset CSV should exist"
        assert dataset_path.suffix == '.csv', "Dataset file should be CSV"
    
    def test_test_dataset_creation(self, test_dataset, config):
        """Test creating test dataset."""
        assert len(test_dataset) > 0, "Test dataset should not be empty"
        assert test_dataset.split == 'test', "Dataset should be test split"
        assert not test_dataset.is_training, "Test dataset should not be in training mode"
        
        print(f"✓ Test dataset created with {len(test_dataset)} samples")
    
    def test_class_distribution(self, test_dataset, config):
        """Test that test dataset has both classes."""
        labels = [test_dataset.df.iloc[i]['label'] for i in range(len(test_dataset))]
        class_counts = {0: labels.count(0), 1: labels.count(1)}
        
        assert class_counts[0] > 0, f"Should have {config.data.class_negative} samples"
        assert class_counts[1] > 0, f"Should have {config.data.class_positive} samples"
        
        print(f"✓ Class distribution: {config.data.class_negative}={class_counts[0]}, {config.data.class_positive}={class_counts[1]}")
    
    def test_sample_loading(self, test_dataset, config):
        """Test loading individual samples."""
        sample_image, sample_label = test_dataset[0]
        
        assert isinstance(sample_image, torch.Tensor), "Image should be a tensor"
        assert isinstance(sample_label, int), "Label should be an integer"
        
        expected_shape = (config.data.channels, *config.data.image_size)
        assert sample_image.shape == expected_shape, f"Image shape should be {expected_shape}"
        assert sample_label in [0, 1], "Label should be 0 or 1"
        
        print(f"✓ Sample loading: image {sample_image.shape}, label {sample_label}")
    
    def test_dataloader_creation(self, test_dataset):
        """Test creating DataLoader for test dataset."""
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Use 0 for testing
            pin_memory=False  # Avoid warnings
        )
        
        assert len(test_loader) > 0, "DataLoader should have batches"
        
        # Test loading a batch
        batch_images, batch_labels = next(iter(test_loader))
        
        assert batch_images.dim() == 4, "Batch images should be 4D tensor"
        assert batch_labels.dim() == 1, "Batch labels should be 1D tensor"
        assert batch_images.shape[0] == batch_labels.shape[0], "Batch size should match"
        
        print(f"✓ DataLoader created: {len(test_loader)} batches, batch shape {batch_images.shape}")


class TestModelEvaluatorIntegration:
    """Integration tests for ModelEvaluator initialization (from test_evaluator_init.py)."""
    
    def test_evaluator_requires_valid_checkpoint(self, config):
        """Test that evaluator initialization requires a valid checkpoint."""
        fake_checkpoint = "nonexistent/checkpoint.pth"
        
        with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
            ModelEvaluator(
                config=config,
                checkpoint_path=fake_checkpoint,
                device='cpu'
            )
        
        print("✓ Properly rejects invalid checkpoint paths")
    
    def test_evaluator_initialization_with_checkpoint(self, config, checkpoint_path):
        """Test full evaluator initialization with real checkpoint."""
        if checkpoint_path is None:
            pytest.skip("No trained checkpoint found - train a model first to test full evaluation")
        
        print(f"\n✓ Testing with checkpoint: {checkpoint_path}")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            config=config,
            checkpoint_path=str(checkpoint_path),
            device='cpu'  # Use CPU for testing
        )
        
        # Verify components are properly initialized
        assert evaluator.model is not None, "Model should be loaded"
        assert evaluator.test_loader is not None, "Test loader should be created"
        assert evaluator.output_dir.exists(), "Output directory should exist"
        assert len(evaluator.test_loader.dataset) > 0, "Test dataset should not be empty"
        
        print(f"✓ ModelEvaluator initialized successfully")
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"✓ Test samples: {len(evaluator.test_loader.dataset)}")
        print(f"✓ Output directory: {evaluator.output_dir}")
        
        # Test that model is in eval mode
        assert not evaluator.model.training, "Model should be in eval mode"
        
        print("✓ Model is in evaluation mode")
    
    def test_config_consistency(self, config, test_dataset):
        """Test that config settings match dataset properties."""
        sample_image, _ = test_dataset[0]
        expected_channels = config.data.channels
        expected_size = tuple(config.data.image_size)
        
        assert sample_image.shape[0] == expected_channels, "Channel count should match config"
        assert sample_image.shape[1:] == expected_size, "Image size should match config"
        
        print(f"✓ Config consistency: {expected_channels}x{expected_size} images")
    
    @pytest.mark.slow
    def test_end_to_end_inference_sample(self, config, checkpoint_path):
        """Test running a small inference sample end-to-end."""
        if checkpoint_path is None:
            pytest.skip("No trained checkpoint found - train a model first to test full evaluation")
        
        print(f"\n✓ Running end-to-end inference test with {checkpoint_path}")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            config=config,
            checkpoint_path=str(checkpoint_path),
            device='cpu'
        )
        
        # Test inference on just a few samples
        test_loader_subset = torch.utils.data.DataLoader(
            evaluator.test_loader.dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        
        # Run inference on one batch
        evaluator.model.eval()
        with torch.no_grad():
            images, labels = next(iter(test_loader_subset))
            images = images.to(evaluator.device)
            outputs = evaluator.model(images)
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        
        assert outputs.shape[0] == images.shape[0], "Output batch size should match input"
        assert outputs.shape[1] == 2, "Should have 2 class outputs for binary classification"
        assert len(predictions) == len(labels), "Predictions should match label count"
        
        print(f"✓ End-to-end test passed:")
        print(f"   Input: {images.shape}")
        print(f"   Output: {outputs.shape}")
        print(f"   Predictions: {predictions.tolist()}")
        print(f"   Ground truth: {labels.tolist()}")


def test_evaluation_setup_summary(config, test_dataset, checkpoint_path):
    """Summary test showing overall evaluation setup status."""
    print("\n" + "=" * 60)
    print("EVALUATION SETUP SUMMARY")
    print("=" * 60)
    
    # Config info
    print(f"Experiment: {config.experiment.name}")
    print(f"Classes: {config.data.class_negative} vs {config.data.class_positive}")
    
    # Dataset info
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Check for checkpoints
    available_checkpoints = list(Path(".").glob("outputs/*/checkpoints/*/checkpoint_*.pth"))
    print(f"Available checkpoints: {len(available_checkpoints)}")
    
    if checkpoint_path:
        print(f"Active checkpoint: {checkpoint_path}")
        print("\n✅ FULLY READY FOR EVALUATION!")
        print("   ✓ Data loading works")
        print("   ✓ Model checkpoint available")
        print("   ✓ Full evaluation pipeline ready")
    else:
        print("\n⚠️  PARTIALLY READY")
        print("   ✓ Data loading works")
        print("   ✗ No model checkpoint found")
        print("   → Train a model first: python scripts/train_classifier.py")
    
    print("=" * 60)