"""Unit tests for model evaluation module."""
import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.eval.evaluator import ModelEvaluator, evaluate_checkpoint
from src.config import Config


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
        with patch.object(ModelEvaluator, 'load_checkpoint'):
            evaluator = ModelEvaluator(
                model=self.mock_model,
                test_loader=self.mock_test_loader,
                config=self.mock_config,
                checkpoint_path=self.checkpoint_path,
                output_dir=self.test_dir
            )
            
            self.assertEqual(evaluator.model, self.mock_model)
            self.assertEqual(evaluator.test_loader, self.mock_test_loader)
            self.assertEqual(evaluator.config, self.mock_config)
            self.assertEqual(evaluator.checkpoint_path, self.checkpoint_path)
            self.assertEqual(evaluator.output_dir, self.test_dir)
    
    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        with patch.object(ModelEvaluator, '__init__', lambda x, **kwargs: None):
            evaluator = ModelEvaluator.__new__(ModelEvaluator)
            evaluator.checkpoint_path = self.checkpoint_path
            evaluator.device = torch.device('cpu')
            evaluator.model = Mock()
            
            evaluator.load_checkpoint()
            
            # Verify model.load_state_dict was called
            evaluator.model.load_state_dict.assert_called_once()
            evaluator.model.to.assert_called_once()
            evaluator.model.eval.assert_called_once()
    
    @patch('src.eval.evaluator.plt')
    @patch('src.eval.evaluator.sns')
    def test_run_inference(self, mock_sns, mock_plt):
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
    @patch('src.eval.evaluator.sns')
    def test_plot_confusion_matrix(self, mock_sns, mock_plt):
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
            mock_sns.heatmap.assert_called()
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
    
    @patch('src.eval.evaluator.ChestXRayClassifier')
    @patch('src.eval.evaluator.ChestXRayDataset')
    @patch('src.eval.evaluator.torch.utils.data.DataLoader')
    def test_load_model_and_data(self, mock_dataloader, mock_dataset, mock_classifier):
        """Test model and data loading helper."""
        from src.eval.evaluator import load_model_and_data
        
        # Mock config
        mock_config = Mock()
        mock_config.data.processed_dir = str(self.test_dir)
        mock_config.data.class_positive = "pneumonia"
        mock_config.data.class_negative = "normal"
        mock_config.data.img_dir = "/fake/img/dir"
        mock_config.training.batch_size = 32
        mock_config.hardware.num_workers = 4
        mock_config.hardware.pin_memory = True
        
        # Create fake test dataset path
        test_csv_path = self.test_dir / "pneumonia_normal" / "dataset_test.csv"
        test_csv_path.parent.mkdir(parents=True, exist_ok=True)
        test_csv_path.touch()
        
        model, test_loader = load_model_and_data(
            config=mock_config,
            checkpoint_path="/fake/checkpoint.pth"
        )
        
        # Verify model and dataset were created
        mock_classifier.assert_called_once_with(mock_config)
        mock_dataset.assert_called_once()
        mock_dataloader.assert_called_once()


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)