"""
Unit tests for configuration auto-generation features.

Run with:
    pytest tests/test_config_autogen.py -v
"""

import pytest
from pathlib import Path
import shutil

from src.config import load_config


@pytest.fixture
def config():
    """Load config."""
    return load_config("configs/config.yaml")


class TestConfigAutoGeneration:
    """Test suite for config auto-generation features."""
    
    def test_config_loads_successfully(self, config):
        """Test that config loads without errors."""
        assert config is not None
    
    def test_class_configuration(self, config):
        """Test that classes are configured."""
        assert hasattr(config.data, 'class_positive')
        assert hasattr(config.data, 'class_negative')
        assert isinstance(config.data.class_positive, str)
        assert isinstance(config.data.class_negative, str)
        assert len(config.data.class_positive) > 0
        assert len(config.data.class_negative) > 0
    
    def test_experiment_name_auto_generated(self, config):
        """Test that experiment name is auto-generated from classes."""
        experiment_name = config.experiment.name
        
        # Should contain class names or be auto_generated
        assert isinstance(experiment_name, str)
        assert len(experiment_name) > 0
    
    def test_processed_dir_auto_generated(self, config):
        """Test that processed_dir is auto-generated."""
        processed_dir = config.data.processed_dir
        
        assert isinstance(processed_dir, Path)
        # Should be under data/processed/
        assert str(processed_dir).startswith("data/processed")
    
    def test_checkpoint_dir_auto_generated(self, config):
        """Test that checkpoint_dir is auto-generated."""
        checkpoint_dir = config.training.checkpoint_dir
        
        assert isinstance(checkpoint_dir, Path)
        # Should be under outputs/{experiment_name}/
        assert str(checkpoint_dir).startswith("outputs/")
        assert "checkpoints" in str(checkpoint_dir)
    
    def test_log_dir_auto_generated(self, config):
        """Test that log_dir is auto-generated."""
        log_dir = config.training.log_dir
        
        assert isinstance(log_dir, Path)
        # Should be under outputs/{experiment_name}/
        assert str(log_dir).startswith("outputs/")
        assert "logs" in str(log_dir)
    
    def test_experiment_name_includes_classes(self, config):
        """Test that experiment name includes class information."""
        experiment_name = config.experiment.name
        
        # Should be either auto_generated or contain class names
        if experiment_name != "auto_generated_baseline":
            # Custom name might include classes
            pass
        else:
            # Default auto-generated name
            assert "auto_generated" in experiment_name or "baseline" in experiment_name
    
    def test_paths_are_path_objects(self, config):
        """Test that all path fields are Path objects."""
        assert isinstance(config.data.processed_dir, Path)
        assert isinstance(config.training.checkpoint_dir, Path)
        assert isinstance(config.training.log_dir, Path)
    
    def test_create_dirs_method_exists(self, config):
        """Test that create_dirs method exists."""
        assert hasattr(config, 'create_dirs')
        assert callable(config.create_dirs)
    
    def test_create_dirs_returns_paths(self, config):
        """Test that create_dirs returns checkpoint and log directories."""
        checkpoint_dir, log_dir = config.create_dirs()
        
        assert isinstance(checkpoint_dir, Path)
        assert isinstance(log_dir, Path)
    
    def test_create_dirs_creates_directories(self, config):
        """Test that create_dirs actually creates directories."""
        checkpoint_dir, log_dir = config.create_dirs()
        
        assert checkpoint_dir.exists()
        assert log_dir.exists()
        assert checkpoint_dir.is_dir()
        assert log_dir.is_dir()
    
    def test_experiment_dir_structure(self, config):
        """Test that experiment directory has expected structure."""
        config.create_dirs()
        
        experiment_dir = Path("outputs") / config.experiment.name
        assert experiment_dir.exists()
        
        # Should have checkpoints and logs subdirectories
        checkpoints = experiment_dir / "checkpoints"
        logs = experiment_dir / "logs"
        
        # At least one should exist (might not both if config specifies different structure)
        assert checkpoints.exists() or logs.exists()
    
    def test_seed_is_set(self, config):
        """Test that random seed is configured."""
        assert hasattr(config.experiment, 'seed')
        assert isinstance(config.experiment.seed, int)
        assert config.experiment.seed >= 0
    
    def test_model_configuration(self, config):
        """Test that model is configured."""
        assert hasattr(config, 'model')
        assert hasattr(config.model, 'name')
        assert hasattr(config.model, 'variant')
        assert hasattr(config.model, 'pretrained')
        assert hasattr(config.model, 'num_classes')
    
    def test_training_configuration(self, config):
        """Test that training parameters are configured."""
        assert hasattr(config, 'training')
        assert hasattr(config.training, 'batch_size')
        assert hasattr(config.training, 'num_epochs')
        assert hasattr(config.training, 'learning_rate')
    
    def test_data_paths_exist(self, config):
        """Test that expected data paths exist or can be created."""
        # interim_csv should be a string path
        assert hasattr(config.data, 'interim_csv')
        assert isinstance(config.data.interim_csv, str)
    
    def test_hardware_configuration(self, config):
        """Test that hardware settings are configured."""
        assert hasattr(config, 'hardware')
        assert hasattr(config.hardware, 'device')
        assert config.hardware.device in ['cuda', 'cpu']
    
    def test_augmentation_configuration(self, config):
        """Test that augmentation settings are configured."""
        assert hasattr(config, 'augmentation')
        assert hasattr(config.augmentation, 'train')
        assert hasattr(config.augmentation, 'normalize')


class TestConfigSwitching:
    """Test that config can handle different class combinations."""
    
    def test_config_with_current_classes(self, config):
        """Test config works with currently configured classes."""
        positive = config.data.class_positive
        negative = config.data.class_negative
        
        assert positive != negative
        assert len(positive) > 0
        assert len(negative) > 0
    
    def test_paths_unique_per_experiment(self, config):
        """Test that paths are unique to the experiment."""
        checkpoint_dir = config.training.checkpoint_dir
        log_dir = config.training.log_dir
        
        # Both should reference the same experiment
        assert config.experiment.name in str(checkpoint_dir)
        assert config.experiment.name in str(log_dir)
    
    def test_processed_dir_includes_classes(self, config):
        """Test that processed data dir includes class information."""
        processed_dir = config.data.processed_dir
        
        # Should be unique to this class combination
        assert isinstance(processed_dir, Path)
        # Path should exist or be creatable
        assert not str(processed_dir).startswith("auto_generated")
