"""pytest tests for configuration management."""
import sys
from pathlib import Path
import pytest
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, Config
from src.config.config_manager import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    AugmentationConfig,
    MetricsConfig,
    HardwareConfig
)


@pytest.fixture(scope="module")
def config_path():
    """Get path to config file."""
    path = Path('configs/config.yaml')
    assert path.exists(), f"Config file not found: {path}"
    return path


@pytest.fixture(scope="module")
def config(config_path):
    """Load configuration."""
    return load_config(config_path)


@pytest.fixture(scope="module")
def config_dict(config_path):
    """Load raw config YAML as dict."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TestConfigLoading:
    """Test configuration loading."""
    
    def test_load_config_returns_config_object(self, config):
        """Test that load_config returns Config object."""
        assert isinstance(config, Config), "Should return Config object"
    
    def test_load_config_with_path_object(self):
        """Test loading config with Path object."""
        config_path = Path('configs/config.yaml')
        config = load_config(config_path)
        assert isinstance(config, Config), "Should handle Path objects"
    
    def test_load_config_with_string_path(self):
        """Test loading config with string path."""
        config = load_config('configs/config.yaml')
        assert isinstance(config, Config), "Should handle string paths"
    
    def test_load_config_nonexistent_file(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config('configs/nonexistent.yaml')


class TestExperimentConfig:
    """Test ExperimentConfig."""
    
    def test_experiment_config_exists(self, config):
        """Test that experiment config exists."""
        assert hasattr(config, 'experiment'), "Config should have experiment"
        assert isinstance(config.experiment, ExperimentConfig), "Should be ExperimentConfig"
    
    def test_experiment_name(self, config, config_dict):
        """Test experiment name."""
        assert config.experiment.name == config_dict['experiment']['name']
        assert isinstance(config.experiment.name, str)
    
    def test_experiment_description(self, config, config_dict):
        """Test experiment description."""
        assert config.experiment.description == config_dict['experiment']['description']
    
    def test_experiment_seed(self, config, config_dict):
        """Test experiment seed."""
        assert config.experiment.seed == config_dict['experiment']['seed']
        assert isinstance(config.experiment.seed, int)
        assert config.experiment.seed >= 0


class TestDataConfig:
    """Test DataConfig."""
    
    def test_data_config_exists(self, config):
        """Test that data config exists."""
        assert hasattr(config, 'data'), "Config should have data"
        assert isinstance(config.data, DataConfig), "Should be DataConfig"
    
    def test_data_paths_are_path_objects(self, config):
        """Test that paths are converted to Path objects."""
        assert isinstance(config.data.interim_csv, Path)
        assert isinstance(config.data.train_val_dir, Path)
        assert isinstance(config.data.test_dir, Path)
        assert isinstance(config.data.processed_dir, Path)
    
    def test_data_csv_filenames_are_strings(self, config):
        """Test that CSV filenames are strings."""
        assert isinstance(config.data.train_csv, str)
        assert isinstance(config.data.val_csv, str)
        assert isinstance(config.data.test_csv, str)
    
    def test_data_classes(self, config, config_dict):
        """Test class configuration."""
        assert config.data.class_positive == config_dict['data']['class_positive']
        assert config.data.class_negative == config_dict['data']['class_negative']
        assert config.data.class_positive != config.data.class_negative
    
    def test_data_split_config(self, config, config_dict):
        """Test split configuration."""
        assert config.data.train_val_split == config_dict['data']['train_val_split']
        assert 0 < config.data.train_val_split < 1, "Split should be between 0 and 1"
        assert config.data.stratified == config_dict['data']['stratified']
        assert isinstance(config.data.stratified, bool)
    
    def test_data_image_config(self, config, config_dict):
        """Test image configuration."""
        assert config.data.image_size == config_dict['data']['image_size']
        assert len(config.data.image_size) == 2, "Image size should be [height, width]"
        assert all(isinstance(x, int) and x > 0 for x in config.data.image_size)
        assert config.data.channels == config_dict['data']['channels']
        assert config.data.channels in [1, 3], "Channels should be 1 or 3"


class TestModelConfig:
    """Test ModelConfig."""
    
    def test_model_config_exists(self, config):
        """Test that model config exists."""
        assert hasattr(config, 'model'), "Config should have model"
        assert isinstance(config.model, ModelConfig), "Should be ModelConfig"
    
    def test_model_name_and_variant(self, config, config_dict):
        """Test model name and variant."""
        assert config.model.name == config_dict['model']['name']
        assert config.model.variant == config_dict['model']['variant']
    
    def test_model_pretrained(self, config):
        """Test pretrained flag."""
        assert isinstance(config.model.pretrained, bool)
    
    def test_model_num_classes(self, config):
        """Test num_classes."""
        assert config.model.num_classes == 2, "Binary classification should have 2 classes"
    
    def test_model_dropout(self, config):
        """Test dropout."""
        assert 0 <= config.model.dropout < 1, "Dropout should be between 0 and 1"


class TestTrainingConfig:
    """Test TrainingConfig."""
    
    def test_training_config_exists(self, config):
        """Test that training config exists."""
        assert hasattr(config, 'training'), "Config should have training"
        assert isinstance(config.training, TrainingConfig), "Should be TrainingConfig"
    
    def test_training_batch_size(self, config):
        """Test batch size."""
        assert isinstance(config.training.batch_size, int)
        assert config.training.batch_size > 0
    
    def test_training_epochs(self, config):
        """Test num_epochs."""
        assert isinstance(config.training.num_epochs, int)
        assert config.training.num_epochs > 0
    
    def test_training_optimizer_config(self, config, config_dict):
        """Test optimizer configuration."""
        assert config.training.optimizer == config_dict['training']['optimizer']
        assert config.training.learning_rate > 0
        assert config.training.weight_decay >= 0
    
    def test_training_scheduler_config(self, config, config_dict):
        """Test scheduler configuration."""
        assert config.training.scheduler == config_dict['training']['scheduler']
        assert config.training.lr_patience > 0
        assert 0 < config.training.lr_factor < 1
        assert config.training.lr_min > 0
    
    def test_training_early_stopping(self, config):
        """Test early stopping configuration."""
        assert isinstance(config.training.early_stopping, bool)
        if config.training.early_stopping:
            assert config.training.early_stopping_patience > 0
            assert config.training.early_stopping_metric in ['val_loss', 'val_accuracy', 'val_f1']
    
    def test_training_paths_are_path_objects(self, config):
        """Test that checkpoint and log dirs are Path objects."""
        assert isinstance(config.training.checkpoint_dir, Path)
        assert isinstance(config.training.log_dir, Path)
    
    def test_training_experiment_specific_paths(self, config):
        """Test that paths include experiment name."""
        exp_name = config.experiment.name
        assert exp_name in str(config.training.checkpoint_dir)
        assert exp_name in str(config.training.log_dir)


class TestAugmentationConfig:
    """Test AugmentationConfig."""
    
    def test_augmentation_config_exists(self, config):
        """Test that augmentation config exists."""
        assert hasattr(config, 'augmentation'), "Config should have augmentation"
        assert isinstance(config.augmentation, AugmentationConfig), "Should be AugmentationConfig"
    
    def test_augmentation_rotation(self, config):
        """Test rotation configuration."""
        assert config.augmentation.rotation_degrees > 0
        # Medical imaging: typically small rotations
        assert config.augmentation.rotation_degrees <= 15
    
    def test_augmentation_brightness_contrast(self, config):
        """Test brightness and contrast."""
        assert 0 < config.augmentation.brightness < 1
        assert 0 < config.augmentation.contrast < 1
    
    def test_augmentation_gaussian_noise(self, config):
        """Test Gaussian noise."""
        assert config.augmentation.gaussian_noise_std > 0
        assert config.augmentation.gaussian_noise_std < 0.1  # Should be small
    
    def test_augmentation_normalization(self, config):
        """Test normalization values."""
        assert len(config.augmentation.normalize_mean) == 3
        assert len(config.augmentation.normalize_std) == 3
        # ImageNet normalization
        assert all(0 <= x <= 1 for x in config.augmentation.normalize_mean)
        assert all(0 < x <= 1 for x in config.augmentation.normalize_std)


class TestMetricsConfig:
    """Test MetricsConfig."""
    
    def test_metrics_config_exists(self, config):
        """Test that metrics config exists."""
        assert hasattr(config, 'metrics'), "Config should have metrics"
        assert isinstance(config.metrics, MetricsConfig), "Should be MetricsConfig"
    
    def test_metrics_track_list(self, config):
        """Test metrics tracking list."""
        assert isinstance(config.metrics.track, list)
        assert len(config.metrics.track) > 0
        
        expected_metrics = {'accuracy', 'precision', 'recall', 'f1', 'auc_roc'}
        tracked_metrics = set(config.metrics.track)
        assert tracked_metrics.issubset(expected_metrics | {'confusion_matrix'})
    
    def test_metrics_primary_metric(self, config):
        """Test primary metric configuration."""
        assert config.metrics.primary_metric in config.metrics.track
        assert config.metrics.primary_mode in ['max', 'min']


class TestHardwareConfig:
    """Test HardwareConfig."""
    
    def test_hardware_config_exists(self, config):
        """Test that hardware config exists."""
        assert hasattr(config, 'hardware'), "Config should have hardware"
        assert isinstance(config.hardware, HardwareConfig), "Should be HardwareConfig"
    
    def test_hardware_device(self, config):
        """Test device configuration."""
        assert config.hardware.device in ['cuda', 'cpu']
    
    def test_hardware_deterministic(self, config):
        """Test deterministic flag."""
        assert isinstance(config.hardware.deterministic, bool)


class TestConfigMethods:
    """Test Config methods."""
    
    def test_create_dirs(self, config, tmp_path):
        """Test create_dirs method."""
        # Create a test config with temp paths
        config.data.processed_dir = tmp_path / 'processed'
        config.training.checkpoint_dir = tmp_path / 'checkpoints'
        config.training.log_dir = tmp_path / 'logs'
        
        # Call create_dirs
        config.create_dirs()
        
        # Check directories were created
        assert config.data.processed_dir.exists()
        assert config.training.checkpoint_dir.exists()
        assert config.training.log_dir.exists()


class TestConfigValidation:
    """Test configuration validation and consistency."""
    
    def test_config_consistency(self, config):
        """Test that config values are internally consistent."""
        # Training batch size should be reasonable
        assert 1 <= config.training.batch_size <= 256
        
        # Learning rate should be reasonable
        assert 1e-6 <= config.training.learning_rate <= 1e-1
        
        # Image size should be reasonable for transformers
        assert all(32 <= x <= 512 for x in config.data.image_size)
    
    def test_no_missing_required_fields(self, config):
        """Test that all required config sections exist."""
        required_sections = [
            'experiment', 'data', 'model', 'training',
            'augmentation', 'metrics', 'hardware'
        ]
        for section in required_sections:
            assert hasattr(config, section), f"Missing required section: {section}"
