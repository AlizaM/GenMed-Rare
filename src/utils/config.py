"""
Configuration management using dataclasses.

This module loads YAML configuration files and converts them to typed dataclasses
for better IDE support and type checking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    seed: int
    output_dir: str
    checkpoint_dir: str
    log_dir: str


@dataclass
class DataConfig:
    """Data configuration."""
    # Input data
    input_csv: str
    train_val_dir: str
    test_dir: str
    
    # Output processed data
    processed_csv: str
    
    # Labels
    label_column: str
    image_column: str
    class_1: str
    class_2: str
    
    # Split settings
    train_split: float
    val_split: float
    stratify: bool
    
    # Image settings
    img_size: int
    channels: int
    normalize_mean: List[float]
    normalize_std: List[float]


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    variant: str
    pretrained: bool
    num_classes: int
    dropout: float


@dataclass
class SchedulerParams:
    """Learning rate scheduler parameters."""
    mode: str
    factor: float
    patience: int
    min_lr: float


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int
    batch_size: int
    num_workers: int
    
    # Optimizer
    optimizer: str
    lr: float
    weight_decay: float
    
    # Learning rate scheduler
    scheduler: str
    scheduler_params: SchedulerParams
    
    # Checkpointing
    save_best_only: bool
    save_frequency: int
    monitor_metric: str
    monitor_mode: str
    
    # Early stopping
    early_stopping: bool
    early_stopping_patience: int


@dataclass
class AugmentationTrainConfig:
    """Training augmentation configuration."""
    rotation_degrees: float
    brightness: float
    contrast: float
    gaussian_noise_std: float
    horizontal_flip: bool
    vertical_flip: bool


@dataclass
class AugmentationValConfig:
    """Validation augmentation configuration."""
    normalize_only: bool


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""
    train: AugmentationTrainConfig
    val: AugmentationValConfig


@dataclass
class LoggingConfig:
    """Logging configuration."""
    use_tensorboard: bool
    log_interval: int
    metrics: List[str]


@dataclass
class Config:
    """Main configuration class."""
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    logging: LoggingConfig


class ConfigManager:
    """Manages loading and parsing of YAML configuration files."""
    
    @staticmethod
    def from_yaml(config_path: str) -> Config:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config object with all settings
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        return ConfigManager._parse_config(yaml_config)
    
    @staticmethod
    def _parse_config(yaml_config: Dict[str, Any]) -> Config:
        """Parse YAML dictionary into Config dataclass."""
        
        # Parse experiment config
        experiment = ExperimentConfig(**yaml_config['experiment'])
        
        # Parse data config
        data = DataConfig(**yaml_config['data'])
        
        # Parse model config
        model = ModelConfig(**yaml_config['model'])
        
        # Parse training config with scheduler params
        training_dict = yaml_config['training'].copy()
        scheduler_params = SchedulerParams(**training_dict.pop('scheduler_params'))
        training = TrainingConfig(scheduler_params=scheduler_params, **training_dict)
        
        # Parse augmentation config
        aug_dict = yaml_config['augmentation']
        aug_train = AugmentationTrainConfig(**aug_dict['train'])
        aug_val = AugmentationValConfig(**aug_dict['val'])
        augmentation = AugmentationConfig(train=aug_train, val=aug_val)
        
        # Parse logging config
        logging = LoggingConfig(**yaml_config['logging'])
        
        return Config(
            experiment=experiment,
            data=data,
            model=model,
            training=training,
            augmentation=augmentation,
            logging=logging
        )
    
    @staticmethod
    def save_yaml(config: Config, output_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            config: Config object to save
            output_path: Path where to save YAML file
        """
        # Convert dataclasses to dict
        config_dict = ConfigManager._config_to_dict(config)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def _config_to_dict(config: Config) -> Dict[str, Any]:
        """Convert Config dataclass to dictionary."""
        from dataclasses import asdict
        return asdict(config)


if __name__ == '__main__':
    # Example usage
    config = ConfigManager.from_yaml('configs/train_effusion_fibrosis.yaml')
    print(f"Loaded configuration for experiment: {config.experiment.name}")
    print(f"Classes: {config.data.class_1} vs {config.data.class_2}")
    print(f"Model: {config.model.name} ({config.model.variant})")
    print(f"Training epochs: {config.training.epochs}")
