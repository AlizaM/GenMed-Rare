"""Configuration manager with dataclasses for type safety."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml
import torch


@dataclass
class ExperimentConfig:
    """Experiment metadata configuration."""
    name: str
    description: str
    seed: int = 42


@dataclass
class DataConfig:
    """Data paths and processing configuration."""
    # Input paths
    interim_csv: Path
    train_val_dir: Path
    test_dir: Path
    
    # Output paths
    processed_dir: Path
    train_csv: str
    val_csv: str
    test_csv: str
    
    # Classes
    class_positive: str  # Rare class (label=1)
    class_negative: str  # Common class (label=0)
    
    # Split configuration
    train_val_split: float
    stratified: bool
    
    # Image configuration
    image_size: List[int]
    channels: int


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str
    variant: str
    pretrained: bool
    num_classes: int
    freeze_backbone: bool
    dropout: float


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    batch_size: int
    num_epochs: int
    num_workers: int
    
    # Optimizer
    optimizer: str
    learning_rate: float
    weight_decay: float
    
    # Learning rate scheduler
    scheduler: str
    lr_patience: int
    lr_factor: float
    lr_min: float
    
    # Early stopping
    early_stopping: bool
    early_stopping_patience: int
    early_stopping_metric: str
    
    # Checkpointing
    save_checkpoints: bool
    checkpoint_dir: Path
    save_best_only: bool
    
    # Logging
    log_dir: Path
    log_interval: int
    
    # Mixed precision
    use_amp: bool


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    # Training augmentations
    rotation_degrees: int
    brightness: float
    contrast: float
    gaussian_noise_std: float
    
    # Normalization
    normalize_mean: List[float]
    normalize_std: List[float]
    
    # Validation/test
    normalize_only: bool


@dataclass
class MetricsConfig:
    """Metrics tracking configuration."""
    track: List[str]
    primary_metric: str
    primary_mode: str  # 'max' or 'min'


@dataclass
class HardwareConfig:
    """Hardware and system configuration."""
    device: str
    pin_memory: bool
    deterministic: bool


@dataclass
class Config:
    """Main configuration container."""
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    metrics: MetricsConfig
    hardware: HardwareConfig
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert string paths to Path objects
        self.data.interim_csv = Path(self.data.interim_csv)
        self.data.train_val_dir = Path(self.data.train_val_dir)
        self.data.test_dir = Path(self.data.test_dir)
        self.data.processed_dir = Path(self.data.processed_dir)
        self.training.checkpoint_dir = Path(self.training.checkpoint_dir)
        self.training.log_dir = Path(self.training.log_dir)
        
        # Auto-detect CUDA if not explicitly set to cpu
        if self.hardware.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            self.hardware.device = "cpu"
        
        # Auto-adjust pin_memory based on device availability
        # pin_memory only makes sense when using CUDA
        if self.hardware.device == "cpu":
            self.hardware.pin_memory = False
    
    def create_dirs(self):
        """Create necessary output directories."""
        self.data.processed_dir.mkdir(parents=True, exist_ok=True)
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific subdirectories
        exp_checkpoint_dir = self.training.checkpoint_dir / self.experiment.name
        exp_log_dir = self.training.log_dir / self.experiment.name
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        exp_log_dir.mkdir(parents=True, exist_ok=True)
        
        return exp_checkpoint_dir, exp_log_dir


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file and convert to dataclasses.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with all settings
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Parse experiment config
    experiment = ExperimentConfig(**raw_config['experiment'])
    
    # Parse data config
    data = DataConfig(**raw_config['data'])
    
    # Parse model config
    model = ModelConfig(**raw_config['model'])
    
    # Parse training config
    training = TrainingConfig(**raw_config['training'])
    
    # Parse augmentation config
    aug_raw = raw_config['augmentation']
    augmentation = AugmentationConfig(
        rotation_degrees=aug_raw['train']['rotation_degrees'],
        brightness=aug_raw['train']['brightness'],
        contrast=aug_raw['train']['contrast'],
        gaussian_noise_std=aug_raw['train']['gaussian_noise_std'],
        normalize_mean=aug_raw['normalize']['mean'],
        normalize_std=aug_raw['normalize']['std'],
        normalize_only=aug_raw['val_test']['normalize_only']
    )
    
    # Parse metrics config
    metrics = MetricsConfig(**raw_config['metrics'])
    
    # Parse hardware config
    hardware = HardwareConfig(**raw_config['hardware'])
    
    # Create main config object
    config = Config(
        experiment=experiment,
        data=data,
        model=model,
        training=training,
        augmentation=augmentation,
        metrics=metrics,
        hardware=hardware
    )
    
    return config
