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
    # Classes
    class_positive: str  # Rare class (label=1)
    class_negative: str  # Common class (label=0)

    # Output paths
    processed_dir: Path

    # Split configuration
    train_val_split: float
    stratified: bool

    # Image configuration
    image_size: List[int]
    channels: int

    # Input paths (only used by preprocess.py, optional for training)
    interim_csv: Path = None
    train_val_dir: Path = None
    test_dir: Path = None


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
    pin_memory: bool = False
    deterministic: bool = False
    
    def __post_init__(self):
        """Auto-detect CUDA if not available."""
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        
        # pin_memory only makes sense when using CUDA
        if self.device == "cpu":
            self.pin_memory = False


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
        # Auto-generate experiment name if not manually set
        if self.experiment.name.endswith('_baseline') and 'vs' not in self.experiment.name:
            self.experiment.name = f"{self.data.class_negative.lower()}_vs_{self.data.class_positive.lower()}_baseline"
        
        # Auto-generate processed_dir based on class names
        class_combo = f"{self.data.class_negative.lower()}_{self.data.class_positive.lower()}"
        if not str(self.data.processed_dir).endswith(class_combo):
            self.data.processed_dir = f"data/processed/{class_combo}"
        
        # Auto-generate output directories based on experiment name
        base_output = f"outputs/{self.experiment.name}"
        if not str(self.training.checkpoint_dir).startswith(base_output):
            self.training.checkpoint_dir = f"{base_output}/checkpoints"
        if not str(self.training.log_dir).startswith(base_output):
            self.training.log_dir = f"{base_output}/logs"
        
        # Convert string paths to Path objects (skip None values)
        if self.data.interim_csv:
            self.data.interim_csv = Path(self.data.interim_csv)
        if self.data.train_val_dir:
            self.data.train_val_dir = Path(self.data.train_val_dir)
        if self.data.test_dir:
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
        """Create necessary output directories and copy configuration files."""
        import shutil
        
        self.data.processed_dir.mkdir(parents=True, exist_ok=True)
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific subdirectories
        exp_checkpoint_dir = self.training.checkpoint_dir / self.experiment.name
        exp_log_dir = self.training.log_dir / self.experiment.name
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        exp_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy configuration file to experiment directory
        experiment_base_dir = Path("outputs") / self.experiment.name
        experiment_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to find and copy the config file
        for config_path in ["configs/config.yaml", "config.yaml"]:
            if Path(config_path).exists():
                config_dest = experiment_base_dir / "config.yaml"
                if not config_dest.exists():
                    shutil.copy2(config_path, config_dest)
                    print(f"Copied configuration to: {config_dest}")
                break
        
        # Copy dataset file to experiment directory if it exists
        dataset_src = self.data.processed_dir / "dataset.csv"
        if dataset_src.exists():
            dataset_dest = experiment_base_dir / "dataset.csv"
            if not dataset_dest.exists():
                shutil.copy2(dataset_src, dataset_dest)
                print(f"Copied dataset to: {dataset_dest}")
        
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
