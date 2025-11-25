"""
Diffusion model configuration manager with dataclasses for type safety.

This module provides configuration management for:
- Diffusion model training (prior-based and standard)
- Diffusion model evaluation
- Image generation

Reuses common components from config_manager.py (HardwareConfig).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
import yaml

# Import shared config classes
from .config_manager import HardwareConfig


@dataclass
class EvaluationConfig:
    """Evaluation-specific configuration."""
    # Target label to evaluate
    label: str
    
    # Checkpoint configuration
    checkpoint_dir: Path
    checkpoint_names: List[str]
    
    # Output directory
    output_dir: Path
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if not isinstance(self.checkpoint_dir, Path):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)


@dataclass
class DiffusionDataConfig:
    """Data configuration for diffusion models."""
    # Data paths
    data_dir: Path
    csv_file: str
    label_subdir: str  # Subdirectory within data_dir containing images
    
    # Image preprocessing
    image_size: int
    center_crop: bool
    random_flip: bool
    
    # Prompt configuration
    prompt_template: str
    
    def __post_init__(self):
        """Convert string paths to Path objects and validate."""
        if not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)
        
        # Validate image size
        if self.image_size not in [256, 512, 1024]:
            raise ValueError(f"Image size must be 256, 512, or 1024, got {self.image_size}")


@dataclass
class DiffusionModelConfig:
    """Diffusion model configuration."""
    pretrained_model: str
    
    def __post_init__(self):
        """Validate model name."""
        if not self.pretrained_model:
            raise ValueError("pretrained_model cannot be empty")


@dataclass
class GenerationConfig:
    """Image generation configuration."""
    num_images: int
    num_inference_steps: int
    guidance_scale: float
    lora_scale: float = 1.0
    negative_prompt: str = "blurry, low quality, distorted, artifacts"
    
    def __post_init__(self):
        """Validate generation parameters."""
        if self.num_images < 1:
            raise ValueError(f"num_images must be >= 1, got {self.num_images}")
        
        if self.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {self.num_inference_steps}")
        
        if self.guidance_scale < 0:
            raise ValueError(f"guidance_scale must be >= 0, got {self.guidance_scale}")
        
        if self.lora_scale < 0:
            raise ValueError(f"lora_scale must be >= 0, got {self.lora_scale}")


@dataclass
class DiffusionHardwareConfig(HardwareConfig):
    """
    Hardware configuration for diffusion models.
    
    Extends base HardwareConfig with diffusion-specific memory optimizations.
    """
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    
    # Inherited from HardwareConfig:
    # - device: str
    # - pin_memory: bool (auto-set based on device)
    # - deterministic: bool


@dataclass
class MetricsConfig:
    """Evaluation metrics configuration."""
    novelty_metric: str = "ssim"
    visualize_top_k: int = 10
    seed: int = 42
    
    def __post_init__(self):
        """Validate metrics configuration."""
        if self.novelty_metric not in ["ssim", "correlation"]:
            raise ValueError(f"novelty_metric must be 'ssim' or 'correlation', got {self.novelty_metric}")
        
        if self.visualize_top_k < 1:
            raise ValueError(f"visualize_top_k must be >= 1, got {self.visualize_top_k}")


@dataclass
class DiffusionEvaluationConfig:
    """Main configuration container for diffusion evaluation."""
    evaluation: EvaluationConfig
    data: DiffusionDataConfig
    model: DiffusionModelConfig
    generation: GenerationConfig
    hardware: DiffusionHardwareConfig
    metrics: MetricsConfig
    
    def __post_init__(self):
        """Post-initialization validation and path resolution."""
        # Auto-generate output directory based on label if not customized
        label_lower = self.evaluation.label.lower()
        expected_output = f"checkpoint_evaluation/{label_lower}"
        
        if not str(self.evaluation.output_dir).endswith(label_lower):
            # Output dir doesn't match label, update it
            base_dir = self.evaluation.output_dir.parent
            self.evaluation.output_dir = base_dir / label_lower
    
    def create_dirs(self):
        """Create necessary output directories."""
        self.evaluation.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directory: {self.evaluation.output_dir}")
    
    def validate_paths(self):
        """
        Validate that all required paths exist.
        
        Raises:
            FileNotFoundError: If checkpoint_dir, data_dir, or CSV file doesn't exist
            ValueError: If no checkpoints found
        """
        # Validate checkpoint directory exists
        if not self.evaluation.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found:\n"
                f"  Path: {self.evaluation.checkpoint_dir}\n"
                f"  Please verify the path is correct."
            )
        
        # Validate individual checkpoints exist
        missing_checkpoints = []
        for checkpoint_name in self.evaluation.checkpoint_names:
            checkpoint_path = self.evaluation.checkpoint_dir / checkpoint_name
            if not checkpoint_path.exists():
                missing_checkpoints.append(checkpoint_name)
        
        if missing_checkpoints:
            available = [p.name for p in self.evaluation.checkpoint_dir.glob("checkpoint-*")]
            raise FileNotFoundError(
                f"Checkpoints not found:\n"
                f"  Missing: {', '.join(missing_checkpoints)}\n"
                f"  Base directory: {self.evaluation.checkpoint_dir}\n"
                f"  Available checkpoints: {', '.join(available) if available else 'None'}"
            )
        
        # Validate data directory exists
        if not self.data.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found:\n"
                f"  Path: {self.data.data_dir}\n"
                f"  Please verify the path is correct."
            )
        
        # Validate CSV file exists
        csv_path = self.data.data_dir / self.data.csv_file
        if not csv_path.exists():
            available_csvs = list(self.data.data_dir.glob("*.csv"))
            raise FileNotFoundError(
                f"CSV file not found:\n"
                f"  Path: {csv_path}\n"
                f"  Available CSV files: {', '.join([f.name for f in available_csvs]) if available_csvs else 'None'}"
            )
        
        # Validate images exist
        image_files = list(self.data.data_dir.glob("**/*.png")) + list(self.data.data_dir.glob("**/*.jpg"))
        if len(image_files) == 0:
            raise FileNotFoundError(
                f"No images found in data directory:\n"
                f"  Path: {self.data.data_dir}"
            )
        
        print(f"✓ Path validation passed")
        print(f"  Checkpoints: {len(self.evaluation.checkpoint_names)} found")
        print(f"  Images: {len(image_files)} found")


def load_diffusion_eval_config(config_path: str) -> DiffusionEvaluationConfig:
    """
    Load diffusion evaluation configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        DiffusionEvaluationConfig object with all settings
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config has missing or invalid fields
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Validate top-level sections exist
    required_sections = ['evaluation', 'data', 'model', 'generation']
    missing_sections = [s for s in required_sections if s not in raw_config]
    if missing_sections:
        raise ValueError(
            f"Config file missing required sections: {', '.join(missing_sections)}\n"
            f"  Config file: {config_path}"
        )
    
    try:
        # Parse evaluation config
        evaluation = EvaluationConfig(**raw_config['evaluation'])
        
        # Parse data config
        data = DiffusionDataConfig(**raw_config['data'])
        
        # Parse model config
        model = DiffusionModelConfig(**raw_config['model'])
        
        # Parse generation config
        generation = GenerationConfig(**raw_config['generation'])
        
        # Parse hardware config (with defaults, extends HardwareConfig)
        hw_raw = raw_config.get('hardware', {})
        # Set defaults for diffusion-specific fields
        hw_raw.setdefault('enable_attention_slicing', True)
        hw_raw.setdefault('enable_vae_slicing', True)
        hw_raw.setdefault('device', 'cuda')
        hw_raw.setdefault('pin_memory', False)
        hw_raw.setdefault('deterministic', False)
        hardware = DiffusionHardwareConfig(**hw_raw)
        
        # Parse metrics config (with defaults)
        metrics = MetricsConfig(**raw_config.get('metrics', {}))
        
        # Create main config object
        config = DiffusionEvaluationConfig(
            evaluation=evaluation,
            data=data,
            model=model,
            generation=generation,
            hardware=hardware,
            metrics=metrics
        )
        
    except TypeError as e:
        raise ValueError(
            f"Config file has missing or invalid fields:\n"
            f"  Error: {e}\n"
            f"  Config file: {config_path}"
        )
    
    return config


# ============================================================================
# Training Configuration Classes
# ============================================================================

@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    rank: int = 64
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["to_k", "to_q", "to_v", "to_out.0"])
    bias: str = "none"
    
    def __post_init__(self):
        """Validate LoRA parameters."""
        if self.rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {self.rank}")
        if self.alpha < 1:
            raise ValueError(f"LoRA alpha must be >= 1, got {self.alpha}")
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"LoRA dropout must be in [0, 1], got {self.dropout}")
        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"LoRA bias must be 'none', 'all', or 'lora_only', got {self.bias}")


@dataclass
class DiffusionTrainingModelConfig:
    """Model configuration for training."""
    pretrained_model: str
    vae_path: Optional[str] = None
    use_lora: bool = True
    lora: Optional[LoRAConfig] = None
    
    def __post_init__(self):
        """Validate and initialize LoRA config."""
        if not self.pretrained_model:
            raise ValueError("pretrained_model cannot be empty")
        
        # Create default LoRA config if using LoRA but no config provided
        if self.use_lora and self.lora is None:
            self.lora = LoRAConfig()
        
        # Convert dict to LoRAConfig if needed
        if self.use_lora and isinstance(self.lora, dict):
            self.lora = LoRAConfig(**self.lora)


@dataclass
class PriorBasedTrainingConfig:
    """Training configuration for prior-based learning."""
    # Training mode
    mode: str = "prior_based"
    target_pathology: str = ""
    # Data paths
    target_images_dir: str = ""
    target_images_csv: str = ""
    prior_images_dir: str = ""
    prior_images_csv: str = ""
    # Prompts
    target_prompt: str = ""
    prior_prompt: str = ""
    # Training hyperparameters
    repeats_per_target: int = 5
    num_train_epochs: int = 20
    train_batch_size: int = 4
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 1
    max_train_steps: Optional[int] = None
    # Image settings
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = False
    # Checkpointing and validation
    checkpoint_dir: str = ""
    log_dir: str = ""
    save_steps: int = 250
    validation_steps: int = 1000
    num_validation_images: int = 4
    validation_prompt: str = ""
    lora_scale: float = 1.0  # LoRA scale for validation generation
    # Optimization
    mixed_precision: str = "fp16"
    enable_xformers_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    pin_memory: bool = False
    use_tensorboard: bool = True
    # Noise scheduler
    noise_scheduler: str = "ddpm"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    num_train_timesteps: int = 1000
    def __post_init__(self):
        """Validate training parameters."""
        if self.mode != "prior_based":
            raise ValueError(f"mode must be 'prior_based', got {self.mode}")
        if not self.target_pathology:
            raise ValueError("target_pathology cannot be empty")
        if self.train_batch_size < 1:
            raise ValueError(f"train_batch_size must be >= 1, got {self.train_batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.resolution not in [256, 512, 1024]:
            raise ValueError(f"resolution must be 256, 512, or 1024, got {self.resolution}")
        if self.mixed_precision not in ["no", "fp16", "bf16"]:
            raise ValueError(f"mixed_precision must be 'no', 'fp16', or 'bf16', got {self.mixed_precision}")
        if not 0 <= self.lora_scale <= 2:
            raise ValueError(f"lora_scale must be in [0, 2], got {self.lora_scale}")
        # Auto-generate validation prompt if not provided
        if not self.validation_prompt and self.target_prompt:
            self.validation_prompt = self.target_prompt
        # Convert paths to Path objects
        if self.target_images_dir and not isinstance(self.target_images_dir, Path):
            self.target_images_dir = Path(self.target_images_dir)
        if self.prior_images_dir and not isinstance(self.prior_images_dir, Path):
            self.prior_images_dir = Path(self.prior_images_dir)
    @property
    def num_epochs(self):
        """Alias for num_train_epochs for compatibility with training script."""
        return self.num_train_epochs
    # Training mode
    mode: str = "prior_based"
    target_pathology: str = ""
    
    # Data paths
    target_images_dir: str = ""
    target_images_csv: str = ""
    prior_images_dir: str = ""
    prior_images_csv: str = ""
    
    # Prompts
    target_prompt: str = ""
    prior_prompt: str = ""
    
    # Training hyperparameters
    repeats_per_target: int = 5
    num_train_epochs: int = 20
    train_batch_size: int = 4
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 1
    max_train_steps: Optional[int] = None
    
    # Image settings
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = False
    
    # Checkpointing and validation
    checkpoint_dir: str = ""
    log_dir: str = ""
    save_steps: int = 250
    validation_steps: int = 1000
    num_validation_images: int = 4
    validation_prompt: str = ""
    lora_scale: float = 1.0  # LoRA scale for validation generation
    
    # Optimization
    mixed_precision: str = "fp16"
    enable_xformers_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    pin_memory: bool = False
    use_tensorboard: bool = True
    
    # Noise scheduler
    noise_scheduler: str = "ddpm"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    num_train_timesteps: int = 1000
    
    def __post_init__(self):
        """Validate training parameters."""
        if self.mode != "prior_based":
            raise ValueError(f"mode must be 'prior_based', got {self.mode}")
        
        if not self.target_pathology:
            raise ValueError("target_pathology cannot be empty")
        
        if self.train_batch_size < 1:
            raise ValueError(f"train_batch_size must be >= 1, got {self.train_batch_size}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        
        if self.resolution not in [256, 512, 1024]:
            raise ValueError(f"resolution must be 256, 512, or 1024, got {self.resolution}")
        
        if self.mixed_precision not in ["no", "fp16", "bf16"]:
            raise ValueError(f"mixed_precision must be 'no', 'fp16', or 'bf16', got {self.mixed_precision}")
        
        if not 0 <= self.lora_scale <= 2:
            raise ValueError(f"lora_scale must be in [0, 2], got {self.lora_scale}")
        
        # Auto-generate validation prompt if not provided
        if not self.validation_prompt and self.target_prompt:
            self.validation_prompt = self.target_prompt
        
        # Convert paths to Path objects
        if self.target_images_dir and not isinstance(self.target_images_dir, Path):
            self.target_images_dir = Path(self.target_images_dir)
        if self.prior_images_dir and not isinstance(self.prior_images_dir, Path):
            self.prior_images_dir = Path(self.prior_images_dir)


@dataclass
class PathsConfig:
    """Output paths configuration."""
    output_dir: str
    logging_dir: str
    
    def __post_init__(self):
        """Convert to Path objects."""
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        if not isinstance(self.logging_dir, Path):
            self.logging_dir = Path(self.logging_dir)


@dataclass
class ExperimentConfig:
    """Experiment metadata configuration."""
    name: str
    tags: List[str] = field(default_factory=list)
    seed: int = 42
    
    def __post_init__(self):
        """Validate experiment config."""
        if not self.name:
            raise ValueError("experiment name cannot be empty")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    report_to: str = "tensorboard"
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate logging config."""
        if self.report_to not in ["tensorboard", "wandb", "none"]:
            raise ValueError(f"report_to must be 'tensorboard', 'wandb', or 'none', got {self.report_to}")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")


@dataclass
class DiffusionTrainingConfig:
    """Main configuration container for diffusion training."""
    model: DiffusionTrainingModelConfig
    training: PriorBasedTrainingConfig
    generation: GenerationConfig
    hardware: DiffusionHardwareConfig
    paths: PathsConfig
    experiment: ExperimentConfig
    logging: LoggingConfig
    
    def create_dirs(self):
        """Create necessary output directories."""
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logging_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint dir if specified
        if self.training.checkpoint_dir:
            Path(self.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created output directories:")
        print(f"  Output: {self.paths.output_dir}")
        print(f"  Logs: {self.paths.logging_dir}")
    
    def validate_paths(self):
        """Validate that required data paths exist."""
        # Validate target images
        if not Path(self.training.target_images_dir).exists():
            raise FileNotFoundError(f"Target images directory not found: {self.training.target_images_dir}")
        
        if not Path(self.training.target_images_csv).exists():
            raise FileNotFoundError(f"Target CSV not found: {self.training.target_images_csv}")
        
        # Validate prior images
        if not Path(self.training.prior_images_dir).exists():
            raise FileNotFoundError(f"Prior images directory not found: {self.training.prior_images_dir}")
        
        if not Path(self.training.prior_images_csv).exists():
            raise FileNotFoundError(f"Prior CSV not found: {self.training.prior_images_csv}")
        
        print(f"✓ Data paths validation passed")


def load_diffusion_training_config(config_path: str) -> DiffusionTrainingConfig:
    """
    Load diffusion training configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        DiffusionTrainingConfig object with all settings
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config has missing or invalid fields
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Validate top-level sections
    required_sections = ['model', 'training', 'paths', 'experiment']
    missing_sections = [s for s in required_sections if s not in raw_config]
    if missing_sections:
        raise ValueError(
            f"Config file missing required sections: {', '.join(missing_sections)}\n"
            f"  Config file: {config_path}"
        )
    
    try:
        # Parse model config (handle nested vae)
        model_raw = raw_config['model'].copy()
        if 'vae' in model_raw and isinstance(model_raw['vae'], dict):
            model_raw['vae_path'] = model_raw['vae'].get('path')
            del model_raw['vae']
        
        # Parse LoRA config if present
        if 'lora_rank' in model_raw:
            # Convert flat config to nested LoRA config
            lora_dict = {
                'rank': model_raw.pop('lora_rank', 64),
                'alpha': model_raw.pop('lora_alpha', 32),
                'dropout': model_raw.pop('lora_dropout', 0.1),
                'target_modules': model_raw.pop('lora_target_modules', ["to_k", "to_q", "to_v", "to_out.0"]),
                'bias': model_raw.pop('lora_bias', "none"),
            }
            model_raw['lora'] = lora_dict
        
        model = DiffusionTrainingModelConfig(**model_raw)
        
        # Parse training config
        training = PriorBasedTrainingConfig(**raw_config['training'])
        
        # Parse generation config
        gen_raw = raw_config.get('generation', {})
        gen_raw.setdefault('num_images', 4)
        gen_raw.setdefault('num_inference_steps', 50)
        gen_raw.setdefault('guidance_scale', 7.5)
        gen_raw.setdefault('lora_scale', 1.0)
        generation = GenerationConfig(**gen_raw)
        
        # Parse hardware config
        hw_raw = raw_config.get('hardware', {})
        hw_raw.setdefault('enable_attention_slicing', True)
        hw_raw.setdefault('enable_vae_slicing', True)
        hw_raw.setdefault('device', 'cuda')
        hw_raw.setdefault('pin_memory', False)
        hw_raw.setdefault('deterministic', False)
        hardware = DiffusionHardwareConfig(**hw_raw)
        
        # Parse paths config
        paths = PathsConfig(**raw_config['paths'])
        
        # Parse experiment config
        experiment = ExperimentConfig(**raw_config['experiment'])
        
        # Parse logging config
        logging = LoggingConfig(**raw_config.get('logging', {}))
        
        # Create main config object
        config = DiffusionTrainingConfig(
            model=model,
            training=training,
            generation=generation,
            hardware=hardware,
            paths=paths,
            experiment=experiment,
            logging=logging
        )
        
    except TypeError as e:
        raise ValueError(
            f"Config file has missing or invalid fields:\n"
            f"  Error: {e}\n"
            f"  Config file: {config_path}"
        )
    
    return config

