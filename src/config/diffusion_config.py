"""
Diffusion model configuration manager with dataclasses for type safety.

This module provides configuration management for:
- Diffusion model training
- Diffusion model evaluation
- Image generation

Reuses common components from config_manager.py (HardwareConfig).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
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
    negative_prompt: str = "blurry, low quality, distorted, artifacts"
    
    def __post_init__(self):
        """Validate generation parameters."""
        if self.num_images < 1:
            raise ValueError(f"num_images must be >= 1, got {self.num_images}")
        
        if self.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {self.num_inference_steps}")
        
        if self.guidance_scale < 0:
            raise ValueError(f"guidance_scale must be >= 0, got {self.guidance_scale}")


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
