"""
Tests for prior-based diffusion training configuration and integration.

Run with:
    pytest tests/test_diffusion_training.py -v
"""

import pytest
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture
def training_config():
    """Load prior-based training config for fibrosis."""
    config_path = Path("configs/config_diffusion_fibrosis.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TestPriorBasedTrainingConfig:
    """Test prior-based training configuration."""
    
    def test_config_file_exists(self):
        """Test that prior-based training config exists."""
        config_path = Path("configs/config_diffusion_fibrosis.yaml")
        assert config_path.exists()
    
    def test_config_loads(self, training_config):
        """Test that config loads successfully."""
        assert training_config is not None
    
    def test_has_required_sections(self, training_config):
        """Test that config has all required sections."""
        assert 'model' in training_config
        assert 'training' in training_config
        assert 'paths' in training_config
        assert 'experiment' in training_config
        assert 'logging' in training_config
    
    def test_training_mode_is_prior_based(self, training_config):
        """Test that training mode is set to prior_based."""
        assert training_config['training']['mode'] == 'prior_based'
    
    def test_has_target_pathology(self, training_config):
        """Test that target pathology is specified."""
        assert 'target_pathology' in training_config['training']
        assert training_config['training']['target_pathology'] == 'fibrosis'
    
    def test_has_data_paths(self, training_config):
        """Test that config has all required data paths."""
        training = training_config['training']
        
        assert 'target_images_dir' in training
        assert 'target_images_csv' in training
        assert 'prior_images_dir' in training
        assert 'prior_images_csv' in training
    
    def test_has_prompts(self, training_config):
        """Test that config has target and prior prompts."""
        training = training_config['training']
        
        assert 'target_prompt' in training
        assert 'prior_prompt' in training
        
        # Verify prompt content
        assert 'fibrosis' in training['target_prompt'].lower()
        assert training['prior_prompt'] == "a chest x-ray"
    
    def test_has_lora_configuration(self, training_config):
        """Test that LoRA configuration is present and valid."""
        training = training_config['training']
        
        assert 'lora' in training
        lora = training['lora']
        
        assert 'rank' in lora
        assert 'alpha' in lora
        assert 'dropout' in lora
        assert 'target_modules' in lora
        
        # Verify values are reasonable
        assert lora['rank'] > 0
        assert lora['alpha'] > 0
        assert 0 <= lora['dropout'] <= 1
        assert isinstance(lora['target_modules'], list)
        assert len(lora['target_modules']) > 0
    
    def test_has_training_parameters(self, training_config):
        """Test that training parameters are configured."""
        training = training_config['training']
        
        assert 'num_epochs' in training
        assert 'batch_size' in training
        assert 'learning_rate' in training
        assert 'gradient_accumulation_steps' in training
        assert 'repeats_per_target' in training
        
        # Verify values are reasonable
        assert training['num_epochs'] > 0
        assert training['batch_size'] > 0
        assert training['learning_rate'] > 0
        assert training['gradient_accumulation_steps'] > 0
        assert training['repeats_per_target'] > 0
    
    def test_has_image_settings(self, training_config):
        """Test that image settings are configured."""
        training = training_config['training']
        
        assert 'resolution' in training
        assert 'center_crop' in training
        assert 'random_flip' in training
        
        assert training['resolution'] in [256, 512, 1024]  # Common sizes
        assert training['random_flip'] == False  # Medical images shouldn't flip
    
    def test_has_checkpoint_settings(self, training_config):
        """Test that checkpoint and validation settings are configured."""
        training = training_config['training']
        
        assert 'save_steps' in training
        assert 'validation_steps' in training
        assert 'num_validation_images' in training
        assert 'validation_prompt' in training
        
        assert training['save_steps'] > 0
        assert training['validation_steps'] > 0
        assert training['num_validation_images'] > 0
    
    def test_has_model_configuration(self, training_config):
        """Test that model configuration is present."""
        model = training_config['model']
        
        assert 'pretrained_model' in model
        assert isinstance(model['pretrained_model'], str)
        assert len(model['pretrained_model']) > 0
    
    def test_has_output_paths(self, training_config):
        """Test that output paths are configured."""
        paths = training_config['paths']
        
        assert 'output_dir' in paths
        assert 'logging_dir' in paths
        
        # Verify paths use outputs/ directory
        assert 'outputs/' in paths['output_dir']
    
    def test_has_experiment_info(self, training_config):
        """Test that experiment information is configured."""
        experiment = training_config['experiment']
        
        assert 'name' in experiment
        assert 'tags' in experiment
        
        assert isinstance(experiment['tags'], list)
        assert len(experiment['tags']) > 0
    
    def test_has_seed(self, training_config):
        """Test that random seed is configured for reproducibility."""
        assert 'seed' in training_config
        assert isinstance(training_config['seed'], int)


class TestDataPathsExistence:
    """Test that data paths in config actually exist."""
    
    def test_target_data_exists(self, training_config):
        """Test that target (fibrosis) data exists."""
        training = training_config['training']
        
        target_dir = Path(training['target_images_dir'])
        target_csv = Path(training['target_images_csv'])
        
        # These should exist for training to work
        if not target_dir.exists():
            pytest.skip(f"Target data directory not found: {target_dir}")
        if not target_csv.exists():
            pytest.skip(f"Target CSV not found: {target_csv}")
        
        assert target_dir.exists()
        assert target_dir.is_dir()
        assert target_csv.exists()
        assert target_csv.is_file()
    
    def test_prior_data_exists(self, training_config):
        """Test that prior (healthy) data exists."""
        training = training_config['training']
        
        prior_dir = Path(training['prior_images_dir'])
        prior_csv = Path(training['prior_images_csv'])
        
        if not prior_dir.exists():
            pytest.skip(f"Prior data directory not found: {prior_dir}")
        if not prior_csv.exists():
            pytest.skip(f"Prior CSV not found: {prior_csv}")
        
        assert prior_dir.exists()
        assert prior_dir.is_dir()
        assert prior_csv.exists()
        assert prior_csv.is_file()


class TestEvaluationConfig:
    """Test evaluation configuration."""
    
    def test_eval_config_exists(self):
        """Test that evaluation config exists."""
        config_path = Path("configs/config_eval_fibrosis.yaml")
        assert config_path.exists()
    
    def test_eval_config_loads(self):
        """Test that evaluation config loads successfully."""
        config_path = Path("configs/config_eval_fibrosis.yaml")
        
        from src.config.diffusion_config import load_diffusion_eval_config
        config = load_diffusion_eval_config(str(config_path))
        
        assert config is not None
    
    def test_eval_config_has_label_subdir(self):
        """Test that evaluation config has label_subdir for nested structure."""
        config_path = Path("configs/config_eval_fibrosis.yaml")
        
        from src.config.diffusion_config import load_diffusion_eval_config
        config = load_diffusion_eval_config(str(config_path))
        
        assert hasattr(config.data, 'label_subdir')
        assert config.data.label_subdir == "fibrosis"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
