"""
Unit tests for diffusion dataset with new prior-based architecture.

Tests cover:
- Dataset loading with label_subdir (nested structure)
- Evaluation config loading
- Prior-based training dataset structure
- Flat vs nested directory handling

Run with:
    pytest tests/test_diffusion_dataset.py -v
"""

import pytest
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.diffusion_dataset import ChestXrayDiffusionDataset, collate_fn
from src.config.diffusion_config import load_diffusion_eval_config


@pytest.fixture
def eval_config():
    """Load evaluation config for fibrosis."""
    config_path = Path("configs/config_eval_fibrosis.yaml")
    return load_diffusion_eval_config(str(config_path))


@pytest.fixture
def training_config():
    """Load prior-based training config for fibrosis."""
    config_path = Path("configs/config_diffusion_fibrosis.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TestEvaluationDataset:
    """Test suite for evaluation dataset with nested directory structure."""
    
    def test_eval_config_loads(self, eval_config):
        """Test that evaluation config loads successfully."""
        assert eval_config is not None
        assert hasattr(eval_config, 'data')
        assert hasattr(eval_config.data, 'label_subdir')
    
    def test_eval_config_has_label_subdir(self, eval_config):
        """Test that evaluation config has label_subdir field."""
        assert eval_config.data.label_subdir == "fibrosis"
    
    def test_eval_dataset_with_label_subdir(self, eval_config):
        """Test dataset creation with label_subdir for nested structure."""
        data_config = eval_config.data
        
        # Use local path for testing
        local_data_dir = Path("data/pure_class_folders")
        local_csv = local_data_dir / "fibrosis_images.csv"
        
        if not local_csv.exists():
            pytest.skip(f"Test data not available at {local_csv}")
        
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(local_csv),
            data_dir=str(local_data_dir),
            image_size=data_config.image_size,
            prompt_template=data_config.prompt_template,
            center_crop=data_config.center_crop,
            random_flip=data_config.random_flip,
            label_subdir=data_config.label_subdir,  # NEW: subdirectory support
        )
        
        assert len(dataset) > 0
        print(f"Loaded {len(dataset)} images from nested structure")
    
    def test_eval_dataset_sample_structure(self, eval_config):
        """Test that evaluation dataset sample has correct structure."""
        local_data_dir = Path("data/pure_class_folders")
        local_csv = local_data_dir / "fibrosis_images.csv"
        
        if not local_csv.exists():
            pytest.skip(f"Test data not available at {local_csv}")
        
        data_config = eval_config.data
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(local_csv),
            data_dir=str(local_data_dir),
            image_size=data_config.image_size,
            prompt_template=data_config.prompt_template,
            center_crop=data_config.center_crop,
            random_flip=data_config.random_flip,
            label_subdir=data_config.label_subdir,
        )
        
        sample = dataset[0]
        
        assert 'pixel_values' in sample
        assert 'text' in sample
        assert 'image_path' in sample
        assert isinstance(sample['pixel_values'], torch.Tensor)
        assert isinstance(sample['text'], str)
    
    def test_eval_dataset_fixed_prompt(self, eval_config):
        """Test that evaluation dataset uses fixed prompt (no {labels} placeholder)."""
        local_data_dir = Path("data/pure_class_folders")
        local_csv = local_data_dir / "fibrosis_images.csv"
        
        if not local_csv.exists():
            pytest.skip(f"Test data not available at {local_csv}")
        
        data_config = eval_config.data
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(local_csv),
            data_dir=str(local_data_dir),
            image_size=data_config.image_size,
            prompt_template=data_config.prompt_template,
            center_crop=data_config.center_crop,
            random_flip=data_config.random_flip,
            label_subdir=data_config.label_subdir,
        )
        
        # All prompts should be identical (fixed)
        prompts = set()
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            prompts.add(sample['text'])
        
        assert len(prompts) == 1  # All should be the same
        assert "fibrosis" in list(prompts)[0].lower()
    
    def test_eval_dataset_image_shape(self, eval_config):
        """Test that evaluation dataset images have correct shape."""
        local_data_dir = Path("data/pure_class_folders")
        local_csv = local_data_dir / "fibrosis_images.csv"
        
        if not local_csv.exists():
            pytest.skip(f"Test data not available at {local_csv}")
        
        data_config = eval_config.data
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(local_csv),
            data_dir=str(local_data_dir),
            image_size=data_config.image_size,
            prompt_template=data_config.prompt_template,
            center_crop=data_config.center_crop,
            random_flip=data_config.random_flip,
            label_subdir=data_config.label_subdir,
        )
        
        sample = dataset[0]
        pixel_values = sample['pixel_values']
        
        expected_size = data_config.image_size
        assert pixel_values.shape == (3, expected_size, expected_size)


class TestPriorBasedTrainingDataset:
    """Test suite for prior-based training dataset structure."""
    
    def test_training_config_loads(self, training_config):
        """Test that prior-based training config loads successfully."""
        assert training_config is not None
        assert 'training' in training_config
        assert 'mode' in training_config['training']
        assert training_config['training']['mode'] == 'prior_based'
    
    def test_training_config_has_prior_paths(self, training_config):
        """Test that training config has prior-based paths."""
        training = training_config['training']
        
        assert 'target_images_dir' in training
        assert 'target_images_csv' in training
        assert 'prior_images_dir' in training
        assert 'prior_images_csv' in training
    
    def test_training_config_has_prompts(self, training_config):
        """Test that training config has target and prior prompts."""
        training = training_config['training']
        
        assert 'target_prompt' in training
        assert 'prior_prompt' in training
        assert training['target_prompt'] == "a chest x-ray with fibrosis"
        assert training['prior_prompt'] == "a chest x-ray"
    
    def test_training_config_has_lora_settings(self, training_config):
        """Test that training config has LoRA settings."""
        training = training_config['training']
        
        assert 'lora' in training
        assert 'rank' in training['lora']
        assert 'alpha' in training['lora']
        assert 'target_modules' in training['lora']
    
    def test_target_dataset_can_be_created(self, training_config):
        """Test that target dataset (fibrosis) can be created."""
        training = training_config['training']
        
        target_dir = Path(training['target_images_dir'])
        target_csv = Path(training['target_images_csv'])
        
        if not target_csv.exists():
            pytest.skip(f"Test data not available at {target_csv}")
        
        # For prior-based training, target images are in flat directory
        # (target_images_dir contains images directly, no subdirectory)
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(target_csv),
            data_dir=str(target_dir),
            image_size=training['resolution'],
            prompt_template=training['target_prompt'],
            center_crop=training['center_crop'],
            random_flip=training['random_flip'],
            label_subdir=None,  # Flat structure for training
        )
        
        assert len(dataset) > 0
        print(f"Loaded {len(dataset)} target (fibrosis) images")
    
    def test_prior_dataset_can_be_created(self, training_config):
        """Test that prior dataset (healthy) can be created."""
        training = training_config['training']
        
        prior_dir = Path(training['prior_images_dir'])
        prior_csv = Path(training['prior_images_csv'])
        
        if not prior_csv.exists():
            pytest.skip(f"Test data not available at {prior_csv}")
        
        # Prior images also in flat directory
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(prior_csv),
            data_dir=str(prior_dir),
            image_size=training['resolution'],
            prompt_template=training['prior_prompt'],
            center_crop=training['center_crop'],
            random_flip=training['random_flip'],
            label_subdir=None,  # Flat structure
        )
        
        assert len(dataset) > 0
        print(f"Loaded {len(dataset)} prior (healthy) images")


class TestDatasetFlexibility:
    """Test that dataset handles both flat and nested structures."""
    
    def test_flat_structure_without_label_subdir(self):
        """Test dataset with flat directory structure (label_subdir=None)."""
        # This would be used for training data in flat directories
        # Skip if test data not available
        config_path = Path("configs/config_diffusion_fibrosis.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        target_csv = Path(config['training']['target_images_csv'])
        if not target_csv.exists():
            pytest.skip(f"Test data not available at {target_csv}")
        
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(target_csv),
            data_dir=config['training']['target_images_dir'],
            image_size=config['training']['resolution'],
            prompt_template=config['training']['target_prompt'],
            center_crop=config['training']['center_crop'],
            random_flip=config['training']['random_flip'],
            label_subdir=None,  # Flat structure
        )
        
        assert len(dataset) > 0
    
    def test_nested_structure_with_label_subdir(self):
        """Test dataset with nested directory structure (label_subdir='fibrosis')."""
        # This is used for evaluation data
        local_data_dir = Path("data/pure_class_folders")
        local_csv = local_data_dir / "fibrosis_images.csv"
        
        if not local_csv.exists():
            pytest.skip(f"Test data not available at {local_csv}")
        
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(local_csv),
            data_dir=str(local_data_dir),
            image_size=512,
            prompt_template="A chest X-ray showing fibrosis",
            center_crop=True,
            random_flip=False,
            label_subdir="fibrosis",  # Nested structure
        )
        
        assert len(dataset) > 0


class TestCollateFunction:
    """Test suite for collate_fn."""
    
    def test_collate_with_single_sample(self):
        """Test collate_fn with a single sample."""
        local_data_dir = Path("data/pure_class_folders")
        local_csv = local_data_dir / "fibrosis_images.csv"
        
        if not local_csv.exists():
            pytest.skip(f"Test data not available at {local_csv}")
        
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(local_csv),
            data_dir=str(local_data_dir),
            image_size=512,
            prompt_template="A chest X-ray showing fibrosis",
            center_crop=True,
            random_flip=False,
            label_subdir="fibrosis",
        )
        
        sample = dataset[0]
        batch = collate_fn([sample])
        
        assert 'pixel_values' in batch
        assert 'text' in batch
        assert 'image_paths' in batch
        
        assert batch['pixel_values'].shape[0] == 1
        assert len(batch['text']) == 1
    
    def test_collate_with_multiple_samples(self):
        """Test collate_fn with multiple samples."""
        local_data_dir = Path("data/pure_class_folders")
        local_csv = local_data_dir / "fibrosis_images.csv"
        
        if not local_csv.exists():
            pytest.skip(f"Test data not available at {local_csv}")
        
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(local_csv),
            data_dir=str(local_data_dir),
            image_size=512,
            prompt_template="A chest X-ray showing fibrosis",
            center_crop=True,
            random_flip=False,
            label_subdir="fibrosis",
        )
        
        samples = [dataset[i] for i in range(min(4, len(dataset)))]
        batch = collate_fn(samples)
        
        assert batch['pixel_values'].shape[0] == len(samples)
        assert len(batch['text']) == len(samples)
        assert len(batch['image_paths']) == len(samples)


class TestDataLoader:
    """Test DataLoader integration."""
    
    def test_dataloader_creation(self):
        """Test that DataLoader works with current dataset structure."""
        local_data_dir = Path("data/pure_class_folders")
        local_csv = local_data_dir / "fibrosis_images.csv"
        
        if not local_csv.exists():
            pytest.skip(f"Test data not available at {local_csv}")
        
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(local_csv),
            data_dir=str(local_data_dir),
            image_size=512,
            prompt_template="A chest X-ray showing fibrosis",
            center_crop=True,
            random_flip=False,
            label_subdir="fibrosis",
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        assert dataloader is not None
        assert len(dataloader) > 0
    
    def test_dataloader_batch_structure(self):
        """Test that DataLoader produces valid batches."""
        local_data_dir = Path("data/pure_class_folders")
        local_csv = local_data_dir / "fibrosis_images.csv"
        
        if not local_csv.exists():
            pytest.skip(f"Test data not available at {local_csv}")
        
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(local_csv),
            data_dir=str(local_data_dir),
            image_size=512,
            prompt_template="A chest X-ray showing fibrosis",
            center_crop=True,
            random_flip=False,
            label_subdir="fibrosis",
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        batch = next(iter(dataloader))
        
        assert 'pixel_values' in batch
        assert 'text' in batch
        assert 'image_paths' in batch
        
        assert batch['pixel_values'].shape == (4, 3, 512, 512)
        assert len(batch['text']) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
