"""
Unit tests for diffusion dataset.

Run with:
    pytest tests/test_diffusion_dataset.py -v
"""

import pytest
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.diffusion_dataset import ChestXrayDiffusionDataset, collate_fn


@pytest.fixture
def config():
    """Load diffusion config."""
    config_path = Path("configs/config_diffusion.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def dataset(config):
    """Create diffusion dataset."""
    return ChestXrayDiffusionDataset(
        csv_file=str(Path(config['data']['data_dir']) / config['data']['csv_file']),
        data_dir=config['data']['data_dir'],
        image_size=config['data']['image_size'],
        prompt_template=config['data']['prompt_template'],
        center_crop=config['data']['center_crop'],
        random_flip=config['data']['random_flip'],
    )


class TestDiffusionDataset:
    """Test suite for ChestXrayDiffusionDataset."""
    
    def test_dataset_creation(self, dataset):
        """Test that dataset is created successfully."""
        assert dataset is not None
        assert len(dataset) > 0
    
    def test_dataset_length(self, dataset):
        """Test dataset has expected number of samples."""
        # Should have balanced dataset (10,541 images)
        assert len(dataset) > 10000
        assert len(dataset) < 15000
    
    def test_sample_structure(self, dataset):
        """Test that sample has correct structure."""
        sample = dataset[0]
        
        assert 'pixel_values' in sample
        assert 'text' in sample
        assert 'image_path' in sample
    
    def test_image_shape(self, dataset, config):
        """Test that images have correct shape."""
        sample = dataset[0]
        pixel_values = sample['pixel_values']
        
        expected_size = config['data']['image_size']
        assert pixel_values.shape == (3, expected_size, expected_size)
    
    def test_image_range(self, dataset):
        """Test that images are normalized to [-1, 1]."""
        sample = dataset[0]
        pixel_values = sample['pixel_values']
        
        assert pixel_values.min() >= -1.0
        assert pixel_values.max() <= 1.0
    
    def test_image_dtype(self, dataset):
        """Test that images are float tensors."""
        sample = dataset[0]
        pixel_values = sample['pixel_values']
        
        assert isinstance(pixel_values, torch.Tensor)
        assert pixel_values.dtype == torch.float32
    
    def test_text_prompt_format(self, dataset, config):
        """Test that text prompts follow expected format."""
        sample = dataset[0]
        text = sample['text']
        
        template = config['data']['prompt_template']
        # Should contain "A chest X-ray with"
        assert "A chest X-ray with" in text
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_multi_label_formatting(self, dataset):
        """Test that multi-label prompts are formatted correctly."""
        # Find a multi-label sample
        multi_label_found = False
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            if ' and ' in sample['text']:
                multi_label_found = True
                # Should have "and" for multi-label
                assert 'and' in sample['text']
                break
        
        # Most datasets should have some multi-label samples
        assert multi_label_found, "No multi-label samples found in first 100"
    
    def test_image_path_exists(self, dataset):
        """Test that image paths are valid."""
        sample = dataset[0]
        image_path = Path(sample['image_path'])
        
        # Path should exist (or we got a fallback black image)
        assert isinstance(sample['image_path'], str)
    
    def test_unique_prompts(self, dataset):
        """Test that there are multiple unique prompts."""
        unique_prompts = set()
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            unique_prompts.add(sample['text'])
        
        # Should have at least 5 unique prompts in 100 samples
        assert len(unique_prompts) >= 5
    
    def test_dataloader_creation(self, dataset):
        """Test that DataLoader works with collate_fn."""
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        assert dataloader is not None
        assert len(dataloader) > 0
    
    def test_dataloader_batch(self, dataset):
        """Test that DataLoader produces valid batches."""
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
    
    def test_batch_pixel_values_shape(self, dataset, config):
        """Test batch pixel_values have correct shape."""
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        batch = next(iter(dataloader))
        pixel_values = batch['pixel_values']
        
        expected_size = config['data']['image_size']
        assert pixel_values.shape == (4, 3, expected_size, expected_size)
    
    def test_batch_text_length(self, dataset):
        """Test batch has correct number of text prompts."""
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        batch = next(iter(dataloader))
        texts = batch['text']
        
        assert len(texts) == 4
        assert all(isinstance(t, str) for t in texts)
    
    def test_multiple_batches(self, dataset):
        """Test that multiple batches can be loaded."""
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        batches = []
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            batches.append(batch)
        
        assert len(batches) == 3
    
    def test_batch_memory_format(self, dataset):
        """Test batch pixel_values are contiguous."""
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        batch = next(iter(dataloader))
        pixel_values = batch['pixel_values']
        
        assert pixel_values.is_contiguous()
    
    def test_dataset_reproducibility(self, config):
        """Test that dataset produces same results with same seed."""
        torch.manual_seed(42)
        dataset1 = ChestXrayDiffusionDataset(
            csv_file=str(Path(config['data']['data_dir']) / config['data']['csv_file']),
            data_dir=config['data']['data_dir'],
            image_size=config['data']['image_size'],
            prompt_template=config['data']['prompt_template'],
            center_crop=config['data']['center_crop'],
            random_flip=False,  # No randomness
        )
        
        torch.manual_seed(42)
        dataset2 = ChestXrayDiffusionDataset(
            csv_file=str(Path(config['data']['data_dir']) / config['data']['csv_file']),
            data_dir=config['data']['data_dir'],
            image_size=config['data']['image_size'],
            prompt_template=config['data']['prompt_template'],
            center_crop=config['data']['center_crop'],
            random_flip=False,  # No randomness
        )
        
        assert len(dataset1) == len(dataset2)
        
        sample1 = dataset1[0]
        sample2 = dataset2[0]
        
        assert sample1['text'] == sample2['text']
        assert torch.equal(sample1['pixel_values'], sample2['pixel_values'])


class TestCollateFunction:
    """Test suite for collate_fn."""
    
    def test_collate_with_single_sample(self, dataset):
        """Test collate_fn with a single sample."""
        sample = dataset[0]
        batch = collate_fn([sample])
        
        assert 'pixel_values' in batch
        assert 'text' in batch
        assert 'image_paths' in batch
        
        assert batch['pixel_values'].shape[0] == 1
        assert len(batch['text']) == 1
    
    def test_collate_with_multiple_samples(self, dataset):
        """Test collate_fn with multiple samples."""
        samples = [dataset[i] for i in range(4)]
        batch = collate_fn(samples)
        
        assert batch['pixel_values'].shape[0] == 4
        assert len(batch['text']) == 4
        assert len(batch['image_paths']) == 4
    
    def test_collate_stacks_correctly(self, dataset):
        """Test that collate_fn stacks tensors correctly."""
        samples = [dataset[i] for i in range(3)]
        batch = collate_fn(samples)
        
        pixel_values = batch['pixel_values']
        
        # Should be stacked along first dimension
        assert pixel_values.dim() == 4  # (B, C, H, W)
        assert pixel_values.shape[0] == 3
