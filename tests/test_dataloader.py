"""pytest tests for dataloader functionality."""
import sys
from pathlib import Path
import pytest
import torch
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data.dataset import ChestXrayDataset, create_dataloaders


@pytest.fixture(scope="module")
def config():
    """Load configuration for tests."""
    return load_config('configs/config.yaml')


@pytest.fixture(scope="module")
def dataset_csv(config):
    """Get path to dataset CSV."""
    csv_path = config.data.processed_dir / 'dataset.csv'
    assert csv_path.exists(), f"Dataset CSV not found: {csv_path}"
    return csv_path


@pytest.fixture(scope="module")
def dataloaders(config):
    """Create train and val dataloaders."""
    return create_dataloaders(config)


class TestChestXrayDataset:
    """Test ChestXrayDataset class."""
    
    def test_dataset_creation_train(self, dataset_csv, config):
        """Test creating train dataset."""
        dataset = ChestXrayDataset(dataset_csv, config, split='train')
        assert len(dataset) > 0, "Train dataset is empty"
        assert dataset.is_training is True, "Train dataset should have is_training=True"
    
    def test_dataset_creation_val(self, dataset_csv, config):
        """Test creating val dataset."""
        dataset = ChestXrayDataset(dataset_csv, config, split='val')
        assert len(dataset) > 0, "Val dataset is empty"
        assert dataset.is_training is False, "Val dataset should have is_training=False"
    
    def test_dataset_creation_test(self, dataset_csv, config):
        """Test creating test dataset."""
        dataset = ChestXrayDataset(dataset_csv, config, split='test')
        assert len(dataset) > 0, "Test dataset is empty"
        assert dataset.is_training is False, "Test dataset should have is_training=False"
    
    def test_dataset_split_filtering(self, dataset_csv, config):
        """Test that dataset correctly filters by split."""
        # Load full CSV
        df = pd.read_csv(dataset_csv)
        
        # Create datasets for each split
        train_dataset = ChestXrayDataset(dataset_csv, config, split='train')
        val_dataset = ChestXrayDataset(dataset_csv, config, split='val')
        test_dataset = ChestXrayDataset(dataset_csv, config, split='test')
        
        # Check counts match
        assert len(train_dataset) == (df['split'] == 'train').sum(), "Train dataset size mismatch"
        assert len(val_dataset) == (df['split'] == 'val').sum(), "Val dataset size mismatch"
        assert len(test_dataset) == (df['split'] == 'test').sum(), "Test dataset size mismatch"
    
    def test_dataset_getitem(self, dataset_csv, config):
        """Test getting items from dataset."""
        dataset = ChestXrayDataset(dataset_csv, config, split='train')
        
        # Get first item
        image, label = dataset[0]
        
        # Check types
        assert isinstance(image, torch.Tensor), "Image should be a tensor"
        assert isinstance(label, int), "Label should be an integer"
        
        # Check shapes
        expected_shape = (config.data.channels, *config.data.image_size)
        assert image.shape == expected_shape, f"Image shape should be {expected_shape}, got {image.shape}"
        
        # Check label values
        assert label in [0, 1], "Label should be 0 or 1"
    
    def test_dataset_labels(self, dataset_csv, config):
        """Test that dataset contains both classes."""
        dataset = ChestXrayDataset(dataset_csv, config, split='train')
        
        labels = dataset.df['label'].values
        unique_labels = set(labels)
        
        assert 0 in unique_labels, "Dataset should contain class 0"
        assert 1 in unique_labels, "Dataset should contain class 1"
    
    def test_class_weights(self, dataset_csv, config):
        """Test class weight calculation."""
        dataset = ChestXrayDataset(dataset_csv, config, split='train')
        weights = dataset.get_class_weights()
        
        assert isinstance(weights, torch.Tensor), "Weights should be a tensor"
        assert len(weights) == 2, "Should have weights for 2 classes"
        assert all(weights > 0), "All weights should be positive"


class TestDataLoaders:
    """Test dataloader creation and functionality."""
    
    def test_dataloader_creation(self, dataloaders):
        """Test that dataloaders are created successfully."""
        train_loader, val_loader = dataloaders
        assert train_loader is not None, "Train loader should not be None"
        assert val_loader is not None, "Val loader should not be None"
    
    def test_dataloader_sizes(self, dataloaders, config):
        """Test dataloader sizes."""
        train_loader, val_loader = dataloaders
        
        assert len(train_loader) > 0, "Train loader should have batches"
        assert len(val_loader) > 0, "Val loader should have batches"
        
        # Check dataset sizes
        assert len(train_loader.dataset) > 0, "Train dataset should have samples"
        assert len(val_loader.dataset) > 0, "Val dataset should have samples"
    
    def test_batch_loading_train(self, dataloaders, config):
        """Test loading a batch from train loader."""
        train_loader, _ = dataloaders
        
        images, labels = next(iter(train_loader))
        
        # Check batch shapes
        assert images.shape[0] == config.training.batch_size, "Batch size mismatch"
        assert images.shape[1:] == (config.data.channels, *config.data.image_size), "Image dimensions mismatch"
        assert labels.shape[0] == config.training.batch_size, "Label batch size mismatch"
        
        # Check data types
        assert images.dtype == torch.float32, "Images should be float32"
        assert labels.dtype == torch.int64, "Labels should be int64"
        
        # Check value ranges
        assert torch.all(labels >= 0) and torch.all(labels <= 1), "Labels should be 0 or 1"
    
    def test_batch_loading_val(self, dataloaders, config):
        """Test loading a batch from val loader."""
        _, val_loader = dataloaders
        
        images, labels = next(iter(val_loader))
        
        # Check batch shapes (may be different if drop_last=False for val)
        assert images.shape[0] <= config.training.batch_size, "Batch size should not exceed config"
        assert images.shape[1:] == (config.data.channels, *config.data.image_size), "Image dimensions mismatch"
        
        # Check data types
        assert images.dtype == torch.float32, "Images should be float32"
        assert labels.dtype == torch.int64, "Labels should be int64"
    
    def test_dataloader_iteration(self, dataloaders):
        """Test iterating through dataloaders."""
        train_loader, val_loader = dataloaders
        
        # Test train loader iteration
        train_batch_count = 0
        for images, labels in train_loader:
            train_batch_count += 1
            if train_batch_count >= 3:  # Test first 3 batches
                break
        
        assert train_batch_count > 0, "Should be able to iterate train loader"
        
        # Test val loader iteration
        val_batch_count = 0
        for images, labels in val_loader:
            val_batch_count += 1
            if val_batch_count >= 3:  # Test first 3 batches
                break
        
        assert val_batch_count > 0, "Should be able to iterate val loader"


class TestDatasetIntegrity:
    """Test dataset integrity and consistency."""
    
    def test_no_overlap_between_splits(self, dataset_csv):
        """Test that there's no overlap between train/val/test splits."""
        df = pd.read_csv(dataset_csv)
        
        train_images = set(df[df['split'] == 'train']['Image Index'])
        val_images = set(df[df['split'] == 'val']['Image Index'])
        test_images = set(df[df['split'] == 'test']['Image Index'])
        
        assert len(train_images & val_images) == 0, "Train and val should not overlap"
        assert len(train_images & test_images) == 0, "Train and test should not overlap"
        assert len(val_images & test_images) == 0, "Val and test should not overlap"
    
    def test_all_images_exist(self, dataset_csv):
        """Test that all image paths in dataset exist."""
        df = pd.read_csv(dataset_csv)
        
        missing_images = []
        for idx, row in df.iterrows():
            image_path = Path(row['image_path'])
            if not image_path.exists():
                missing_images.append(str(image_path))
            
            # Only check first 100 to keep test fast
            if idx >= 100:
                break
        
        assert len(missing_images) == 0, f"Found missing images: {missing_images[:10]}"
    
    def test_label_consistency(self, dataset_csv, config):
        """Test that labels are consistent with Finding Labels."""
        df = pd.read_csv(dataset_csv)
        
        class_pos = config.data.class_positive
        class_neg = config.data.class_negative
        
        for idx, row in df.iterrows():
            finding_labels = row['Finding Labels']
            label = row['label']
            
            if class_pos in finding_labels:
                assert label == 1, f"Row {idx}: {class_pos} should have label=1"
            elif class_neg in finding_labels:
                assert label == 0, f"Row {idx}: {class_neg} should have label=0"
            
            # Only check first 100 to keep test fast
            if idx >= 100:
                break
