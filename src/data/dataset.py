"""PyTorch Dataset for medical image binary classification."""
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from src.config import Config


class AddGaussianNoise:
    """Add Gaussian noise to tensor (medical imaging augmentation)."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        """
        Args:
            mean: Mean of Gaussian noise
            std: Standard deviation of Gaussian noise
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to tensor."""
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
    
    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


class ChestXrayDataset(Dataset):
    """
    Dataset for chest X-ray binary classification.

    Medical imaging considerations:
    - NO horizontal or vertical flips (orientation matters)
    - Grayscale images converted to RGB for pretrained models
    - Minimal augmentations: rotation, brightness, contrast, Gaussian noise
    """

    def __init__(
        self,
        csv_path: Path,
        config: Config,
        split: str = 'train',
        data_root: Optional[Path] = None
    ):
        """
        Args:
            csv_path: Path to unified CSV file with image paths, labels, and split column
            config: Configuration object
            split: Which split to use ('train', 'val', or 'test')
            data_root: Optional root directory for data paths (replaces 'data/' prefix)
        """
        df_full = pd.read_csv(csv_path)
        # Filter for specific split
        self.df = df_full[df_full['split'] == split].reset_index(drop=True)
        self.config = config
        self.split = split
        self.is_training = (split == 'train')
        self.data_root = data_root

        # Build transforms
        self.transform = self._build_transforms()
        
    def _build_transforms(self) -> transforms.Compose:
        """Build augmentation pipeline."""
        aug_config = self.config.augmentation
        img_size = tuple(self.config.data.image_size)
        
        if self.is_training:
            # Training augmentations (medical imaging safe)
            transform_list = [
                transforms.Resize(img_size),
                transforms.RandomRotation(
                    degrees=aug_config.rotation_degrees,
                    fill=0  # Black fill for rotations
                ),
                transforms.ColorJitter(
                    brightness=aug_config.brightness,
                    contrast=aug_config.contrast
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=aug_config.normalize_mean,
                    std=aug_config.normalize_std
                ),
                AddGaussianNoise(
                    mean=0.0,
                    std=aug_config.gaussian_noise_std
                )
            ]
        else:
            # Validation/test (no augmentation, only normalization)
            transform_list = [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=aug_config.normalize_mean,
                    std=aug_config.normalize_std
                )
            ]
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        row = self.df.iloc[idx]

        # Load image - handle data_root override
        image_path = row['image_path']
        if self.data_root and image_path.startswith('data/'):
            image_path = str(self.data_root / image_path[5:])  # Remove 'data/' prefix
        image = Image.open(image_path)
        
        # Convert grayscale to RGB (for pretrained models)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Get label
        label = int(row['label'])
        
        return image_tensor, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            Tensor of class weights [weight_class_0, weight_class_1]
        """
        labels = self.df['label'].values
        class_counts = np.bincount(labels)
        total = len(labels)
        
        # Inverse frequency weighting
        weights = total / (len(class_counts) * class_counts)
        
        return torch.FloatTensor(weights)


def create_dataloaders(
    config: Config,
    data_root: Optional[Path] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        config: Configuration object
        data_root: Optional root directory for data paths (replaces 'data/' prefix in image paths)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Path to unified dataset CSV - try 'dataset.csv' first, then 'train_augmented.csv'
    dataset_csv = config.data.processed_dir / 'dataset.csv'
    if not dataset_csv.exists():
        dataset_csv = config.data.processed_dir / 'train_augmented.csv'

    # Check if CSV exists
    if not dataset_csv.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found in {config.data.processed_dir}\n"
            f"Looked for: dataset.csv, train_augmented.csv"
        )

    # Create datasets for train and val splits
    train_dataset = ChestXrayDataset(dataset_csv, config, split='train', data_root=data_root)
    val_dataset = ChestXrayDataset(dataset_csv, config, split='val', data_root=data_root)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.hardware.pin_memory,
        drop_last=True  # Drop incomplete batches for stable training
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.hardware.pin_memory
    )
    
    return train_loader, val_loader
