"""
PyTorch Dataset for Stable Diffusion fine-tuning on chest X-rays.
"""

import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXrayDiffusionDataset(Dataset):
    """
    Dataset for loading chest X-rays with text prompts for diffusion model training.
    
    Args:
        csv_file: Path to CSV with columns ['Image Index', 'Finding Labels']
        data_dir: Directory containing images
        image_size: Target image size (height, width)
        prompt_template: Template string with {labels} placeholder
        center_crop: Whether to center crop images
        random_flip: Whether to apply random horizontal flip (False for medical images!)
    """
    
    def __init__(
        self,
        csv_file: str,
        data_dir: str,
        image_size: int = 512,
        prompt_template: str = "A chest X-ray with {labels}",
        center_crop: bool = True,
        random_flip: bool = False,
        label_subdir: str = None,
    ):
        self.data_dir = Path(data_dir)
        self.label_subdir = label_subdir
        self.prompt_template = prompt_template
        
        # Load CSV
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} images from {csv_file}")
        
        # Image transforms
        transform_list = []
        
        # Resize to slightly larger than target
        if center_crop:
            transform_list.append(transforms.Resize(int(image_size * 1.1)))
            transform_list.append(transforms.CenterCrop(image_size))
        else:
            transform_list.append(transforms.Resize((image_size, image_size)))
        
        # Random flip (disabled for medical images by default)
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # Convert to tensor and normalize to [-1, 1] for SD
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Convert [0, 1] to [-1, 1]
        ])
        
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'pixel_values': Tensor of shape (3, H, W) normalized to [-1, 1]
                - 'text': Text prompt string
                - 'image_path': Original image path
        """
        row = self.df.iloc[idx]
        
        # Load image
        image_name = row['Image Index']
        # If label_subdir is specified, images are in data_dir/label_subdir/
        if self.label_subdir:
            image_path = self.data_dir / self.label_subdir / image_name
        else:
            image_path = self.data_dir / image_name
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (512, 512), color='black')
        
        # Apply transforms
        pixel_values = self.transform(image)
        
        # Create text prompt from labels
        labels = row['Finding Labels']
        # Convert pipe-separated labels to comma-separated
        # "Fibrosis|Pneumonia" -> "Fibrosis and Pneumonia"
        if '|' in labels:
            label_list = labels.split('|')
            labels_formatted = ' and '.join(label_list)
        else:
            labels_formatted = labels
        
        text_prompt = self.prompt_template.format(labels=labels_formatted)
        
        return {
            'pixel_values': pixel_values,
            'text': text_prompt,
            'image_path': str(image_path)
        }


def collate_fn(examples):
    """
    Collate function for DataLoader.
    
    Args:
        examples: List of dicts from __getitem__
    
    Returns:
        dict with batched tensors and lists
    """
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    texts = [example['text'] for example in examples]
    image_paths = [example['image_path'] for example in examples]
    
    return {
        'pixel_values': pixel_values,
        'text': texts,
        'image_paths': image_paths
    }


if __name__ == "__main__":
    # Test dataset
    dataset = ChestXrayDiffusionDataset(
        csv_file="data/diffusion_data/diffusion_dataset_balanced.csv",
        data_dir="data/diffusion_data",
        image_size=512,
        prompt_template="A chest X-ray with {labels}",
        center_crop=True,
        random_flip=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first item
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['pixel_values'].shape}")
    print(f"  Image range: [{sample['pixel_values'].min():.2f}, {sample['pixel_values'].max():.2f}]")
    print(f"  Text: {sample['text']}")
    print(f"  Path: {sample['image_path']}")
    
    # Test collate
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch:")
    print(f"  pixel_values shape: {batch['pixel_values'].shape}")
    print(f"  texts: {batch['text']}")
