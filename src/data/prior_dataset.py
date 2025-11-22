"""
Prior-based Dataset for Diffusion Training

This dataset pairs target pathology images with healthy "prior" images for training.
Each target image is repeated multiple times with different healthy priors.
"""

import os
import random
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PriorBasedDiffusionDataset(Dataset):
    """
    Dataset that pairs target pathology images with healthy prior images.
    
    Each target image is repeated 'repeats_per_target' times, each time paired
    with a different randomly selected healthy image as the "prior".
    """
    
    def __init__(
        self,
        target_images_dir,
        target_images_csv,
        prior_images_dir, 
        prior_images_csv,
        target_prompt,
        prior_prompt,
        repeats_per_target=10,
        resolution=512,
        center_crop=True,
        random_flip=False,
        tokenizer=None,
        seed=42
    ):
        self.target_images_dir = Path(target_images_dir)
        self.prior_images_dir = Path(prior_images_dir)
        self.target_prompt = target_prompt
        self.prior_prompt = prior_prompt
        self.repeats_per_target = repeats_per_target
        self.tokenizer = tokenizer
        
        # Set random seed for reproducible prior selection
        random.seed(seed)
        
        # Load target images (e.g., fibrosis)
        self.target_df = pd.read_csv(target_images_csv)
        self.target_images = self.target_df['Image Index'].tolist()
        
        # Load prior images (healthy)
        self.prior_df = pd.read_csv(prior_images_csv)
        self.prior_images = self.prior_df['Image Index'].tolist()
        
        print(f"Loaded {len(self.target_images)} target images")
        print(f"Loaded {len(self.prior_images)} prior images") 
        print(f"Total training samples: {len(self.target_images)} × {repeats_per_target} = {len(self.target_images) * repeats_per_target}")
        
        # Create image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])
        
        # Pre-generate pairings for reproducibility
        self.pairings = []
        for target_img in self.target_images:
            # For each target image, create 'repeats_per_target' pairings
            for _ in range(repeats_per_target):
                prior_img = random.choice(self.prior_images)
                self.pairings.append((target_img, prior_img))
        
        print(f"Generated {len(self.pairings)} target-prior pairings")
    
    def __len__(self):
        return len(self.pairings)
    
    def __getitem__(self, idx):
        target_img_name, prior_img_name = self.pairings[idx]
        
        # Load target image
        target_img_path = self.target_images_dir / target_img_name
        target_image = Image.open(target_img_path).convert("RGB")
        target_image = self.image_transforms(target_image)
        
        # Load prior image  
        prior_img_path = self.prior_images_dir / prior_img_name
        prior_image = Image.open(prior_img_path).convert("RGB")
        prior_image = self.image_transforms(prior_image)
        
        # Tokenize prompts if tokenizer provided
        if self.tokenizer is not None:
            target_prompt_ids = self.tokenizer(
                self.target_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            
            prior_prompt_ids = self.tokenizer(
                self.prior_prompt,
                truncation=True,
                padding="max_length", 
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        else:
            target_prompt_ids = self.target_prompt
            prior_prompt_ids = self.prior_prompt
        
        return {
            "target_image": target_image,
            "prior_image": prior_image,
            "target_prompt": self.target_prompt,
            "prior_prompt": self.prior_prompt,
            "target_prompt_ids": target_prompt_ids,
            "prior_prompt_ids": prior_prompt_ids,
            "target_image_name": target_img_name,
            "prior_image_name": prior_img_name,
        }


def collate_fn(examples):
    """Custom collate function for prior-based training."""
    target_images = torch.stack([example["target_image"] for example in examples])
    prior_images = torch.stack([example["prior_image"] for example in examples])
    
    if isinstance(examples[0]["target_prompt_ids"], torch.Tensor):
        target_prompt_ids = torch.stack([example["target_prompt_ids"] for example in examples])
        prior_prompt_ids = torch.stack([example["prior_prompt_ids"] for example in examples])
    else:
        target_prompt_ids = [example["target_prompt_ids"] for example in examples]
        prior_prompt_ids = [example["prior_prompt_ids"] for example in examples]
    
    return {
        "target_images": target_images,
        "prior_images": prior_images,
        "target_prompts": [example["target_prompt"] for example in examples],
        "prior_prompts": [example["prior_prompt"] for example in examples], 
        "target_prompt_ids": target_prompt_ids,
        "prior_prompt_ids": prior_prompt_ids,
        "target_image_names": [example["target_image_name"] for example in examples],
        "prior_image_names": [example["prior_image_name"] for example in examples],
    }