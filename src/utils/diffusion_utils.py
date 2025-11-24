"""
Utilities for diffusion model loading and image generation.

This module provides reusable functions for:
- Loading Stable Diffusion pipelines with LoRA weights
- Generating images from prompts
- Converting between image formats
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

from .image_utils import pil_to_numpy


def load_pipeline(
    checkpoint_path: str,
    pretrained_model: str,
    enable_attention_slicing: bool = True,
    enable_vae_slicing: bool = True,
    device: Optional[str] = None
) -> StableDiffusionPipeline:
    """
    Load Stable Diffusion pipeline with LoRA weights.
    
    Args:
        checkpoint_path: Path to LoRA checkpoint directory
        pretrained_model: HuggingFace model ID or path to base model
        enable_attention_slicing: Enable memory-efficient attention
        enable_vae_slicing: Enable VAE slicing for memory efficiency
        device: Device to load model on (auto-detects if None)
    
    Returns:
        Loaded StableDiffusionPipeline
    
    Raises:
        FileNotFoundError: If checkpoint path doesn't exist
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}\n"
            f"Please verify the checkpoint path is correct."
        )
    
    print(f"Loading base model: {pretrained_model}")
    
    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    
    # Load LoRA weights
    print(f"Loading LoRA weights from: {checkpoint_path}")
    pipeline.unet = PeftModel.from_pretrained(
        pipeline.unet,
        checkpoint_path,
        is_trainable=False
    )
    
    # Move to GPU
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipeline = pipeline.to(device)
    
    # Enable memory optimizations
    if enable_attention_slicing:
        pipeline.enable_attention_slicing()
    
    if enable_vae_slicing:
        pipeline.enable_vae_slicing()
    
    print(f"Pipeline loaded on {device}")
    
    return pipeline


def generate_images(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    num_images: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: str = "blurry, low quality, distorted, artifacts",
    seed: Optional[int] = None,
    return_numpy: bool = False,
):
    """
    Generate multiple images from a single prompt, each with different seed.
    
    Each image gets a unique seed (seed+i) to ensure diversity.
    
    Args:
        pipeline: Loaded StableDiffusionPipeline
        prompt: Text prompt for generation
        num_images: Number of images to generate
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        negative_prompt: Negative prompt to guide generation away from
        seed: Base random seed (each image gets seed+i for diversity)
        return_numpy: If True, return numpy arrays; if False, return PIL Images
    
    Returns:
        List[np.ndarray] if return_numpy=True, else List[Image.Image]
    """
    images = []
    
    for i in range(num_images):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=pipeline.device).manual_seed(seed + i)
        
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        if return_numpy:
            images.append(pil_to_numpy(image))
        else:
            images.append(image)
    
    return images
