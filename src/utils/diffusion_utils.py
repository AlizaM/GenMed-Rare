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
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel, LoraConfig, get_peft_model
from safetensors.torch import load_file

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
    
    Handles two checkpoint formats:
    1. LoRA adapter files: adapter_config.json + adapter_model.safetensors
    2. Accelerate checkpoints: model.safetensors + optimizer.bin (training format)
    
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
        ValueError: If checkpoint format is not recognized
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}\n"
            f"Please verify the checkpoint path is correct."
        )

    # Use rglob to find both adapter_config.json and adapter_model.safetensors
    adapter_configs = list(checkpoint_path.rglob("adapter_config.json"))
    adapter_models = list(checkpoint_path.rglob("adapter_model.safetensors"))
    adapter_dirs = [cfg.parent for cfg in adapter_configs if any(m.parent == cfg.parent for m in adapter_models)]
    if adapter_dirs:
        print(f"Found LoRA adapter files in: {adapter_dirs[0]}")
        checkpoint_path = adapter_dirs[0]

    print(f"Loading base model: {pretrained_model}")

    # Determine device first
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set dtype based on device
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
    )

    # Replace scheduler with DDIM for efficient inference
    from diffusers import DDIMScheduler
    pipeline.scheduler = DDIMScheduler.from_pretrained(
        pretrained_model,
        subfolder="scheduler"
    )
    print(f"✓ Using DDIMScheduler for efficient inference (compatible with DDPM training)")

    # Detect checkpoint format
    has_adapter_config = (checkpoint_path / "adapter_config.json").exists()
    has_model_safetensors = (checkpoint_path / "model.safetensors").exists()

    if has_adapter_config:
        # Format 1: LoRA adapter files (evaluation-friendly format)
        print(f"Loading LoRA adapter from: {checkpoint_path}")
        pipeline.unet = PeftModel.from_pretrained(
            pipeline.unet,
            checkpoint_path,
            is_trainable=False
        )
    elif has_model_safetensors:
        # Format 2: Accelerate checkpoint (old training format - PEFT model with LoRA)
        # The model.safetensors contains UNet state dict with PEFT structure:
        # - base_model.model.X.base_layer.weight (base weights)
        # - base_model.model.X.lora_A/B.default.weight (LoRA weights)
        print(f"⚠️  Detected old Accelerate checkpoint format at: {checkpoint_path}")
        print(f"Loading PEFT UNet state from model.safetensors...")
        print(f"Note: Future checkpoints will use LoRA adapter format (adapter_*.safetensors)")
        
        # Load the state dict from safetensors
        state_dict = load_file(str(checkpoint_path / "model.safetensors"))
        
        # Check structure
        sample_keys = list(state_dict.keys())[:5]
        has_peft_structure = any('base_model.model.' in key for key in sample_keys)
        
        if has_peft_structure:
            print(f"✓ Detected PEFT structure with base_model.model prefix")
            print(f"Loading checkpoint with PEFT wrapper (not merging)...")
            
            # Convert checkpoint to target dtype
            print(f"Converting checkpoint from float32 to {dtype}...")
            state_dict_converted = {}
            for k, v in state_dict.items():
                if v.dtype == torch.float32 and dtype == torch.float16:
                    state_dict_converted[k] = v.half()
                elif v.dtype == torch.float16 and dtype == torch.float32:
                    state_dict_converted[k] = v.float()
                else:
                    state_dict_converted[k] = v
            
            # Wrap the UNet with PEFT and load the full state dict
            # This preserves the exact same runtime behavior as during training
            from peft import LoraConfig, get_peft_model
            
            # Create LoRA config matching training
            lora_config = LoraConfig(
                r=64,
                lora_alpha=64,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                lora_dropout=0.0,
                bias="none",
            )
            
            # Wrap UNet with PEFT BEFORE loading state dict
            pipeline.unet = get_peft_model(pipeline.unet, lora_config)
            print(f"✓ Wrapped UNet with PEFT")
            
            # Now load the full state dict (includes both base and LoRA weights)
            # The state dict keys match the PEFT model structure
            missing_keys, unexpected_keys = pipeline.unet.load_state_dict(state_dict_converted, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys: {len(missing_keys)}")
                if len(missing_keys) <= 10:
                    for k in missing_keys[:10]:
                        print(f"    - {k}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 10:
                    for k in unexpected_keys[:10]:
                        print(f"    - {k}")
            
            print(f"✓ Loaded PEFT UNet with LoRA weights ({len(state_dict_converted)} keys)")
        else:
            # Regular state dict without PEFT structure
            missing_keys, unexpected_keys = pipeline.unet.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {len(unexpected_keys)} keys")
            
            print(f"✓ Loaded UNet state dict ({len(state_dict)} keys)")
    else:
        # Check what files exist for better error message
        files = list(checkpoint_path.iterdir())
        file_list = "\n".join([f"  - {f.name}" for f in files])
        
        raise ValueError(
            f"Unrecognized checkpoint format at: {checkpoint_path}\n"
            f"Expected either:\n"
            f"  - LoRA adapter: adapter_config.json + adapter_model.safetensors\n"
            f"  - Accelerate checkpoint: model.safetensors + optimizer.bin\n\n"
            f"Found files:\n{file_list}\n\n"
            f"Please ensure checkpoints are saved in one of these formats."
        )

    # Move to device AFTER loading checkpoint
    # This prevents CUDA misaligned address errors by ensuring proper memory layout
    pipeline = pipeline.to(device)

    # Enable memory optimizations
    if enable_attention_slicing:
        pipeline.enable_attention_slicing()

    if enable_vae_slicing:
        pipeline.enable_vae_slicing()

    print(f"✓ Pipeline ready for inference on {device}")

    return pipeline


def generate_images(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    num_images: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    lora_scale: float = 1.0,
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
        lora_scale: LoRA adaptation strength (0.0=disabled, 1.0=full, >1.0=amplified)
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
            cross_attention_kwargs={"scale": lora_scale},
            generator=generator,
        ).images[0]
        
        if return_numpy:
            images.append(pil_to_numpy(image))
        else:
            images.append(image)
    
    return images
