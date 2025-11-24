"""
General image conversion utilities.

This module provides format conversion between:
- PIL Images
- Numpy arrays
- PyTorch tensors

These utilities are general-purpose and can be used across different tasks
(classification, diffusion, evaluation, etc.)
"""

import numpy as np
from PIL import Image
import torch


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image
    
    Returns:
        Numpy array of shape (H, W, C) or (H, W), dtype=uint8
    """
    return np.array(image)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array of shape (H, W, C) or (H, W), dtype=uint8
    
    Returns:
        PIL Image
    """
    return Image.fromarray(array)


def tensor_to_numpy(tensor: torch.Tensor, value_range: str = "[-1,1]") -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W)
        value_range: Input value range - "[-1,1]" or "[0,1]"
    
    Returns:
        Numpy array of shape (H, W, C) or (B, H, W, C) in range [0, 255], dtype=uint8
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        # (B, C, H, W) -> (B, H, W, C)
        img_np = tensor.permute(0, 2, 3, 1).cpu().numpy()
    else:
        # (C, H, W) -> (H, W, C)
        img_np = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Convert to [0, 255] based on input range
    if value_range == "[-1,1]":
        img_np = ((img_np + 1) * 127.5).astype(np.uint8)
    elif value_range == "[0,1]":
        img_np = (img_np * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown value_range: {value_range}. Use '[-1,1]' or '[0,1]'")
    
    return img_np


def numpy_to_tensor(array: np.ndarray, value_range: str = "[-1,1]") -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        array: Numpy array of shape (H, W, C) or (B, H, W, C), dtype=uint8 in range [0, 255]
        value_range: Output value range - "[-1,1]" or "[0,1]"
    
    Returns:
        Tensor of shape (C, H, W) or (B, C, H, W) in specified range
    """
    # Convert to float and normalize
    img_float = array.astype(np.float32)
    
    if value_range == "[-1,1]":
        img_float = (img_float / 127.5) - 1.0
    elif value_range == "[0,1]":
        img_float = img_float / 255.0
    else:
        raise ValueError(f"Unknown value_range: {value_range}. Use '[-1,1]' or '[0,1]'")
    
    # Convert to tensor
    tensor = torch.from_numpy(img_float)
    
    # Handle batch dimension
    if tensor.dim() == 4:
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
    
    return tensor


def pil_to_tensor(image: Image.Image, value_range: str = "[-1,1]") -> torch.Tensor:
    """
    Convert PIL Image to PyTorch tensor.
    
    Args:
        image: PIL Image
        value_range: Output value range - "[-1,1]" or "[0,1]"
    
    Returns:
        Tensor of shape (C, H, W) in specified range
    """
    array = pil_to_numpy(image)
    return numpy_to_tensor(array, value_range)


def tensor_to_pil(tensor: torch.Tensor, value_range: str = "[-1,1]") -> Image.Image:
    """
    Convert PyTorch tensor to PIL Image.
    
    Args:
        tensor: Tensor of shape (C, H, W)
        value_range: Input value range - "[-1,1]" or "[0,1]"
    
    Returns:
        PIL Image
    """
    array = tensor_to_numpy(tensor, value_range)
    return numpy_to_pil(array)
