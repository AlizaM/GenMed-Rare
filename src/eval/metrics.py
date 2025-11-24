"""
Evaluation metrics for diffusion model outputs.

This module provides:
- SSIM (Structural Similarity Index)
- Pixel correlation
- CLIP text-image alignment scores
- Novelty metrics (nearest neighbor analysis)
"""

from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from transformers import CLIPProcessor, CLIPModel


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM between two RGB images.
    
    Args:
        img1: First image as numpy array (H, W, C) or (H, W)
        img2: Second image as numpy array (H, W, C) or (H, W)
    
    Returns:
        SSIM score in range [-1, 1], where 1 = identical
    """
    # Convert to grayscale for SSIM
    if len(img1.shape) == 3:
        img1_gray = np.mean(img1, axis=2)
    else:
        img1_gray = img1
    
    if len(img2.shape) == 3:
        img2_gray = np.mean(img2, axis=2)
    else:
        img2_gray = img2
    
    # Normalize to [0, 1]
    img1_gray = img1_gray / 255.0
    img2_gray = img2_gray / 255.0
    
    return ssim(img1_gray, img2_gray, data_range=1.0)


def compute_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute pixel-wise Pearson correlation between two images.
    
    Args:
        img1: First image as numpy array
        img2: Second image as numpy array
    
    Returns:
        Correlation coefficient in range [-1, 1]
    """
    corr, _ = pearsonr(img1.flatten(), img2.flatten())
    return corr


def find_nearest_neighbor(
    generated_img: np.ndarray,
    training_images: List[np.ndarray],
    metric: str = "ssim"
) -> Tuple[int, float, np.ndarray]:
    """
    Find nearest neighbor in training set.
    
    Args:
        generated_img: Generated image as numpy array
        training_images: List of training images as numpy arrays
        metric: Similarity metric ("ssim" or "correlation")
    
    Returns:
        Tuple of (index, similarity_score, training_image)
    """
    best_idx = -1
    best_score = -np.inf
    
    for idx, train_img in enumerate(training_images):
        if metric == "ssim":
            score = compute_ssim(generated_img, train_img)
        elif metric == "correlation":
            score = compute_correlation(generated_img, train_img)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_idx = idx
    
    return best_idx, best_score, training_images[best_idx]


def compute_novelty_metrics(
    generated_images: List[np.ndarray],
    training_images: List[np.ndarray],
    metric: str = "ssim",
    show_progress: bool = True
) -> Dict[str, any]:
    """
    Compute novelty metrics for generated images.
    
    Lower similarity scores indicate higher novelty (less copying from training set).
    
    Args:
        generated_images: List of generated images as numpy arrays
        training_images: List of training images as numpy arrays
        metric: Similarity metric ("ssim" or "correlation")
        show_progress: Show progress bar
    
    Returns:
        Dictionary with:
            - max_similarity: Maximum similarity to any training image
            - p99_similarity: 99th percentile similarity
            - p95_similarity: 95th percentile similarity
            - mean_similarity: Mean similarity
            - median_similarity: Median similarity
            - min_similarity: Minimum similarity
            - nn_indices: List of nearest neighbor indices
            - nn_scores: List of similarity scores
    """
    nn_scores = []
    nn_indices = []
    
    iterator = tqdm(generated_images, desc=f"Computing {metric} novelty") if show_progress else generated_images
    
    for gen_img in iterator:
        nn_idx, nn_score, _ = find_nearest_neighbor(gen_img, training_images, metric)
        nn_scores.append(nn_score)
        nn_indices.append(nn_idx)
    
    nn_scores = np.array(nn_scores)
    
    return {
        'max_similarity': float(np.max(nn_scores)),
        'p99_similarity': float(np.percentile(nn_scores, 99)),
        'p95_similarity': float(np.percentile(nn_scores, 95)),
        'mean_similarity': float(np.mean(nn_scores)),
        'median_similarity': float(np.median(nn_scores)),
        'min_similarity': float(np.min(nn_scores)),
        'nn_indices': nn_indices,
        'nn_scores': nn_scores.tolist(),
    }


def load_clip_model(device: str = "cuda"):
    """
    Load CLIP model for text-image alignment.
    
    Args:
        device: Device to load model on
    
    Returns:
        Tuple of (model, processor)
    """
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    return model, processor


def compute_clip_scores(
    images: List[np.ndarray],
    prompts: List[str],
    clip_model,
    clip_processor,
    device: str,
    show_progress: bool = True
) -> List[float]:
    """
    Compute CLIP scores between images and their prompts.
    
    Higher scores indicate better text-image alignment.
    
    Args:
        images: List of images as numpy arrays
        prompts: List of text prompts (one per image)
        clip_model: CLIP model
        clip_processor: CLIP processor
        device: Device for computation
        show_progress: Show progress bar
    
    Returns:
        List of CLIP scores (one per image)
    """
    scores = []
    
    iterator = zip(images, prompts)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Computing CLIP scores")
    
    for img, prompt in iterator:
        # Convert numpy to PIL
        pil_img = Image.fromarray(img)
        
        # Process
        inputs = clip_processor(
            text=[prompt],
            images=[pil_img],
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Compute similarity
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image[0, 0].item()
        
        scores.append(score)
    
    return scores


def compute_clip_metrics(clip_scores: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for CLIP scores.
    
    Args:
        clip_scores: List of CLIP scores
    
    Returns:
        Dictionary with mean, median, max, min, std
    """
    clip_scores = np.array(clip_scores)
    
    return {
        'mean_score': float(np.mean(clip_scores)),
        'median_score': float(np.median(clip_scores)),
        'max_score': float(np.max(clip_scores)),
        'min_score': float(np.min(clip_scores)),
        'std_score': float(np.std(clip_scores)),
        'clip_scores': clip_scores.tolist(),
    }
