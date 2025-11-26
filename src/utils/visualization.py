"""
Visualization utilities for evaluation results.

This module provides functions for:
- Visualizing generated vs training image pairs
- Creating summary comparison plots
"""

from typing import List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def visualize_similar_pairs(
    generated_images: List[np.ndarray],
    training_images: List[np.ndarray],
    nn_indices: List[int],
    nn_scores: List[float],
    output_path: Path,
    checkpoint_name: str,
    label: str,
    top_k: int = 10
):
    """
    Visualize most similar generated-training image pairs.
    
    Args:
        generated_images: List of generated images
        training_images: List of training images
        nn_indices: Nearest neighbor indices for each generated image
        nn_scores: Similarity scores for each nearest neighbor
        output_path: Path to save visualization
        checkpoint_name: Name of checkpoint being evaluated
        label: Label being evaluated
        top_k: Number of pairs to visualize
    """
    # Get top k most similar pairs (highest scores)
    sorted_indices = np.argsort(nn_scores)[::-1][:top_k]
    
    fig, axes = plt.subplots(top_k, 2, figsize=(10, 5 * top_k))
    
    # Handle case where top_k = 1
    if top_k == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(sorted_indices):
        gen_img = generated_images[idx]
        train_img = training_images[nn_indices[idx]]
        score = nn_scores[idx]
        
        # Plot generated image
        axes[i, 0].imshow(gen_img)
        axes[i, 0].set_title(f"Generated #{idx}")
        axes[i, 0].axis('off')
        
        # Plot most similar training image
        axes[i, 1].imshow(train_img)
        axes[i, 1].set_title(f"Training NN (SSIM: {score:.4f})")
        axes[i, 1].axis('off')
    
    plt.suptitle(f"{checkpoint_name} - {label}\nTop {top_k} Most Similar Pairs", fontsize=16)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization to {output_path}")


def plot_score_distributions(
    all_results: dict,
    output_dir: Path,
    label: str
):
    """
    Plot SSIM and CLIP score distributions across checkpoints.
    
    Args:
        all_results: Dictionary of results keyed by checkpoint name
        output_dir: Directory to save plots
        label: Label being evaluated
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SSIM distribution
    for checkpoint_name, checkpoint_data in all_results.items():
        if label not in checkpoint_data:
            continue
        
        nn_scores = checkpoint_data[label]['novelty']['nn_scores']
        axes[0].hist(nn_scores, bins=30, alpha=0.5, label=checkpoint_name)
    
    axes[0].set_xlabel('SSIM Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('SSIM Distribution (Novelty)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # CLIP distribution
    for checkpoint_name, checkpoint_data in all_results.items():
        if label not in checkpoint_data:
            continue
        
        clip_scores = checkpoint_data[label]['clip']['clip_scores']
        axes[1].hist(clip_scores, bins=30, alpha=0.5, label=checkpoint_name)
    
    axes[1].set_xlabel('CLIP Score')
    axes[1].set_ylabel('Count')
    axes[1].set_title('CLIP Score Distribution (Text-Image Alignment)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f'{label}_score_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved score distributions to {output_path}")


def plot_checkpoint_comparison(
    summary_df,
    output_dir: Path,
    label: str,
    novelty_metric: str = "correlation"
):
    """
    Create bar plots comparing checkpoints across metrics.

    Args:
        summary_df: DataFrame with checkpoint comparison results
        output_dir: Directory to save plots
        label: Label being evaluated
        novelty_metric: Metric used for novelty ("ssim" or "correlation")
    """
    label_df = summary_df[summary_df['Label'] == label].copy()

    if len(label_df) == 0:
        print(f"⚠ No results found for label: {label}")
        return

    # Determine metric suffix based on novelty metric
    metric_suffix = "SSIM" if novelty_metric == "ssim" else "Correlation"
    p99_col = f'P99 {metric_suffix}'
    mean_col = f'Mean {metric_suffix}'

    # Determine text-image alignment column (BioViL or CLIP)
    if 'Mean BioViL' in label_df.columns:
        alignment_col = 'Mean BioViL'
        alignment_label = 'BioViL Score'
    elif 'Mean CLIP' in label_df.columns:
        alignment_col = 'Mean CLIP'
        alignment_label = 'CLIP Score'
    else:
        alignment_col = None
        alignment_label = 'N/A'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # P99 novelty metric (lower is better - more novel)
    if p99_col in label_df.columns:
        axes[0, 0].bar(range(len(label_df)), label_df[p99_col])
        axes[0, 0].set_ylabel(p99_col)
        axes[0, 0].set_title('Novelty (lower = more novel)')
    else:
        axes[0, 0].text(0.5, 0.5, f'{p99_col} not available',
                        ha='center', va='center', transform=axes[0, 0].transAxes)
    axes[0, 0].set_xticks(range(len(label_df)))
    axes[0, 0].set_xticklabels(label_df['Checkpoint'], rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)

    # Mean novelty metric
    if mean_col in label_df.columns:
        axes[0, 1].bar(range(len(label_df)), label_df[mean_col])
        axes[0, 1].set_ylabel(mean_col)
        axes[0, 1].set_title('Mean Similarity to Training')
    else:
        axes[0, 1].text(0.5, 0.5, f'{mean_col} not available',
                        ha='center', va='center', transform=axes[0, 1].transAxes)
    axes[0, 1].set_xticks(range(len(label_df)))
    axes[0, 1].set_xticklabels(label_df['Checkpoint'], rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)

    # Text-image alignment (BioViL or CLIP, higher is better)
    if alignment_col and alignment_col in label_df.columns:
        axes[1, 0].bar(range(len(label_df)), label_df[alignment_col])
        axes[1, 0].set_ylabel(alignment_label)
        axes[1, 0].set_title('Text-Image Alignment (higher = better)')
    else:
        axes[1, 0].text(0.5, 0.5, f'{alignment_label} not available',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_xticks(range(len(label_df)))
    axes[1, 0].set_xticklabels(label_df['Checkpoint'], rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)

    # Combined score (lower is better)
    if 'Combined Score' in label_df.columns:
        axes[1, 1].bar(range(len(label_df)), label_df['Combined Score'])
        axes[1, 1].set_ylabel('Combined Score')
        axes[1, 1].set_title('Combined Ranking (lower = better)')
    else:
        axes[1, 1].text(0.5, 0.5, 'Combined Score not available',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_xticks(range(len(label_df)))
    axes[1, 1].set_xticklabels(label_df['Checkpoint'], rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Checkpoint Comparison - {label}', fontsize=16)
    plt.tight_layout()

    output_path = output_dir / f'{label}_checkpoint_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved checkpoint comparison to {output_path}")
