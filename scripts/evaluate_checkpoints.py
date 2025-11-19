"""
Evaluate and compare diffusion model checkpoints.

This script:
1. Generates images from multiple checkpoints for specified labels
2. Computes SSIM/correlation with training images (novelty metric)
3. Computes CLIP scores (text-image alignment)
4. Visualizes most similar image pairs
5. Produces comparison summary

Usage:
    python scripts/evaluate_checkpoints.py \
        --checkpoints checkpoint-1000 checkpoint-2000 checkpoint-3000 \
        --labels "Fibrosis" "Pneumonia" \
        --num-images 100
"""

import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

# Diffusion imports
from transformers import CLIPProcessor, CLIPModel

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.data.diffusion_dataset import ChestXrayDiffusionDataset, collate_fn

# Import reusable functions from generate_xrays
from generate_xrays import setup_pipeline as load_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate diffusion checkpoints")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_diffusion.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint names (e.g., checkpoint-1000 checkpoint-2000)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["Fibrosis", "Pneumonia"],
        help="Labels to generate images for"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to generate per label per checkpoint"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/checkpoint_evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_images_as_numpy(
    pipeline,
    prompt: str,
    num_images: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
    seed: int,
    device: str
) -> List[np.ndarray]:
    """
    Generate images from prompt and convert to numpy arrays.
    
    Note: This wraps the batch generation from generate_xrays.generate_images
    but converts to individual numpy arrays for evaluation.
    """
    images_np = []
    
    for i in range(num_images):
        generator = torch.Generator(device=device).manual_seed(seed + i)
        
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        # Convert PIL to numpy array
        images_np.append(np.array(image))
    
    return images_np


def load_training_images(config: dict) -> List[np.ndarray]:
    """Load all training images as numpy arrays."""
    print("Loading training images...")
    
    dataset = ChestXrayDiffusionDataset(
        csv_file=str(Path(config['data']['data_dir']) / config['data']['csv_file']),
        data_dir=config['data']['data_dir'],
        image_size=config['data']['image_size'],
        prompt_template=config['data']['prompt_template'],
        center_crop=config['data']['center_crop'],
        random_flip=config['data']['random_flip'],
    )
    
    training_images = []
    for i in tqdm(range(len(dataset)), desc="Loading training images"):
        sample = dataset[i]
        # Convert from tensor [-1, 1] to numpy [0, 255]
        img_tensor = sample['pixel_values']
        img_np = ((img_tensor.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
        training_images.append(img_np)
    
    return training_images


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two RGB images."""
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
    """Compute pixel-wise correlation between two images."""
    # Flatten and compute correlation
    corr, _ = pearsonr(img1.flatten(), img2.flatten())
    return corr


def find_nearest_neighbor(
    generated_img: np.ndarray,
    training_images: List[np.ndarray],
    metric: str = "ssim"
) -> Tuple[int, float, np.ndarray]:
    """
    Find nearest neighbor in training set.
    
    Returns:
        (index, similarity_score, training_image)
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
    metric: str = "ssim"
) -> Dict[str, float]:
    """
    Compute novelty metrics for generated images.
    
    Returns dict with:
        - max_similarity: Maximum similarity to any training image
        - p99_similarity: 99th percentile similarity
        - mean_similarity: Mean similarity
        - median_similarity: Median similarity
        - nn_indices: List of nearest neighbor indices
        - nn_scores: List of similarity scores
    """
    print(f"Computing novelty metrics using {metric}...")
    
    nn_scores = []
    nn_indices = []
    
    for gen_img in tqdm(generated_images, desc="Finding nearest neighbors"):
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


def load_clip_model(device: str):
    """Load CLIP model for text-image alignment."""
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
    device: str
) -> List[float]:
    """Compute CLIP scores between images and their prompts."""
    print("Computing CLIP scores...")
    
    scores = []
    
    for img, prompt in tqdm(zip(images, prompts), total=len(images), desc="CLIP scoring"):
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


def visualize_similar_pairs(
    generated_images: List[np.ndarray],
    training_images: List[np.ndarray],
    nn_indices: List[int],
    nn_scores: List[float],
    output_dir: Path,
    checkpoint_name: str,
    label: str,
    top_k: int = 10
):
    """Visualize most similar generated-training image pairs."""
    print(f"Visualizing top {top_k} most similar pairs...")
    
    # Get top k most similar pairs (highest scores)
    sorted_indices = np.argsort(nn_scores)[::-1][:top_k]
    
    fig, axes = plt.subplots(top_k, 2, figsize=(10, 5 * top_k))
    
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
    
    output_path = output_dir / f"{checkpoint_name}_{label}_similar_pairs.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path}")


def save_checkpoint_results(
    results: Dict,
    output_dir: Path,
    checkpoint_name: str,
    label: str
):
    """Save detailed results for a checkpoint-label combination."""
    output_file = output_dir / f"{checkpoint_name}_{label}_results.yaml"
    
    # Remove large lists for summary
    summary = {k: v for k, v in results.items() if k not in ['nn_indices', 'nn_scores', 'clip_scores']}
    
    with open(output_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    print(f"Saved results to {output_file}")


def create_summary_table(all_results: Dict, output_dir: Path):
    """Create summary comparison table across all checkpoints."""
    print("Creating summary table...")
    
    rows = []
    for checkpoint_name, checkpoint_data in all_results.items():
        for label, metrics in checkpoint_data.items():
            row = {
                'Checkpoint': checkpoint_name,
                'Label': label,
                'Max SSIM': metrics['novelty']['max_similarity'],
                'P99 SSIM': metrics['novelty']['p99_similarity'],
                'P95 SSIM': metrics['novelty']['p95_similarity'],
                'Mean SSIM': metrics['novelty']['mean_similarity'],
                'Mean CLIP': metrics['clip']['mean_score'],
                'Median CLIP': metrics['clip']['median_score'],
                'Min CLIP': metrics['clip']['min_score'],
                'Max CLIP': metrics['clip']['max_score'],
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by P99 SSIM (lower is better for novelty) and Mean CLIP (higher is better)
    df['Novelty Rank'] = df.groupby('Label')['P99 SSIM'].rank(ascending=True)
    df['CLIP Rank'] = df.groupby('Label')['Mean CLIP'].rank(ascending=False)
    df['Combined Score'] = (df['Novelty Rank'] + df['CLIP Rank']) / 2
    
    # Save to CSV
    csv_path = output_dir / 'checkpoint_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved summary table to {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("CHECKPOINT EVALUATION SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("\n" + "="*80)
    print("BEST CHECKPOINT PER LABEL")
    print("="*80)
    
    for label in df['Label'].unique():
        label_df = df[df['Label'] == label]
        best_novelty = label_df.loc[label_df['P99 SSIM'].idxmin()]
        best_clip = label_df.loc[label_df['Mean CLIP'].idxmax()]
        best_combined = label_df.loc[label_df['Combined Score'].idxmin()]
        
        print(f"\n{label}:")
        print(f"  Best Novelty (lowest P99 SSIM): {best_novelty['Checkpoint']} (P99={best_novelty['P99 SSIM']:.4f})")
        print(f"  Best CLIP Alignment: {best_clip['Checkpoint']} (Mean CLIP={best_clip['Mean CLIP']:.2f})")
        print(f"  Best Combined: {best_combined['Checkpoint']} (Score={best_combined['Combined Score']:.2f})")
    
    print("\n" + "="*80)
    
    return df


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load training images (once)
    training_images = load_training_images(config)
    
    # Load CLIP model (once)
    clip_model, clip_processor = load_clip_model(args.device)
    
    # Store all results
    all_results = {}
    
    # Evaluate each checkpoint
    for checkpoint_name in args.checkpoints:
        print("\n" + "="*80)
        print(f"Evaluating checkpoint: {checkpoint_name}")
        print("="*80)
        
        checkpoint_path = Path(config['training']['checkpoint_dir']) / checkpoint_name
        
        # Load pipeline (reuse from generate_xrays.py)
        pipeline = load_pipeline(str(checkpoint_path), config)
        
        all_results[checkpoint_name] = {}
        
        # Evaluate for each label
        for label in args.labels:
            print(f"\nProcessing label: {label}")
            
            # Create prompt
            prompt = config['data']['prompt_template'].format(labels=label)
            print(f"Prompt: {prompt}")
            
            # Generate images
            print(f"Generating {args.num_images} images...")
            negative_prompt = "blurry, low quality, distorted, artifacts"
            generated_images = generate_images_as_numpy(
                pipeline=pipeline,
                prompt=prompt,
                num_images=args.num_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=negative_prompt,
                seed=args.seed,
                device=args.device
            )
            
            # Compute novelty metrics (SSIM)
            novelty_metrics = compute_novelty_metrics(
                generated_images,
                training_images,
                metric="ssim"
            )
            
            # Compute CLIP scores
            prompts = [prompt] * len(generated_images)
            clip_scores = compute_clip_scores(
                generated_images,
                prompts,
                clip_model,
                clip_processor,
                args.device
            )
            
            clip_metrics = {
                'mean_score': float(np.mean(clip_scores)),
                'median_score': float(np.median(clip_scores)),
                'max_score': float(np.max(clip_scores)),
                'min_score': float(np.min(clip_scores)),
                'std_score': float(np.std(clip_scores)),
                'clip_scores': clip_scores,
            }
            
            # Visualize most similar pairs
            visualize_similar_pairs(
                generated_images,
                training_images,
                novelty_metrics['nn_indices'],
                novelty_metrics['nn_scores'],
                output_dir,
                checkpoint_name,
                label,
                top_k=10
            )
            
            # Store results
            all_results[checkpoint_name][label] = {
                'novelty': novelty_metrics,
                'clip': clip_metrics,
            }
            
            # Save detailed results
            save_checkpoint_results(
                all_results[checkpoint_name][label],
                output_dir,
                checkpoint_name,
                label
            )
        
        # Clean up pipeline to free memory
        del pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create summary comparison table
    summary_df = create_summary_table(all_results, output_dir)
    
    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
