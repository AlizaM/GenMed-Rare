"""
Evaluate and compare diffusion model checkpoints.

This script:
1. Loads and validates config using dataclass-based config manager
2. Generates images from multiple checkpoints
3. Computes SSIM/correlation with training images (novelty metric)
4. Computes CLIP scores (text-image alignment)
5. Visualizes most similar image pairs
6. Produces comparison summary

Usage:
    # Run with config file (recommended)
    python scripts/evaluate_checkpoints.py --config configs/config_eval_fibrosis.yaml
    
    # Override specific settings
    python scripts/evaluate_checkpoints.py --config configs/config_eval_fibrosis.yaml --num-images 200
"""

import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from src.config import load_diffusion_eval_config, DiffusionEvaluationConfig
from src.data.diffusion_dataset import ChestXrayDiffusionDataset
from src.utils.diffusion_utils import load_pipeline, generate_images
from src.utils.image_utils import tensor_to_numpy, pil_to_numpy
from src.eval.metrics import (
    compute_novelty_metrics,
    load_clip_model,
    compute_clip_scores,
    compute_clip_metrics
)
from src.utils.visualization import (
    visualize_similar_pairs,
    plot_score_distributions,
    plot_checkpoint_comparison
)

# Import shared image saving utilities
from scripts.generate_xrays import save_images_with_seeds


def fail_fast_message(error_type: str, message: str):
    """
    Log a clear error message and exit.
    
    Args:
        error_type: Type of error (e.g., "CHECKPOINT ERROR")
        message: Detailed error message
    """
    logger.error("=" * 80)
    logger.error(f"ERROR: {error_type}")
    logger.error("=" * 80)
    logger.error(message)
    logger.error("=" * 80)
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate diffusion checkpoints")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_eval_fibrosis.yaml",
        help="Path to evaluation config file"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Override number of images to generate per checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)"
    )
    return parser.parse_args()


def load_training_images(config: DiffusionEvaluationConfig) -> List[np.ndarray]:
    """
    Load all training images as numpy arrays.
    
    Args:
        config: Diffusion evaluation configuration
    
    Returns:
        List of training images as numpy arrays
    
    Raises:
        SystemExit: If dataset loading fails
    """
    logger.info("=" * 80)
    logger.info("Loading training images...")
    logger.info("=" * 80)
    
    csv_path = config.data.data_dir / config.data.csv_file
    
    try:
        dataset = ChestXrayDiffusionDataset(
            csv_file=str(csv_path),
            data_dir=str(config.data.data_dir),
            image_size=config.data.image_size,
            prompt_template=config.data.prompt_template,
            center_crop=config.data.center_crop,
            random_flip=config.data.random_flip,
            label_subdir=config.data.label_subdir,
        )
    except FileNotFoundError as e:
        fail_fast_message(
            "DATA LOADING ERROR",
            f"Failed to load dataset\n"
            f"  CSV file: {csv_path}\n"
            f"  Error: {e}\n\n"
            f"  This script requires the dataset to exist.\n"
            f"  Please verify:\n"
            f"    1. Data directory exists: {config.data.data_dir}\n"
            f"    2. CSV file exists: {csv_path}\n"
            f"    3. Images are in the data directory"
        )
    except Exception as e:
        fail_fast_message(
            "DATA LOADING ERROR",
            f"Unexpected error loading dataset: {e}"
        )
    
    if len(dataset) == 0:
        fail_fast_message(
            "DATA LOADING ERROR",
            f"Dataset is empty!\n"
            f"  CSV file: {csv_path}\n"
            f"  Please verify the CSV file contains valid entries."
        )
    
    logger.info(f"Dataset loaded: {len(dataset)} images")
    
    training_images = []
    for i in tqdm(range(len(dataset)), desc="Converting to numpy"):
        sample = dataset[i]
        # Convert from tensor [-1, 1] to numpy [0, 255]
        img_np = tensor_to_numpy(sample['pixel_values'])
        training_images.append(img_np)
    
    logger.info(f"✓ Loaded {len(training_images)} training images")
    
    return training_images


def save_checkpoint_results(
    results: dict,
    output_dir: Path,
    checkpoint_name: str,
    label: str
):
    """Save detailed results for a checkpoint-label combination."""
    import yaml
    
    output_file = output_dir / f"{checkpoint_name}_{label}_results.yaml"
    
    # Remove large lists for summary
    summary = {k: v for k, v in results.items() if k not in ['nn_indices', 'nn_scores', 'clip_scores']}
    
    with open(output_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    logger.info(f"✓ Saved detailed results to {output_file}")


def create_summary_table(all_results: dict, output_dir: Path, label: str):
    """Create summary comparison table across all checkpoints."""
    logger.info("=" * 80)
    logger.info("Creating summary table...")
    logger.info("=" * 80)
    
    rows = []
    for checkpoint_name, checkpoint_data in all_results.items():
        if label not in checkpoint_data:
            continue
        
        metrics = checkpoint_data[label]
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
    
    if not rows:
        logger.warning(f"No results to summarize for label: {label}")
        return None
    
    df = pd.DataFrame(rows)
    
    # Sort by P99 SSIM (lower is better for novelty) and Mean CLIP (higher is better)
    df['Novelty Rank'] = df['P99 SSIM'].rank(ascending=True)
    df['CLIP Rank'] = df['Mean CLIP'].rank(ascending=False)
    df['Combined Score'] = (df['Novelty Rank'] + df['CLIP Rank']) / 2
    
    # Save to CSV
    csv_path = output_dir / 'checkpoint_comparison.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved summary table to {csv_path}")
    
    # Print summary
    logger.info("=" * 80)
    logger.info("CHECKPOINT EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\n{df.to_string(index=False)}")
    
    logger.info("=" * 80)
    logger.info(f"BEST CHECKPOINT FOR {label}")
    logger.info("=" * 80)
    
    best_novelty = df.loc[df['P99 SSIM'].idxmin()]
    best_clip = df.loc[df['Mean CLIP'].idxmax()]
    best_combined = df.loc[df['Combined Score'].idxmin()]
    
    logger.info(f"\nBest Novelty (lowest P99 SSIM):")
    logger.info(f"  Checkpoint: {best_novelty['Checkpoint']}")
    logger.info(f"  P99 SSIM: {best_novelty['P99 SSIM']:.4f}")
    logger.info(f"  Mean CLIP: {best_novelty['Mean CLIP']:.2f}")
    
    logger.info(f"\nBest CLIP Alignment (highest Mean CLIP):")
    logger.info(f"  Checkpoint: {best_clip['Checkpoint']}")
    logger.info(f"  Mean CLIP: {best_clip['Mean CLIP']:.2f}")
    logger.info(f"  P99 SSIM: {best_clip['P99 SSIM']:.4f}")
    
    logger.info(f"\nBest Combined Score:")
    logger.info(f"  Checkpoint: {best_combined['Checkpoint']}")
    logger.info(f"  Combined Score: {best_combined['Combined Score']:.2f}")
    logger.info(f"  P99 SSIM: {best_combined['P99 SSIM']:.4f}")
    logger.info(f"  Mean CLIP: {best_combined['Mean CLIP']:.2f}")
    
    logger.info("=" * 80)
    
    return df


def main():
    args = parse_args()
    
    # Load and validate config using config manager
    logger.info(f"Loading config from: {args.config}")
    
    try:
        config = load_diffusion_eval_config(args.config)
    except FileNotFoundError as e:
        fail_fast_message("CONFIG ERROR", str(e))
    except ValueError as e:
        fail_fast_message("CONFIG VALIDATION ERROR", str(e))
    except Exception as e:
        fail_fast_message("CONFIG ERROR", f"Unexpected error loading config: {e}")
    
    # Apply command-line overrides
    if args.num_images is not None:
        config.generation.num_images = args.num_images
    if args.output_dir is not None:
        config.evaluation.output_dir = Path(args.output_dir)
    if args.device is not None:
        config.hardware.device = args.device
    
    # Validate all paths exist (fail-fast)
    logger.info("=" * 80)
    logger.info("Validating paths...")
    logger.info("=" * 80)
    
    try:
        config.validate_paths()
    except FileNotFoundError as e:
        fail_fast_message("PATH VALIDATION ERROR", str(e))
    
    # Create output directory
    config.create_dirs()
    
    # Set seed
    torch.manual_seed(config.metrics.seed)
    np.random.seed(config.metrics.seed)
    logger.info(f"✓ Random seed set to: {config.metrics.seed}")
    logger.info(f"✓ Using device: {config.hardware.device}")
    
    # Load training images
    training_images = load_training_images(config)
    
    # Load CLIP model
    logger.info("=" * 80)
    logger.info("Loading CLIP model...")
    logger.info("=" * 80)
    clip_model, clip_processor = load_clip_model(config.hardware.device)
    logger.info("✓ CLIP model loaded")
    
    # Store all results
    all_results = {}
    
    # Create prompt
    prompt = config.data.prompt_template.format(labels=config.evaluation.label)
    
    logger.info("=" * 80)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Label: {config.evaluation.label}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Checkpoints to evaluate: {len(config.evaluation.checkpoint_names)}")
    for name in config.evaluation.checkpoint_names:
        logger.info(f"  - {name}")
    logger.info(f"Images per checkpoint: {config.generation.num_images}")
    logger.info(f"Inference steps: {config.generation.num_inference_steps}")
    logger.info(f"Guidance scale: {config.generation.guidance_scale}")
    logger.info("=" * 80)
    
    # Evaluate each checkpoint
    for checkpoint_name in config.evaluation.checkpoint_names:
        logger.info("=" * 80)
        logger.info(f"Evaluating checkpoint: {checkpoint_name}")
        logger.info("=" * 80)
        
        checkpoint_path = config.evaluation.checkpoint_dir / checkpoint_name
        
        # Load pipeline
        try:
            pipeline = load_pipeline(
                checkpoint_path=str(checkpoint_path),
                pretrained_model=config.model.pretrained_model,
                enable_attention_slicing=config.hardware.enable_attention_slicing,
                enable_vae_slicing=config.hardware.enable_vae_slicing,
                device=config.hardware.device
            )
        except FileNotFoundError as e:
            fail_fast_message("CHECKPOINT LOADING ERROR", str(e))
        except Exception as e:
            fail_fast_message("CHECKPOINT LOADING ERROR", f"Failed to load pipeline: {e}")
        
        # Generate images
        logger.info(f"Generating {config.generation.num_images} images...")
        generated_images_pil = generate_images(
            pipeline=pipeline,
            prompt=prompt,
            num_images=config.generation.num_images,
            num_inference_steps=config.generation.num_inference_steps,
            guidance_scale=config.generation.guidance_scale,
            lora_scale=config.generation.lora_scale,
            negative_prompt=config.generation.negative_prompt,
            seed=config.metrics.seed,
            return_numpy=False  # Get PIL Images first
        )
        logger.info(f"✓ Generated {len(generated_images_pil)} images")
        
        # Save generated images using shared utility
        images_dir = config.evaluation.output_dir / f"{checkpoint_name}_{config.evaluation.label}_images"
        save_images_with_seeds(generated_images_pil, images_dir, config.metrics.seed)
        logger.info(f"✓ Saved images to {images_dir}")
        
        # Convert to numpy for metrics
        generated_images = [pil_to_numpy(img) for img in generated_images_pil]
        
        # Compute novelty metrics
        logger.info("Computing novelty metrics...")
        novelty_metrics = compute_novelty_metrics(
            generated_images,
            training_images,
            metric=config.metrics.novelty_metric,
            show_progress=True
        )
        logger.info(f"✓ Novelty metrics computed")
        logger.info(f"  Max SSIM: {novelty_metrics['max_similarity']:.4f}")
        logger.info(f"  P99 SSIM: {novelty_metrics['p99_similarity']:.4f}")
        logger.info(f"  Mean SSIM: {novelty_metrics['mean_similarity']:.4f}")
        
        # Compute CLIP scores
        logger.info("Computing CLIP scores...")
        prompts = [prompt] * len(generated_images)
        clip_scores = compute_clip_scores(
            generated_images,
            prompts,
            clip_model,
            clip_processor,
            config.hardware.device,
            show_progress=True
        )
        clip_metrics = compute_clip_metrics(clip_scores)
        logger.info(f"✓ CLIP scores computed")
        logger.info(f"  Mean CLIP: {clip_metrics['mean_score']:.2f}")
        logger.info(f"  Median CLIP: {clip_metrics['median_score']:.2f}")
        
        # Visualize most similar pairs
        logger.info("Creating visualizations...")
        viz_path = config.evaluation.output_dir / f"{checkpoint_name}_{config.evaluation.label}_similar_pairs.png"
        visualize_similar_pairs(
            generated_images,
            training_images,
            novelty_metrics['nn_indices'],
            novelty_metrics['nn_scores'],
            viz_path,
            checkpoint_name,
            config.evaluation.label,
            top_k=config.metrics.visualize_top_k
        )
        
        # Store results
        all_results[checkpoint_name] = {
            config.evaluation.label: {
                'novelty': novelty_metrics,
                'clip': clip_metrics,
            }
        }
        
        # Save detailed results
        save_checkpoint_results(
            all_results[checkpoint_name][config.evaluation.label],
            config.evaluation.output_dir,
            checkpoint_name,
            config.evaluation.label
        )
        
        # Clean up pipeline to free memory
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create summary comparison table
    summary_df = create_summary_table(all_results, config.evaluation.output_dir, config.evaluation.label)
    
    # Create comparison plots
    if summary_df is not None and len(summary_df) > 0:
        logger.info("Creating comparison plots...")
        plot_score_distributions(all_results, config.evaluation.output_dir, config.evaluation.label)
        plot_checkpoint_comparison(summary_df, config.evaluation.output_dir, config.evaluation.label)
    
    logger.info("=" * 80)
    logger.info("✓ EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {config.evaluation.output_dir}")
    logger.info(f"\nKey files:")
    logger.info(f"  - checkpoint_comparison.csv")
    logger.info(f"  - {config.evaluation.label}_score_distributions.png")
    logger.info(f"  - {config.evaluation.label}_checkpoint_comparison.png")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
