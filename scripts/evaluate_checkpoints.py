"""
Evaluate and compare diffusion model checkpoints.

This script:
1. For each checkpoint, checks if sufficient images exist (>= min_images)
2. Generates missing images if needed
3. Evaluates using DiffusionGenerationEvaluator with configurable metric preset
4. Compares checkpoints with summary tables and visualizations

Metric Presets:
- 'checkpoint' (default): novelty, pathology, biovil, diversity (fast)
- 'diversity': diversity-focused metrics
- 'full': all metrics including FMD and t-SNE (expensive)

Usage:
    # Run with config file (default preset: checkpoint)
    python scripts/evaluate_checkpoints.py --config configs/config_eval_fibrosis.yaml
    
    # Use full metrics
    python scripts/evaluate_checkpoints.py --config configs/config_eval_fibrosis.yaml --preset full
    
    # Override number of images
    python scripts/evaluate_checkpoints.py --config configs/config_eval_fibrosis.yaml --num-images 200
"""

import argparse
from pathlib import Path
from typing import List, Dict
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
from src.utils.diffusion_utils import load_pipeline, generate_images
from src.utils.image_utils import pil_to_numpy
from src.eval.diffusion_evaluator import DiffusionGenerationEvaluator
from src.utils.visualization import plot_checkpoint_comparison

# Import shared image saving utilities
from scripts.generate_xrays import save_images_with_seeds

# Metric presets (same as evaluate_diffusion_generation.py)
METRIC_PRESETS = {
    'checkpoint': {
        'description': 'Fast metrics for checkpoint comparison',
        'metrics': {
            'compute_novelty': True,
            'compute_pathology_confidence': True,
            'compute_biovil': True,
            'compute_diversity': True,
            'compute_pixel_variance': False,
            'compute_feature_dispersion': False,
            'compute_self_similarity': False,
            'compute_fmd': False,
            'compute_tsne': False,
        }
    },
    'diversity': {
        'description': 'Diversity-focused metrics for mode collapse detection',
        'metrics': {
            'compute_novelty': False,
            'compute_pathology_confidence': False,
            'compute_biovil': False,
            'compute_diversity': True,
            'compute_pixel_variance': True,
            'compute_feature_dispersion': True,
            'compute_self_similarity': True,
            'compute_fmd': False,
            'compute_tsne': False,
        }
    },
    'full': {
        'description': 'All metrics (expensive, for final evaluation)',
        'metrics': {
            'compute_novelty': True,
            'compute_pathology_confidence': True,
            'compute_biovil': True,
            'compute_diversity': True,
            'compute_pixel_variance': True,
            'compute_feature_dispersion': True,
            'compute_self_similarity': True,
            'compute_fmd': True,
            'compute_tsne': True,
        }
    },
}


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
    parser = argparse.ArgumentParser(
        description="Evaluate and compare diffusion checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default checkpoint preset (fast)
  python scripts/evaluate_checkpoints.py --config configs/config_eval_fibrosis.yaml

  # Full evaluation with all metrics
  python scripts/evaluate_checkpoints.py --config configs/config_eval_fibrosis.yaml --preset full

  # Override number of images
  python scripts/evaluate_checkpoints.py --config configs/config_eval_fibrosis.yaml --num-images 200

Presets:
  checkpoint: novelty, pathology, biovil, diversity (default, fast)
  diversity: diversity-focused metrics for mode collapse detection
  full: all 9 metrics including FMD and t-SNE (expensive)
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_eval_fibrosis.yaml",
        help="Path to evaluation config file"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(METRIC_PRESETS.keys()),
        default="checkpoint",
        help="Metric preset (default: checkpoint)"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Override number of images to generate per checkpoint"
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=100,
        help="Minimum images required to skip generation (default: 100)"
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


def count_images_in_dir(image_dir: Path) -> int:
    """Count valid image files in directory."""
    if not image_dir.exists():
        return 0
    
    image_extensions = {'.png', '.jpg', '.jpeg'}
    count = sum(1 for f in image_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in image_extensions)
    return count


def ensure_images_exist(
    checkpoint_name: str,
    checkpoint_path: Path,
    images_dir: Path,
    config: DiffusionEvaluationConfig,
    min_images: int
) -> Path:
    """
    Ensure at least min_images exist for checkpoint. Generate if needed.
    
    Args:
        checkpoint_name: Name of checkpoint
        checkpoint_path: Path to checkpoint directory
        images_dir: Directory where images should be stored
        config: Evaluation configuration
        min_images: Minimum number of images required
    
    Returns:
        Path to images directory
    """
    existing_count = count_images_in_dir(images_dir)
    
    if existing_count >= min_images:
        logger.info(f"✓ Found {existing_count} existing images (>= {min_images}), skipping generation")
        return images_dir
    
    logger.info(f"Found {existing_count} images, need {config.generation.num_images}")
    logger.info(f"Generating images for {checkpoint_name}...")
    
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
    
    # Create prompt
    prompt = config.data.prompt_template.format(labels=config.evaluation.label)
    
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
        return_numpy=False
    )
    logger.info(f"✓ Generated {len(generated_images_pil)} images")
    
    # Save images
    images_dir.mkdir(parents=True, exist_ok=True)
    save_images_with_seeds(generated_images_pil, images_dir, config.metrics.seed)
    logger.info(f"✓ Saved images to {images_dir}")
    
    # Clean up pipeline
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return images_dir


def load_training_images_dir(config: DiffusionEvaluationConfig) -> Path:
    """
    Get directory containing training images for the label.
    
    Args:
        config: Diffusion evaluation configuration
    
    Returns:
        Path to directory with real training images
    
    Raises:
        SystemExit: If directory not found
    """
    # Assume pure_class_folders structure: data/pure_class_folders/{label}/
    label_lower = config.evaluation.label.lower()
    real_dir = Path("data/pure_class_folders") / label_lower
    
    if not real_dir.exists():
        fail_fast_message(
            "DATA DIRECTORY ERROR",
            f"Real images directory not found: {real_dir}\n"
            f"Expected structure: data/pure_class_folders/{label_lower}/\n"
            f"Please ensure real training images are organized by label."
        )
    
    image_count = count_images_in_dir(real_dir)
    if image_count == 0:
        fail_fast_message(
            "DATA DIRECTORY ERROR",
            f"No images found in: {real_dir}\n"
            f"Please ensure the directory contains training images."
        )
    
    logger.info(f"✓ Found {image_count} real images in {real_dir}")
    return real_dir


def create_summary_table(all_results: Dict, output_dir: Path, label: str, preset: str, novelty_metric: str = "correlation") -> pd.DataFrame:
    """Create summary comparison table across all checkpoints."""
    logger.info("=" * 80)
    logger.info("Creating summary table...")
    logger.info("=" * 80)

    # Determine column name suffix based on metric
    metric_suffix = "SSIM" if novelty_metric == "ssim" else "Correlation"

    rows = []
    for checkpoint_name, results in all_results.items():
        row = {
            'Checkpoint': checkpoint_name,
            'Label': label,
        }

        # Add metrics based on what was computed
        if 'novelty' in results and results['novelty'] is not None:
            row[f'Max Novelty'] = results['novelty']['max_novelty']
            row[f'P99 Novelty'] = results['novelty']['p99_novelty']
            row[f'Mean Novelty'] = results['novelty']['mean_novelty']

        if 'pathology' in results and results['pathology'] is not None:
            row['Mean Pathology'] = results['pathology']['mean_confidence']
            row['Median Pathology'] = results['pathology']['median_confidence']
        
        if 'biovil' in results and results['biovil'] is not None:
            row['Mean BioViL'] = results['biovil'].get('mean_score')
            row['Median BioViL'] = results['biovil'].get('median_score')
        
        if 'diversity' in results and results['diversity'] is not None:
            row['Diversity'] = results['diversity']['overall_diversity']
            row['Diversity Std'] = results['diversity']['mean_std']

        if 'pixel_variance' in results and results['pixel_variance'] is not None:
            row['Pixel Variance'] = results['pixel_variance']['overall_variance']

        if 'feature_dispersion' in results and results['feature_dispersion'] is not None:
            row['Feature Dispersion'] = results['feature_dispersion']['log_det']

        if 'self_similarity' in results and results['self_similarity'] is not None:
            row['Self-Similarity'] = results['self_similarity']['mean_ssim']

        if 'fmd' in results and results['fmd'] is not None:
            row['FMD'] = results['fmd']['fmd']
        
        rows.append(row)
    
    if not rows:
        logger.warning(f"No results to summarize for label: {label}")
        return None
    
    df = pd.DataFrame(rows)

    # Compute ranks based on available metrics
    p99_col = f'P99 {metric_suffix}'
    if p99_col in df.columns:
        # Lower similarity = more novel (better)
        # For both SSIM and Correlation, lower means more different from training
        df['Novelty Rank'] = df[p99_col].rank(ascending=True)
    
    if 'Mean Pathology' in df.columns:
        df['Pathology Rank'] = df['Mean Pathology'].rank(ascending=False)  # Higher is better
    
    if 'Mean BioViL' in df.columns:
        df['BioViL Rank'] = df['Mean BioViL'].rank(ascending=False)  # Higher is better
    
    if 'Diversity' in df.columns:
        df['Diversity Rank'] = df['Diversity'].rank(ascending=False)  # Higher is better
    
    # Combined score (average of available ranks)
    rank_cols = [col for col in df.columns if col.endswith('Rank')]
    if rank_cols:
        df['Combined Score'] = df[rank_cols].mean(axis=1)
    
    # Save to CSV
    csv_path = output_dir / 'checkpoint_comparison.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved summary table to {csv_path}")
    
    # Print summary
    logger.info("=" * 80)
    logger.info(f"CHECKPOINT EVALUATION SUMMARY ({preset.upper()} PRESET)")
    logger.info("=" * 80)
    logger.info(f"\n{df.to_string(index=False)}")
    
    # Print best checkpoints
    logger.info("=" * 80)
    logger.info(f"BEST CHECKPOINT FOR {label}")
    logger.info("=" * 80)
    
    if p99_col in df.columns:
        best_novelty = df.loc[df[p99_col].idxmin()]
        logger.info(f"\nBest Novelty (lowest {p99_col}):")
        logger.info(f"  Checkpoint: {best_novelty['Checkpoint']}")
        logger.info(f"  {p99_col}: {best_novelty[p99_col]:.4f}")
    
    if 'Mean Pathology' in df.columns and df['Mean Pathology'].notna().any():
        best_pathology = df.loc[df['Mean Pathology'].idxmax()]
        logger.info(f"\nBest Pathology Confidence:")
        logger.info(f"  Checkpoint: {best_pathology['Checkpoint']}")
        logger.info(f"  Mean Pathology: {best_pathology['Mean Pathology']:.3f}")

    if 'Mean BioViL' in df.columns and df['Mean BioViL'].notna().any():
        best_biovil = df.loc[df['Mean BioViL'].idxmax()]
        logger.info(f"\nBest BioViL Alignment:")
        logger.info(f"  Checkpoint: {best_biovil['Checkpoint']}")
        logger.info(f"  Mean BioViL: {best_biovil['Mean BioViL']:.3f}")

    if 'Diversity' in df.columns and df['Diversity'].notna().any():
        best_diversity = df.loc[df['Diversity'].idxmax()]
        logger.info(f"\nBest Diversity:")
        logger.info(f"  Checkpoint: {best_diversity['Checkpoint']}")
        logger.info(f"  Diversity: {best_diversity['Diversity']:.4f}")
    
    if 'Combined Score' in df.columns:
        best_combined = df.loc[df['Combined Score'].idxmin()]
        logger.info(f"\nBest Combined Score (recommended):")
        logger.info(f"  Checkpoint: {best_combined['Checkpoint']}")
        logger.info(f"  Combined Score: {best_combined['Combined Score']:.2f}")
        for col in df.columns:
            if col not in ['Checkpoint', 'Label', 'Combined Score'] and not col.endswith('Rank'):
                if col in best_combined and best_combined[col] is not None and not pd.isna(best_combined[col]):
                    logger.info(f"  {col}: {best_combined[col]:.4f}")
    
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
    
    # Get metric configuration from preset
    preset = METRIC_PRESETS[args.preset]
    logger.info(f"Using metric preset: {args.preset} - {preset['description']}")
    metric_config = preset['metrics']
    
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
    
    # Get real images directory
    real_images_dir = load_training_images_dir(config)
    
    # Create prompt
    prompt = config.data.prompt_template.format(labels=config.evaluation.label)
    
    logger.info("=" * 80)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Label: {config.evaluation.label}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Metric preset: {args.preset}")
    logger.info(f"Checkpoints to evaluate: {len(config.evaluation.checkpoint_names)}")
    for name in config.evaluation.checkpoint_names:
        logger.info(f"  - {name}")
    logger.info(f"Images per checkpoint: {config.generation.num_images}")
    logger.info(f"Minimum images to skip generation: {args.min_images}")
    logger.info(f"Inference steps: {config.generation.num_inference_steps}")
    logger.info(f"Guidance scale: {config.generation.guidance_scale}")
    logger.info("=" * 80)
    
    # Store all results
    all_results = {}

    # Evaluate each checkpoint
    for checkpoint_name in config.evaluation.checkpoint_names:
        logger.info("=" * 80)
        logger.info(f"Evaluating checkpoint: {checkpoint_name}")
        logger.info("=" * 80)

        checkpoint_path = config.evaluation.checkpoint_dir / checkpoint_name
        images_dir = config.evaluation.output_dir / f"{checkpoint_name}_{config.evaluation.label}_images"

        # Ensure images exist (generate if needed)
        images_dir = ensure_images_exist(
            checkpoint_name,
            checkpoint_path,
            images_dir,
            config,
            args.min_images
        )

        # Evaluate using DiffusionGenerationEvaluator
        logger.info("Running evaluation metrics...")

        evaluator = DiffusionGenerationEvaluator(
            generated_images_dir=str(images_dir),
            real_images_dir=str(real_images_dir),
            label=config.evaluation.label,
            output_dir=config.evaluation.output_dir / f"{checkpoint_name}_eval",
            device=config.hardware.device,
            # Metric selection from preset
            **metric_config,
            # Parameters
            prompt_template=config.data.prompt_template,
            novelty_metric=config.metrics.novelty_metric,
            max_real_images=None,  # Use all real images
        )

        # Run evaluation with all enabled metrics
        try:
            results = evaluator.evaluate()

            # Save results
            evaluator.save_results()

            # Print summary
            evaluator.print_summary()

        except Exception as e:
            logger.error(f"Evaluation failed for checkpoint '{checkpoint_name}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            results = {}

        # Store for comparison (extract metrics from nested structure)
        # The evaluator returns {'config': {...}, 'novelty': {...}, 'pathology': {...}, ...}
        # We want just the metric results for comparison
        metric_results = {k: v for k, v in results.items() if k != 'config'}
        all_results[checkpoint_name] = metric_results

        logger.info(f"✓ Completed evaluation for {checkpoint_name}")
    
    # Create summary comparison table
    summary_df = create_summary_table(
        all_results,
        config.evaluation.output_dir,
        config.evaluation.label,
        args.preset,
        config.metrics.novelty_metric
    )
    
    # Create comparison plots if we have results
    if summary_df is not None and len(summary_df) > 0:
        logger.info("Creating comparison plot...")
        plot_checkpoint_comparison(
            summary_df,
            config.evaluation.output_dir,
            config.evaluation.label,
            config.metrics.novelty_metric
        )
    
    logger.info("=" * 80)
    logger.info("✓ EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {config.evaluation.output_dir}")
    logger.info(f"\nKey files:")
    logger.info(f"  - checkpoint_comparison.csv")
    logger.info(f"  - {config.evaluation.label}_checkpoint_comparison.png")
    logger.info(f"  - Individual checkpoint results in *_eval/ subdirectories")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
