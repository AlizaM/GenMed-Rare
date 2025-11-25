"""
Evaluate diffusion model generated images.

Comprehensive evaluation with flexible metric selection for different use cases:
1. Checkpoint comparison (fast, medical-focused metrics)
2. Diversity analysis (post-hoc analysis of generated images)
3. Final evaluation (all metrics including expensive FMD/t-SNE)

Usage:
    # Checkpoint comparison (default: novelty, pathology, biovil, diversity)
    python scripts/evaluate_diffusion_generation.py \\
        --generated-dir outputs/.../checkpoint-6500_Fibrosis_images \\
        --real-dir data/pure_class_folders/fibrosis \\
        --label Fibrosis \\
        --output-dir outputs/evaluation
    
    # Diversity analysis only
    python scripts/evaluate_diffusion_generation.py \\
        --generated-dir outputs/.../checkpoint-6500_Fibrosis_images \\
        --real-dir data/pure_class_folders/fibrosis \\
        --label Fibrosis \\
        --preset diversity \\
        --output-dir outputs/diversity_analysis
    
    # Full evaluation (all metrics)
    python scripts/evaluate_diffusion_generation.py \\
        --generated-dir outputs/.../checkpoint-6500_Fibrosis_images \\
        --real-dir data/pure_class_folders/fibrosis \\
        --label Fibrosis \\
        --preset full \\
        --output-dir outputs/final_evaluation
    
    # Custom metric selection
    python scripts/evaluate_diffusion_generation.py \\
        --generated-dir outputs/.../checkpoint-6500_Fibrosis_images \\
        --real-dir data/pure_class_folders/fibrosis \\
        --label Fibrosis \\
        --metrics novelty pathology biovil diversity fmd \\
        --output-dir outputs/custom_evaluation
"""

import argparse
from pathlib import Path
import logging
import sys

import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.eval.diffusion_evaluator import DiffusionGenerationEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Metric presets
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

# Individual metric names to parameter mapping
METRIC_NAMES = {
    'novelty': 'compute_novelty',
    'pathology': 'compute_pathology_confidence',
    'biovil': 'compute_biovil',
    'diversity': 'compute_diversity',
    'pixel_variance': 'compute_pixel_variance',
    'feature_dispersion': 'compute_feature_dispersion',
    'self_similarity': 'compute_self_similarity',
    'fmd': 'compute_fmd',
    'tsne': 'compute_tsne',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion-generated medical images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick checkpoint comparison
  python scripts/evaluate_diffusion_generation.py \\
      --generated-dir outputs/checkpoint-6500_Fibrosis_images \\
      --real-dir data/pure_class_folders/fibrosis \\
      --label Fibrosis

  # Diversity analysis
  python scripts/evaluate_diffusion_generation.py \\
      --generated-dir outputs/checkpoint-6500_Fibrosis_images \\
      --real-dir data/pure_class_folders/fibrosis \\
      --label Fibrosis \\
      --preset diversity

  # Full evaluation with all metrics
  python scripts/evaluate_diffusion_generation.py \\
      --generated-dir outputs/checkpoint-6500_Fibrosis_images \\
      --real-dir data/pure_class_folders/fibrosis \\
      --label Fibrosis \\
      --preset full

Available metrics:
  novelty, pathology, biovil, diversity, pixel_variance, feature_dispersion, 
  self_similarity, fmd, tsne
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--generated-dir",
        type=str,
        required=True,
        help="Directory containing generated images"
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        required=True,
        help="Directory containing real training images"
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label name (e.g., 'Fibrosis', 'Pneumonia')"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: outputs/evaluation/<label>)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (default: cuda if available)"
    )
    
    # Metric selection (mutually exclusive groups)
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument(
        "--preset",
        type=str,
        choices=list(METRIC_PRESETS.keys()),
        default="checkpoint",
        help="Metric preset (default: checkpoint)"
    )
    metric_group.add_argument(
        "--metrics",
        nargs='+',
        choices=list(METRIC_NAMES.keys()),
        help="Custom metric selection (overrides preset)"
    )
    
    # Parameters
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="A chest X-ray showing {label}",
        help="Prompt template for BioViL (use {label} placeholder)"
    )
    parser.add_argument(
        "--max-real-images",
        type=int,
        default=None,
        help="Limit number of real images (for speed)"
    )
    parser.add_argument(
        "--self-similarity-samples",
        type=int,
        default=100,
        help="Number of pairs for self-similarity computation"
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter"
    )
    parser.add_argument(
        "--tsne-n-iter",
        type=int,
        default=1000,
        help="t-SNE number of iterations"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path("outputs") / "evaluation" / args.label
    else:
        output_dir = Path(args.output_dir)
    
    # Determine metric configuration
    if args.metrics is not None:
        # Custom metric selection
        logger.info("Using custom metric selection")
        metric_config = {param: False for param in METRIC_NAMES.values()}
        for metric_name in args.metrics:
            param_name = METRIC_NAMES[metric_name]
            metric_config[param_name] = True
    else:
        # Use preset
        preset = METRIC_PRESETS[args.preset]
        logger.info(f"Using preset: {args.preset} - {preset['description']}")
        metric_config = preset['metrics']
    
    # Create evaluator
    evaluator = DiffusionGenerationEvaluator(
        generated_images_dir=args.generated_dir,
        real_images_dir=args.real_dir,
        label=args.label,
        output_dir=output_dir,
        device=args.device,
        # Metric selection
        **metric_config,
        # Parameters
        prompt_template=args.prompt_template,
        max_real_images=args.max_real_images,
        self_similarity_samples=args.self_similarity_samples,
        tsne_perplexity=args.tsne_perplexity,
        tsne_n_iter=args.tsne_n_iter,
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Save results
    evaluator.save_results()
    
    # Create visualizations
    evaluator.create_visualizations()
    
    # Print summary
    evaluator.print_summary()
    
    logger.info(f"✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
