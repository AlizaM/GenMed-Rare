#!/usr/bin/env python3
"""
Create augmented training datasets by selecting best generated images.

This script creates TWO SEPARATE augmented datasets:
1. Effusion vs Fibrosis (augmented with generated Fibrosis images)
2. Effusion vs Pneumonia (augmented with generated Pneumonia images)

Each dataset combines the original training data with top-N generated images
selected by pathology confidence scores.

Usage:
    python scripts/create_augmented_dataset.py \
        --fibrosis-eval outputs/fibrosis_evaluation/Fibrosis_evaluation_full.json \
        --pneumonia-eval outputs/pneumonia_evaluation/Pneumonia_evaluation_full.json \
        --fibrosis-csv data/processed/effusion_fibrosis/dataset.csv \
        --pneumonia-csv data/processed/effusion_pneumonia/dataset.csv \
        --num-fibrosis 1770 \
        --num-pneumonia 1200
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_evaluation_results(eval_json_path: Path) -> Dict:
    """Load evaluation results JSON."""
    with open(eval_json_path, 'r') as f:
        results = json.load(f)
    return results


def select_top_images_by_pathology(
    eval_results: Dict,
    generated_dir: Path,
    num_images: int,
    label: str
) -> Tuple[List[Path], np.ndarray, float, float]:
    """
    Select top N generated images based on pathology confidence scores.

    Args:
        eval_results: Evaluation results dictionary
        generated_dir: Directory containing generated images
        num_images: Number of images to select
        label: Pathology label name

    Returns:
        Tuple of (selected_paths, selected_scores, min_score, max_score)
    """
    logger.info(f"Selecting top {num_images} {label} images by pathology confidence...")

    # Check if pathology metric exists
    if 'pathology' not in eval_results or eval_results['pathology'] is None:
        raise ValueError(f"Pathology confidence not found in evaluation results for {label}")

    pathology_results = eval_results['pathology']

    # Get confidence scores (should be per-image)
    if 'confidences' in pathology_results:
        confidences = np.array(pathology_results['confidences'])
    else:
        raise ValueError(f"Individual confidence scores not found for {label}")

    # Get list of generated images
    generated_files = sorted(generated_dir.glob("*.png")) + sorted(generated_dir.glob("*.jpg"))

    if len(generated_files) != len(confidences):
        logger.warning(
            f"Mismatch: {len(generated_files)} images vs {len(confidences)} scores. "
            f"Using minimum count."
        )
        min_count = min(len(generated_files), len(confidences))
        generated_files = generated_files[:min_count]
        confidences = confidences[:min_count]

    # Select top N by confidence
    num_to_select = min(num_images, len(confidences))
    top_indices = np.argsort(confidences)[-num_to_select:][::-1]  # Descending order

    selected_paths = [generated_files[i] for i in top_indices]
    selected_scores = confidences[top_indices]

    min_score = float(selected_scores.min())
    max_score = float(selected_scores.max())

    logger.info(f"Selected {len(selected_paths)} images")
    logger.info(f"  Pathology confidence range: [{min_score:.4f}, {max_score:.4f}]")
    logger.info(f"  Mean confidence: {selected_scores.mean():.4f}")

    return selected_paths, selected_scores, min_score, max_score


def create_single_augmented_dataset(
    original_csv: Path,
    generated_paths: List[Path],
    generated_scores: np.ndarray,
    label: str,
    output_dir: Path,
    copy_images: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Create a single augmented dataset by adding generated images of one pathology.

    Args:
        original_csv: Path to original training CSV
        generated_paths: List of selected generated image paths
        generated_scores: Confidence scores for selected images
        label: Pathology label (e.g., "Fibrosis" or "Pneumonia")
        output_dir: Output directory for this dataset
        copy_images: Whether to copy images to output directory

    Returns:
        Tuple of (augmented DataFrame, statistics dict)
    """
    logger.info(f"Creating augmented dataset for {label}...")
    logger.info(f"  Original CSV: {original_csv}")

    # Load original data
    df_original = pd.read_csv(original_csv)
    logger.info(f"  Original dataset: {len(df_original)} samples")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Build list of augmented samples matching the original CSV format:
    # Image Index, image_path, Finding Labels, label, split
    augmented_samples = []

    for img_idx, img_path in enumerate(generated_paths):
        # Create new filename
        new_filename = f"generated_{label.lower()}_{img_idx:04d}.png"
        new_path = images_dir / new_filename

        if copy_images:
            # Copy image to augmented data directory
            shutil.copy2(img_path, new_path)

        # Add to samples with same format as original CSV
        augmented_samples.append({
            'Image Index': new_filename,              # Image filename
            'image_path': str(new_path),              # Full path to image
            'Finding Labels': label,                  # Pathology name (Fibrosis/Pneumonia)
            'label': 1,                               # 1 = rare class (positive)
            'split': 'train',                         # All generated go to train
            'source': 'generated',                    # Mark as generated
            'confidence': float(generated_scores[img_idx])  # Pathology confidence
        })

    # Create DataFrame for generated images
    df_generated = pd.DataFrame(augmented_samples)

    # Add source and confidence columns to original data
    df_original = df_original.copy()
    df_original['source'] = 'real'
    df_original['confidence'] = None

    # Combine original and generated
    df_augmented = pd.concat([df_original, df_generated], ignore_index=True)

    # Compute statistics
    stats = {
        'total_samples': len(df_augmented),
        'real_samples': int((df_augmented['source'] == 'real').sum()),
        'generated_samples': int((df_augmented['source'] == 'generated').sum()),
        'min_confidence': float(generated_scores.min()),
        'max_confidence': float(generated_scores.max()),
        'mean_confidence': float(generated_scores.mean()),
        'class_distribution': {}
    }

    logger.info(f"  Augmented dataset: {stats['total_samples']} samples")
    logger.info(f"    Real: {stats['real_samples']}")
    logger.info(f"    Generated: {stats['generated_samples']}")

    # Class distribution (using Finding Labels for pathology names)
    logger.info("  Class distribution:")
    for finding in df_augmented['Finding Labels'].unique():
        count = int((df_augmented['Finding Labels'] == finding).sum())
        real_count = int(((df_augmented['Finding Labels'] == finding) & (df_augmented['source'] == 'real')).sum())
        gen_count = int(((df_augmented['Finding Labels'] == finding) & (df_augmented['source'] == 'generated')).sum())
        stats['class_distribution'][finding] = {
            'total': count,
            'real': real_count,
            'generated': gen_count
        }
        logger.info(f"    {finding}: {count} total ({real_count} real + {gen_count} generated)")

    # Save CSV
    output_csv = output_dir / "train_augmented.csv"
    df_augmented.to_csv(output_csv, index=False)
    logger.info(f"  ✓ Saved to {output_csv}")

    return df_augmented, stats


def write_summary(
    output_path: Path,
    fibrosis_stats: Optional[Dict],
    pneumonia_stats: Optional[Dict],
    fibrosis_output: Optional[Path],
    pneumonia_output: Optional[Path]
):
    """Write summary file for all created datasets."""
    with open(output_path, 'w') as f:
        f.write("Augmented Datasets Summary\n")
        f.write("=" * 80 + "\n\n")

        if fibrosis_stats:
            f.write("FIBROSIS AUGMENTED DATASET\n")
            f.write("-" * 40 + "\n")
            f.write(f"Output directory: {fibrosis_output}\n")
            f.write(f"Total samples: {fibrosis_stats['total_samples']}\n")
            f.write(f"  Real: {fibrosis_stats['real_samples']}\n")
            f.write(f"  Generated: {fibrosis_stats['generated_samples']}\n")
            f.write(f"Pathology confidence range: [{fibrosis_stats['min_confidence']:.4f}, {fibrosis_stats['max_confidence']:.4f}]\n")
            f.write(f"Mean confidence: {fibrosis_stats['mean_confidence']:.4f}\n")
            f.write("Class distribution:\n")
            for lbl, counts in fibrosis_stats['class_distribution'].items():
                f.write(f"  {lbl}: {counts['total']} ({counts['real']} real + {counts['generated']} gen)\n")
            f.write("\n")

        if pneumonia_stats:
            f.write("PNEUMONIA AUGMENTED DATASET\n")
            f.write("-" * 40 + "\n")
            f.write(f"Output directory: {pneumonia_output}\n")
            f.write(f"Total samples: {pneumonia_stats['total_samples']}\n")
            f.write(f"  Real: {pneumonia_stats['real_samples']}\n")
            f.write(f"  Generated: {pneumonia_stats['generated_samples']}\n")
            f.write(f"Pathology confidence range: [{pneumonia_stats['min_confidence']:.4f}, {pneumonia_stats['max_confidence']:.4f}]\n")
            f.write(f"Mean confidence: {pneumonia_stats['mean_confidence']:.4f}\n")
            f.write("Class distribution:\n")
            for lbl, counts in pneumonia_stats['class_distribution'].items():
                f.write(f"  {lbl}: {counts['total']} ({counts['real']} real + {counts['generated']} gen)\n")
            f.write("\n")

    logger.info(f"✓ Summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create augmented training datasets with selected generated images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create both Fibrosis and Pneumonia augmented datasets
  python scripts/create_augmented_dataset.py \\
      --fibrosis-eval outputs/fibrosis_eval/Fibrosis_evaluation_full.json \\
      --pneumonia-eval outputs/pneumonia_eval/Pneumonia_evaluation_full.json \\
      --fibrosis-csv data/processed/effusion_fibrosis/dataset.csv \\
      --pneumonia-csv data/processed/effusion_pneumonia/dataset.csv \\
      --num-fibrosis 1770 \\
      --num-pneumonia 1200

  # Create only Fibrosis augmented dataset
  python scripts/create_augmented_dataset.py \\
      --fibrosis-eval outputs/fibrosis_eval/Fibrosis_evaluation_full.json \\
      --fibrosis-csv data/processed/effusion_fibrosis/dataset.csv \\
      --num-fibrosis 1770
        """
    )

    # Fibrosis arguments
    parser.add_argument(
        "--fibrosis-eval",
        type=str,
        default=None,
        help="Path to Fibrosis evaluation results JSON (optional, skip if not provided)"
    )
    parser.add_argument(
        "--fibrosis-csv",
        type=str,
        default=None,
        help="Path to original Fibrosis training CSV (e.g., data/processed/effusion_fibrosis/dataset.csv)"
    )
    parser.add_argument(
        "--fibrosis-dir",
        type=str,
        default="outputs/generated_fibrosis_2000",
        help="Directory with generated Fibrosis images"
    )
    parser.add_argument(
        "--num-fibrosis",
        type=int,
        default=1770,
        help="Number of Fibrosis images to select (default: 1770)"
    )

    # Pneumonia arguments
    parser.add_argument(
        "--pneumonia-eval",
        type=str,
        default=None,
        help="Path to Pneumonia evaluation results JSON (optional, skip if not provided)"
    )
    parser.add_argument(
        "--pneumonia-csv",
        type=str,
        default=None,
        help="Path to original Pneumonia training CSV (e.g., data/processed/effusion_pneumonia/dataset.csv)"
    )
    parser.add_argument(
        "--pneumonia-dir",
        type=str,
        default="outputs/generated_pneumonia_2000",
        help="Directory with generated Pneumonia images"
    )
    parser.add_argument(
        "--num-pneumonia",
        type=int,
        default=1200,
        help="Number of Pneumonia images to select (default: 1200)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/augmented_data",
        help="Base output directory for augmented datasets"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Don't copy images, just create CSV with original paths"
    )

    args = parser.parse_args()

    # Validate that at least one dataset is requested
    do_fibrosis = args.fibrosis_eval is not None
    do_pneumonia = args.pneumonia_eval is not None

    if not do_fibrosis and not do_pneumonia:
        parser.error("At least one of --fibrosis-eval or --pneumonia-eval must be provided")

    # Validate Fibrosis arguments
    if do_fibrosis:
        if args.fibrosis_csv is None:
            parser.error("--fibrosis-csv is required when --fibrosis-eval is provided")
        fibrosis_eval = Path(args.fibrosis_eval)
        fibrosis_csv = Path(args.fibrosis_csv)
        fibrosis_dir = Path(args.fibrosis_dir)
        if not fibrosis_eval.exists():
            raise FileNotFoundError(f"Fibrosis evaluation not found: {fibrosis_eval}")
        if not fibrosis_csv.exists():
            raise FileNotFoundError(f"Fibrosis CSV not found: {fibrosis_csv}")
        if not fibrosis_dir.exists():
            raise FileNotFoundError(f"Fibrosis images not found: {fibrosis_dir}")

    # Validate Pneumonia arguments
    if do_pneumonia:
        if args.pneumonia_csv is None:
            parser.error("--pneumonia-csv is required when --pneumonia-eval is provided")
        pneumonia_eval = Path(args.pneumonia_eval)
        pneumonia_csv = Path(args.pneumonia_csv)
        pneumonia_dir = Path(args.pneumonia_dir)
        if not pneumonia_eval.exists():
            raise FileNotFoundError(f"Pneumonia evaluation not found: {pneumonia_eval}")
        if not pneumonia_csv.exists():
            raise FileNotFoundError(f"Pneumonia CSV not found: {pneumonia_csv}")
        if not pneumonia_dir.exists():
            raise FileNotFoundError(f"Pneumonia images not found: {pneumonia_dir}")

    output_dir = Path(args.output_dir)
    copy_images = not args.no_copy

    logger.info("=" * 80)
    logger.info("Creating Augmented Training Datasets")
    logger.info("=" * 80)
    if do_fibrosis:
        logger.info(f"Fibrosis: {args.num_fibrosis} images -> {output_dir}/fibrosis/")
    if do_pneumonia:
        logger.info(f"Pneumonia: {args.num_pneumonia} images -> {output_dir}/pneumonia/")
    logger.info(f"Copy images: {copy_images}")
    logger.info("")

    fibrosis_stats = None
    pneumonia_stats = None
    fibrosis_output = None
    pneumonia_output = None

    # Process Fibrosis
    if do_fibrosis:
        logger.info("=" * 80)
        logger.info("Processing FIBROSIS Dataset")
        logger.info("=" * 80)

        fibrosis_results = load_evaluation_results(fibrosis_eval)
        fibrosis_paths, fib_scores, _, _ = select_top_images_by_pathology(
            fibrosis_results,
            fibrosis_dir,
            args.num_fibrosis,
            "Fibrosis"
        )

        fibrosis_output = output_dir / "fibrosis"
        _, fibrosis_stats = create_single_augmented_dataset(
            fibrosis_csv,
            fibrosis_paths,
            fib_scores,
            "Fibrosis",
            fibrosis_output,
            copy_images=copy_images
        )
        logger.info("")

    # Process Pneumonia
    if do_pneumonia:
        logger.info("=" * 80)
        logger.info("Processing PNEUMONIA Dataset")
        logger.info("=" * 80)

        pneumonia_results = load_evaluation_results(pneumonia_eval)
        pneumonia_paths, pneu_scores, _, _ = select_top_images_by_pathology(
            pneumonia_results,
            pneumonia_dir,
            args.num_pneumonia,
            "Pneumonia"
        )

        pneumonia_output = output_dir / "pneumonia"
        _, pneumonia_stats = create_single_augmented_dataset(
            pneumonia_csv,
            pneumonia_paths,
            pneu_scores,
            "Pneumonia",
            pneumonia_output,
            copy_images=copy_images
        )
        logger.info("")

    # Write summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "augmentation_summary.txt"
    write_summary(summary_path, fibrosis_stats, pneumonia_stats, fibrosis_output, pneumonia_output)

    # Final summary
    logger.info("=" * 80)
    logger.info("Augmented Datasets Created Successfully!")
    logger.info("=" * 80)
    if do_fibrosis:
        logger.info(f"Fibrosis:  {fibrosis_output}/train_augmented.csv")
    if do_pneumonia:
        logger.info(f"Pneumonia: {pneumonia_output}/train_augmented.csv")
    logger.info(f"Summary:   {summary_path}")
    logger.info("")


if __name__ == "__main__":
    main()
