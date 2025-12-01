"""
Diffusion model generation evaluator.

Comprehensive evaluation of diffusion-generated medical images with configurable metrics.
"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import yaml
import logging
from tqdm import tqdm
from PIL import Image

import torch

from src.data.preprocess import crop_border_and_resize
from src.eval.metrics import (
    # Novelty metrics
    compute_novelty_metrics,
    # Medical-specific metrics
    load_torchxrayvision_model,
    compute_pathology_confidence,
    load_biovil_model,
    compute_biovil_scores,
    compute_score_metrics,
    compute_pathology_biovil_correlation,
    # Diversity metrics
    compute_diversity_metrics,
    compute_intra_class_variance,
    compute_feature_dispersion,
    compute_self_similarity,
    # Distribution metrics
    compute_fmd,
    compute_tsne_overlap,
)
from src.utils.image_utils import pil_to_numpy

logger = logging.getLogger(__name__)


class DiffusionGenerationEvaluator:
    """
    Comprehensive evaluator for diffusion-generated medical images.
    
    Supports flexible metric selection for different evaluation scenarios:
    - Checkpoint comparison (fast metrics)
    - Diversity analysis (already-generated images)
    - Final evaluation (all metrics including expensive FMD/t-SNE)
    """
    
    def __init__(
        self,
        generated_images_dir: Path,
        real_images_dir: Path,
        label: str,
        output_dir: Path,
        device: str = "cuda",
        healthy_images_dir: Optional[Path] = None,
        # Metric selection
        compute_novelty: bool = True,
        compute_pathology_confidence: bool = True,
        compute_biovil: bool = True,
        compute_diversity: bool = True,
        compute_pixel_variance: bool = False,
        compute_feature_dispersion: bool = False,
        compute_self_similarity: bool = False,
        compute_fmd: bool = False,
        compute_tsne: bool = False,
        # Parameters
        prompt_template: str = "A chest X-ray showing {label}",
        novelty_metric: str = "ssim",
        max_real_images: Optional[int] = None,
        self_similarity_samples: int = 100,
        tsne_perplexity: int = 30,
        tsne_n_iter: int = 1000,
        # Preprocessing
        crop_border_pixels: int = 0,  # Set to 10 to remove border artifacts
    ):
        """
        Initialize evaluator.
        
        Args:
            generated_images_dir: Directory with generated images
            real_images_dir: Directory with real training images (pathology)
            label: Label name (e.g., "Fibrosis")
            output_dir: Output directory for results
            device: Device for computation
            healthy_images_dir: Optional directory with healthy images for t-SNE comparison
            
            Metric selection flags:
            compute_novelty: SSIM-based novelty vs training set
            compute_pathology_confidence: TorchXRayVision pathology classification
            compute_biovil: BioViL medical text-image alignment
            compute_diversity: TorchXRayVision probability std dev
            compute_pixel_variance: Pixel-level variance
            compute_feature_dispersion: Feature space dispersion
            compute_self_similarity: Pairwise SSIM within batch
            compute_fmd: Fréchet MedicalNet Distance (expensive)
            compute_tsne: t-SNE overlap analysis (very expensive)
            
            Parameters:
            prompt_template: Template for text prompts (use {label} placeholder)
            max_real_images: Limit real images for speed
            self_similarity_samples: Number of pairs for self-similarity
            tsne_perplexity: t-SNE perplexity parameter
            tsne_n_iter: t-SNE iterations
            crop_border_pixels: Crop N pixels from each side and resize back (0 = no crop)
        """
        self.generated_images_dir = Path(generated_images_dir)
        self.real_images_dir = Path(real_images_dir)
        self.healthy_images_dir = Path(healthy_images_dir) if healthy_images_dir is not None else None
        self.label = label
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Metric flags
        self.compute_novelty = compute_novelty
        self.compute_pathology_confidence = compute_pathology_confidence
        self.compute_biovil = compute_biovil
        self.compute_diversity = compute_diversity
        self.compute_pixel_variance = compute_pixel_variance
        self.compute_feature_dispersion = compute_feature_dispersion
        self.compute_self_similarity = compute_self_similarity
        self.compute_fmd = compute_fmd
        self.compute_tsne = compute_tsne
        
        # Parameters
        self.prompt_template = prompt_template
        self.novelty_metric = novelty_metric
        self.max_real_images = max_real_images
        self.self_similarity_samples = self_similarity_samples
        self.tsne_perplexity = tsne_perplexity
        self.tsne_n_iter = tsne_n_iter
        self.crop_border_pixels = crop_border_pixels

        # Validate paths
        if not self.generated_images_dir.exists():
            raise FileNotFoundError(f"Generated images not found: {self.generated_images_dir}")
        if not self.real_images_dir.exists():
            raise FileNotFoundError(f"Real images not found: {self.real_images_dir}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.generated_images: List[np.ndarray] = []
        self.real_images: List[np.ndarray] = []
        self.healthy_images: List[np.ndarray] = []
        self.results: Dict = {}
        
        # Models (loaded on demand)
        self.xrv_model = None
        self.biovil_image_inference = None
        self.biovil_device = None  # Actual device used for BioViL (may fallback to CPU)
        
        logger.info("=" * 80)
        logger.info("DiffusionGenerationEvaluator initialized")
        logger.info("=" * 80)
        logger.info(f"Generated images: {self.generated_images_dir}")
        logger.info(f"Real images: {self.real_images_dir}")
        logger.info(f"Label: {self.label}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info("")
        logger.info("Metrics enabled:")
        logger.info(f"  Novelty (SSIM): {self.compute_novelty}")
        logger.info(f"  Pathology Confidence: {self.compute_pathology_confidence}")
        logger.info(f"  BioViL: {self.compute_biovil}")
        logger.info(f"  Diversity (XRV): {self.compute_diversity}")
        logger.info(f"  Pixel Variance: {self.compute_pixel_variance}")
        logger.info(f"  Feature Dispersion: {self.compute_feature_dispersion}")
        logger.info(f"  Self-Similarity: {self.compute_self_similarity}")
        logger.info(f"  FMD: {self.compute_fmd}")
        logger.info(f"  t-SNE: {self.compute_tsne}")
        logger.info("=" * 80)
    
    def load_images(self):
        """Load generated and real images as grayscale with optional border cropping."""
        logger.info("Loading images...")
        logger.info("  All images loaded as grayscale (medical X-rays)")

        if self.crop_border_pixels > 0:
            logger.info(f"  Border crop enabled: {self.crop_border_pixels}px (scaled proportionally to image size)")
            logger.info(f"    - 512x512 images: {self.crop_border_pixels}px crop")
            logger.info(f"    - 1024x1024 images: {self.crop_border_pixels * 2}px crop (same zoom ratio)")

        # Load generated images
        logger.info(f"Loading generated images from {self.generated_images_dir}...")
        gen_files = sorted(self.generated_images_dir.glob("*.png")) + sorted(self.generated_images_dir.glob("*.jpg"))
        if len(gen_files) == 0:
            raise ValueError(f"No images found in {self.generated_images_dir}")

        for img_path in tqdm(gen_files, desc="Loading generated"):
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_np = pil_to_numpy(img)

            # Apply border cropping if requested
            if self.crop_border_pixels > 0:
                img_np = crop_border_and_resize(img_np, crop_pixels=self.crop_border_pixels)

            self.generated_images.append(img_np)

        logger.info(f"✓ Loaded {len(self.generated_images)} generated images (shape: {self.generated_images[0].shape})")

        # Load real images
        logger.info(f"Loading real images from {self.real_images_dir}...")
        real_files = sorted(self.real_images_dir.glob("*.png")) + sorted(self.real_images_dir.glob("*.jpg"))
        if len(real_files) == 0:
            raise ValueError(f"No images found in {self.real_images_dir}")

        if self.max_real_images is not None and len(real_files) > self.max_real_images:
            logger.info(f"Limiting to {self.max_real_images} real images (out of {len(real_files)})")
            real_files = real_files[:self.max_real_images]

        for img_path in tqdm(real_files, desc="Loading real"):
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_np = pil_to_numpy(img)

            # Apply border cropping if requested
            if self.crop_border_pixels > 0:
                img_np = crop_border_and_resize(img_np, crop_pixels=self.crop_border_pixels)

            self.real_images.append(img_np)

        logger.info(f"✓ Loaded {len(self.real_images)} real images (shape: {self.real_images[0].shape})")

        # Load healthy images if directory provided
        if self.healthy_images_dir is not None and self.compute_tsne:
            logger.info(f"Loading healthy images from {self.healthy_images_dir}...")
            healthy_files = sorted(self.healthy_images_dir.glob("*.png")) + sorted(self.healthy_images_dir.glob("*.jpg"))

            if len(healthy_files) == 0:
                logger.warning(f"No healthy images found in {self.healthy_images_dir}, skipping healthy group in t-SNE")
            else:
                # Optionally limit healthy images to match generated count
                if len(healthy_files) > len(self.generated_images):
                    logger.info(f"Limiting to {len(self.generated_images)} healthy images (out of {len(healthy_files)})")
                    healthy_files = healthy_files[:len(self.generated_images)]

                for img_path in tqdm(healthy_files, desc="Loading healthy"):
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img_np = pil_to_numpy(img)

                    # Apply border cropping if requested
                    if self.crop_border_pixels > 0:
                        img_np = crop_border_and_resize(img_np, crop_pixels=self.crop_border_pixels)

                    self.healthy_images.append(img_np)

                logger.info(f"✓ Loaded {len(self.healthy_images)} healthy images (shape: {self.healthy_images[0].shape})")
    
    def _ensure_xrv_model(self):
        """Load TorchXRayVision model if not already loaded."""
        if self.xrv_model is None:
            logger.info("Loading TorchXRayVision model...")
            self.xrv_model = load_torchxrayvision_model(self.device)
            logger.info("✓ TorchXRayVision model loaded")
    
    def _ensure_biovil_model(self):
        """Load BioViL model if not already loaded."""
        if self.biovil_image_inference is None:
            logger.info("Loading BioViL model...")
            self.biovil_image_inference, self.biovil_device = load_biovil_model(self.device)
            if self.biovil_image_inference is not None:
                logger.info(f"✓ BioViL model loaded on {self.biovil_device}")
    
    def evaluate(self) -> Dict:
        """
        Run complete evaluation with selected metrics.
        
        Returns:
            Dictionary of all computed metrics
        """
        logger.info("=" * 80)
        logger.info("STARTING EVALUATION")
        logger.info("=" * 80)
        
        # Load images
        self.load_images()
        
        # Prepare results dictionary
        self.results = {
            'config': {
                'generated_dir': str(self.generated_images_dir),
                'real_dir': str(self.real_images_dir),
                'label': self.label,
                'num_generated': len(self.generated_images),
                'num_real': len(self.real_images),
            }
        }
        
        # 1. Novelty metrics
        if self.compute_novelty:
            try:
                logger.info("=" * 80)
                logger.info(f"Computing Novelty Metrics ({self.novelty_metric.upper()})")
                logger.info("=" * 80)
                novelty = compute_novelty_metrics(
                    self.generated_images,
                    self.real_images,
                    metric=self.novelty_metric,
                    show_progress=True
                )
                self.results['novelty'] = novelty
                metric_name = "SSIM" if self.novelty_metric == "ssim" else "Correlation"
                logger.info(f"✓ Max Novelty (1-{metric_name}): {novelty['max_novelty']:.4f}")
                logger.info(f"✓ P99 Novelty (1-{metric_name}): {novelty['p99_novelty']:.4f}")
                logger.info(f"✓ Mean Novelty (1-{metric_name}): {novelty['mean_novelty']:.4f}")
            except Exception as e:
                logger.error(f"✗ Novelty metric failed: {e}")
                self.results['novelty'] = None
        
        # 2. Pathology confidence
        if self.compute_pathology_confidence:
            try:
                self._ensure_xrv_model()
                logger.info("=" * 80)
                logger.info(f"Computing Pathology Confidence ({self.label})")
                logger.info("=" * 80)
                pathology = compute_pathology_confidence(
                    self.generated_images,
                    self.label,
                    self.xrv_model,
                    self.device,
                    show_progress=True
                )
                self.results['pathology'] = pathology
                logger.info(f"✓ Mean confidence: {pathology['mean_confidence']:.3f}")
                logger.info(f"✓ Median confidence: {pathology['median_confidence']:.3f}")
            except Exception as e:
                logger.error(f"✗ Pathology confidence failed: {e}")
                self.results['pathology'] = None
        
        # 3. BioViL scores
        if self.compute_biovil:
            try:
                self._ensure_biovil_model()
                logger.info("=" * 80)
                logger.info("Computing BioViL Scores")
                logger.info("=" * 80)
                prompt = self.prompt_template.format(label=self.label)
                prompts = [prompt] * len(self.generated_images)
                # Use biovil_device if available (may fallback to CPU)
                biovil_device = getattr(self, 'biovil_device', None) or self.device
                biovil_scores = compute_biovil_scores(
                    self.generated_images,
                    prompts,
                    self.biovil_image_inference,
                    None,  # text_inference unused
                    biovil_device,
                    show_progress=True
                )
                biovil_metrics = compute_score_metrics(biovil_scores)
                self.results['biovil'] = biovil_metrics
                if biovil_metrics and biovil_metrics.get('mean_score') is not None:
                    logger.info(f"✓ Mean BioViL: {biovil_metrics['mean_score']:.3f}")
                    logger.info(f"✓ Median BioViL: {biovil_metrics['median_score']:.3f}")
                else:
                    logger.warning("✗ BioViL model not available or all scores failed")
            except Exception as e:
                logger.error(f"✗ BioViL metric failed: {e}")
                self.results['biovil'] = None

        # 3b. Pathology-BioViL Correlation (if both computed)
        if (self.results.get('pathology') is not None and
            self.results.get('biovil') is not None and
            'confidences' in self.results['pathology'] and
            'scores' in self.results['biovil']):
            try:
                logger.info("=" * 80)
                logger.info("Computing Pathology-BioViL Correlation")
                logger.info("=" * 80)
                correlation = compute_pathology_biovil_correlation(
                    self.results['pathology']['confidences'],
                    self.results['biovil']['scores']
                )
                self.results['pathology_biovil_correlation'] = correlation
                if correlation['pearson_r'] is not None:
                    logger.info(f"✓ Pearson r: {correlation['pearson_r']:.3f} (p={correlation['pearson_p']:.2e})")
                    logger.info(f"✓ Spearman r: {correlation['spearman_r']:.3f} (p={correlation['spearman_p']:.2e})")
                    logger.info(f"✓ Valid pairs: {correlation['num_valid_pairs']}")
            except Exception as e:
                logger.error(f"✗ Pathology-BioViL correlation failed: {e}")
                self.results['pathology_biovil_correlation'] = None

        # 4. Diversity (XRV std dev)
        logger.info(f"DEBUG: compute_diversity = {self.compute_diversity}")
        if self.compute_diversity:
            try:
                logger.info("DEBUG: Starting diversity computation...")
                self._ensure_xrv_model()
                logger.info("=" * 80)
                logger.info("Computing Diversity (XRV Std Dev)")
                logger.info("=" * 80)
                logger.info(f"Number of images for diversity: {len(self.generated_images)}")
                diversity = compute_diversity_metrics(
                    self.generated_images,
                    self.xrv_model,
                    self.device,
                    show_progress=True
                )
                logger.info("DEBUG: Diversity computation completed")
                self.results['diversity'] = diversity
                logger.info(f"✓ Overall diversity: {diversity['overall_diversity']:.4f}")
                logger.info(f"✓ Mean std: {diversity['mean_std']:.4f}")
            except Exception as e:
                logger.error(f"✗ Diversity metric failed: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.results['diversity'] = None
        else:
            logger.info("DEBUG: Skipping diversity (compute_diversity=False)")
        
        # 5. Pixel variance
        if self.compute_pixel_variance:
            try:
                logger.info("=" * 80)
                logger.info("Computing Pixel Variance")
                logger.info("=" * 80)
                pixel_var = compute_intra_class_variance(self.generated_images)
                self.results['pixel_variance'] = pixel_var
                logger.info(f"✓ Mean pixel variance: {pixel_var['mean_pixel_variance']:.2f}")
            except Exception as e:
                logger.error(f"✗ Pixel variance metric failed: {e}")
                self.results['pixel_variance'] = None
        
        # 6. Feature dispersion
        if self.compute_feature_dispersion:
            try:
                self._ensure_xrv_model()
                logger.info("=" * 80)
                logger.info("Computing Feature Dispersion")
                logger.info("=" * 80)
                dispersion = compute_feature_dispersion(
                    self.generated_images,
                    self.xrv_model,
                    self.device,
                    show_progress=True
                )
                self.results['feature_dispersion'] = dispersion
                logger.info(f"✓ Mean pairwise distance: {dispersion['mean_pairwise_distance']:.2f}")
                logger.info(f"✓ Trace of covariance: {dispersion['trace_covariance']:.2e}")
            except Exception as e:
                logger.error(f"✗ Feature dispersion metric failed: {e}")
                self.results['feature_dispersion'] = None
        
        # 7. Self-similarity
        if self.compute_self_similarity:
            try:
                logger.info("=" * 80)
                logger.info("Computing Self-Similarity")
                logger.info("=" * 80)
                self_sim = compute_self_similarity(
                    self.generated_images,
                    num_samples=self.self_similarity_samples,
                    show_progress=True
                )
                self.results['self_similarity'] = self_sim
                logger.info(f"✓ Mean self-SSIM: {self_sim['mean_self_ssim']:.4f}")
                logger.info(f"✓ Median self-SSIM: {self_sim['median_self_ssim']:.4f}")
            except Exception as e:
                logger.error(f"✗ Self-similarity metric failed: {e}")
                self.results['self_similarity'] = None
        
        # 8. FMD
        if self.compute_fmd:
            try:
                self._ensure_xrv_model()
                logger.info("=" * 80)
                logger.info("Computing FMD (Fréchet MedicalNet Distance)")
                logger.info("=" * 80)
                logger.info("This is computationally expensive...")
                fmd_score = compute_fmd(
                    self.generated_images,
                    self.real_images,
                    self.xrv_model,
                    self.device,
                    show_progress=True
                )
                self.results['fmd'] = {'score': float(fmd_score)}
                logger.info(f"✓ FMD: {fmd_score:.2f}")
            except Exception as e:
                logger.error(f"✗ FMD metric failed: {e}")
                self.results['fmd'] = None
        
        # 9. t-SNE
        if self.compute_tsne:
            try:
                self._ensure_xrv_model()
                logger.info("=" * 80)
                logger.info("Computing t-SNE Overlap")
                logger.info("=" * 80)
                logger.info("This is very computationally expensive...")
                tsne_results = compute_tsne_overlap(
                    self.generated_images,
                    self.real_images,
                    self.xrv_model,
                    self.device,
                    perplexity=self.tsne_perplexity,
                    n_iter=self.tsne_n_iter,
                    show_progress=True,
                    healthy_images=self.healthy_images if len(self.healthy_images) > 0 else None
                )
                self.results['tsne'] = tsne_results
                logger.info(f"✓ t-SNE overlap: {tsne_results['overlap_score']:.3f}")
                logger.info(f"✓ Mean distance: {tsne_results['mean_distance']:.3f}")
            except Exception as e:
                logger.error(f"✗ t-SNE metric failed: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.results['tsne'] = None
        
        logger.info("=" * 80)
        logger.info("✓ EVALUATION COMPLETE")
        logger.info("=" * 80)
        
        return self.results
    
    def save_results(self):
        """Save results to YAML file (summary) and JSON file (full results)."""
        # Save summary YAML (without large arrays)
        results_path = self.output_dir / f"{self.label}_evaluation_results.yaml"

        # Create a copy without large lists for summary
        summary = {k: v for k, v in self.results.items()}

        # Remove large arrays to keep file size manageable
        if 'novelty' in summary and summary['novelty'] is not None and 'nn_scores' in summary['novelty']:
            summary['novelty'] = {k: v for k, v in summary['novelty'].items()
                                 if k not in ['nn_scores', 'nn_indices']}
        if 'pathology' in summary and summary['pathology'] is not None and 'confidences' in summary['pathology']:
            summary['pathology'] = {k: v for k, v in summary['pathology'].items()
                                   if k != 'confidences'}  # Keep only statistics, not all 2000 probabilities
        if 'biovil' in summary and summary['biovil'] is not None and 'scores' in summary['biovil']:
            summary['biovil'] = {k: v for k, v in summary['biovil'].items()
                                if k != 'scores'}
        if 'diversity' in summary and summary['diversity'] is not None and 'pathology_stds' in summary['diversity']:
            summary['diversity'] = {k: v for k, v in summary['diversity'].items()
                                   if k != 'pathology_stds'}
        if 'self_similarity' in summary and summary['self_similarity'] is not None and 'self_ssim_scores' in summary['self_similarity']:
            summary['self_similarity'] = {k: v for k, v in summary['self_similarity'].items()
                                         if k != 'self_ssim_scores'}
        if 'tsne' in summary and summary['tsne'] is not None:
            summary['tsne'] = {k: v for k, v in summary['tsne'].items()
                              if k not in ['tsne_embeddings', 'labels']}

        # Save summary YAML
        with open(results_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)

        logger.info(f"✓ Summary saved to {results_path}")

        # Save full results as JSON (includes per-image scores for augmentation)
        json_path = self.output_dir / f"{self.label}_evaluation_full.json"
        import json

        # Convert numpy arrays to lists for JSON serialization
        full_results = {}
        for key, value in self.results.items():
            if value is None:
                full_results[key] = None
            elif isinstance(value, dict):
                full_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        full_results[key][k] = v.tolist()
                    else:
                        full_results[key][k] = v
            else:
                full_results[key] = value

        with open(json_path, 'w') as f:
            json.dump(full_results, f, indent=2)

        logger.info(f"✓ Full results saved to {json_path}")
    
    def create_visualizations(self):
        """Create visualization plots."""
        import matplotlib.pyplot as plt
        
        # t-SNE visualization if computed
        if self.compute_tsne and 'tsne' in self.results and self.results['tsne'] is not None:
            logger.info("Creating t-SNE visualization...")

            embeddings = np.array(self.results['tsne']['tsne_embeddings'])
            labels = np.array(self.results['tsne']['labels'])

            plt.figure(figsize=(12, 10))

            # Plot real pathology images (label 0)
            plt.scatter(
                embeddings[labels == 0, 0],
                embeddings[labels == 0, 1],
                c='blue',
                alpha=0.4,
                label=f'Real {self.label} (n={len(self.real_images)})',
                s=30,
                edgecolors='none'
            )

            # Plot generated images (label 1)
            plt.scatter(
                embeddings[labels == 1, 0],
                embeddings[labels == 1, 1],
                c='red',
                alpha=0.4,
                label=f'Generated {self.label} (n={len(self.generated_images)})',
                s=30,
                edgecolors='none'
            )

            # Plot healthy images if present (label 2)
            if len(self.healthy_images) > 0:
                plt.scatter(
                    embeddings[labels == 2, 0],
                    embeddings[labels == 2, 1],
                    c='green',
                    alpha=0.4,
                    label=f'Healthy (n={len(self.healthy_images)})',
                    s=30,
                    edgecolors='none'
                )
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            
            title = f't-SNE: Real vs Generated {self.label}\nOverlap: {self.results["tsne"]["overlap_score"]:.3f}'
            if 'fmd' in self.results:
                title += f' | FMD: {self.results["fmd"]["score"]:.2f}'
            plt.title(title, fontsize=14, fontweight='bold')
            
            plt.legend(fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            tsne_path = self.output_dir / f'{self.label}_tsne_visualization.png'
            plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ t-SNE visualization saved to {tsne_path}")

        # Novelty visualization: Top 5 most similar real-generated pairs
        if self.compute_novelty and 'novelty' in self.results and self.results['novelty'] is not None:
            novelty_results = self.results['novelty']
            if 'nn_indices' in novelty_results and 'nn_scores' in novelty_results:
                logger.info("Creating novelty comparison visualization (top 5 most similar pairs)...")

                nn_indices = np.array(novelty_results['nn_indices'])
                nn_novelty_scores = np.array(novelty_results['nn_scores'])  # These are NOVELTY scores (1 - similarity)

                # Convert novelty back to similarity for display (similarity = 1 - novelty)
                nn_similarity_scores = 1.0 - nn_novelty_scores

                # Get indices of 5 most similar pairs (LOWEST novelty = highest similarity)
                top_k = min(5, len(nn_novelty_scores))
                top_indices = np.argsort(nn_novelty_scores)[:top_k]  # Ascending order (lowest novelty first)

                # Create 2-row grid: top row = real, bottom row = generated
                fig, axes = plt.subplots(2, top_k, figsize=(top_k * 3, 6))

                for i, sorted_idx  in enumerate(top_indices):
                    gen_img = self.generated_images[sorted_idx]
                    real_idx = nn_indices[sorted_idx ]
                    real_img = self.real_images[real_idx]
                    similarity_score = nn_similarity_scores[sorted_idx]

                    # Top row: Real images
                    ax_real = axes[0, i] if top_k > 1 else axes[0]
                    ax_real.imshow(real_img, cmap='gray')
                    ax_real.axis('off')
                    ax_real.set_title(f'Real #{real_idx}', fontsize=10, fontweight='bold')

                    # Bottom row: Generated images
                    ax_gen = axes[1, i] if top_k > 1 else axes[1]
                    ax_gen.imshow(gen_img, cmap='gray')
                    ax_gen.axis('off')
                    metric_name = "SSIM" if self.novelty_metric == "ssim" else "Corr"
                    ax_gen.set_title(f'Gen #{sorted_idx}\n{metric_name}={similarity_score:.3f}',
                                    fontsize=10, fontweight='bold')

                # Add row labels
                fig.text(0.02, 0.75, 'Real Images', va='center', rotation='vertical',
                        fontsize=12, fontweight='bold')
                fig.text(0.02, 0.25, 'Generated Images', va='center', rotation='vertical',
                        fontsize=12, fontweight='bold')

                metric_full_name = "SSIM" if self.novelty_metric == "ssim" else "Correlation"
                plt.suptitle(f'Top {top_k} Most Similar Real-Generated Pairs ({self.label})\n'
                           f'Metric: {metric_full_name} (higher = more similar)',
                           fontsize=14, fontweight='bold')
                plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])

                novelty_path = self.output_dir / f'{self.label}_top_similar_pairs.png'
                plt.savefig(novelty_path, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"✓ Novelty comparison saved to {novelty_path}")

        # Pathology-BioViL correlation scatter plot
        if ('pathology_biovil_correlation' in self.results and
            self.results['pathology_biovil_correlation'] is not None and
            self.results['pathology_biovil_correlation'].get('pearson_r') is not None):
            logger.info("Creating Pathology-BioViL correlation scatter plot...")

            corr = self.results['pathology_biovil_correlation']
            pathology_scores = np.array(corr['pathology_scores'])
            biovil_scores = np.array(corr['biovil_scores'])

            fig, ax = plt.subplots(figsize=(10, 8))

            # Scatter plot with color based on density
            scatter = ax.scatter(
                pathology_scores,
                biovil_scores,
                c='steelblue',
                alpha=0.5,
                s=20,
                edgecolors='none'
            )

            # Add regression line
            z = np.polyfit(pathology_scores, biovil_scores, 1)
            p = np.poly1d(z)
            x_line = np.linspace(pathology_scores.min(), pathology_scores.max(), 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Linear fit')

            ax.set_xlabel('Pathology Confidence (TorchXRayVision)', fontsize=12)
            ax.set_ylabel('BioViL Score (Text-Image Alignment)', fontsize=12)

            title = (f'Pathology vs BioViL Scores - {self.label}\n'
                    f'Pearson r={corr["pearson_r"]:.3f} (p={corr["pearson_p"]:.2e}) | '
                    f'Spearman r={corr["spearman_r"]:.3f} (p={corr["spearman_p"]:.2e})\n'
                    f'n={corr["num_valid_pairs"]} images')
            ax.set_title(title, fontsize=12, fontweight='bold')

            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

            plt.tight_layout()

            scatter_path = self.output_dir / f'{self.label}_pathology_biovil_scatter.png'
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✓ Pathology-BioViL scatter plot saved to {scatter_path}")

    def print_summary(self):
        """Print evaluation summary."""
        logger.info("=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Label: {self.label}")
        logger.info(f"Generated images: {self.results['config']['num_generated']}")
        logger.info(f"Real images: {self.results['config']['num_real']}")
        logger.info("")

        # Determine metric name for novelty
        metric_name = "SSIM" if self.novelty_metric == "ssim" else "Correlation"

        if 'novelty' in self.results and self.results['novelty'] is not None:
            logger.info("Novelty:")
            logger.info(f"  P99 Novelty (1-{metric_name}): {self.results['novelty']['p99_novelty']:.4f} (higher = more novel)")

        if 'pathology' in self.results and self.results['pathology'] is not None:
            logger.info("Pathology Confidence:")
            logger.info(f"  Mean: {self.results['pathology']['mean_confidence']:.3f} (higher = better)")

        if 'biovil' in self.results and self.results['biovil'] is not None:
            if self.results['biovil'].get('mean_score') is not None:
                logger.info("BioViL:")
                logger.info(f"  Mean: {self.results['biovil']['mean_score']:.3f} (higher = better)")

        if ('pathology_biovil_correlation' in self.results and
            self.results['pathology_biovil_correlation'] is not None and
            self.results['pathology_biovil_correlation'].get('spearman_r') is not None):
            corr = self.results['pathology_biovil_correlation']
            logger.info("Pathology-BioViL Correlation:")
            logger.info(f"  Spearman r: {corr['spearman_r']:.3f} (p={corr['spearman_p']:.2e})")

        if 'diversity' in self.results and self.results['diversity'] is not None:
            logger.info("Diversity:")
            logger.info(f"  Overall: {self.results['diversity']['overall_diversity']:.4f} (higher = more diverse)")
        
        if 'self_similarity' in self.results and self.results['self_similarity'] is not None:
            logger.info("Self-Similarity:")
            logger.info(f"  Mean: {self.results['self_similarity']['mean_self_ssim']:.4f} (lower = more diverse)")

        if 'fmd' in self.results and self.results['fmd'] is not None:
            logger.info("FMD:")
            logger.info(f"  Score: {self.results['fmd']['score']:.2f} (lower = better match to real)")

        if 'tsne' in self.results and self.results['tsne'] is not None:
            logger.info("t-SNE:")
            logger.info(f"  Overlap: {self.results['tsne']['overlap_score']:.3f} (higher = better)")

        logger.info("=" * 80)
