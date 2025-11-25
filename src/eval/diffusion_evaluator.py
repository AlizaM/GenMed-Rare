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

from src.eval.metrics import (
    # Novelty metrics
    compute_novelty_metrics,
    # Medical-specific metrics
    load_torchxrayvision_model,
    compute_pathology_confidence,
    load_biovil_model,
    compute_biovil_scores,
    compute_score_metrics,
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
        max_real_images: Optional[int] = None,
        self_similarity_samples: int = 100,
        tsne_perplexity: int = 30,
        tsne_n_iter: int = 1000,
    ):
        """
        Initialize evaluator.
        
        Args:
            generated_images_dir: Directory with generated images
            real_images_dir: Directory with real training images
            label: Label name (e.g., "Fibrosis")
            output_dir: Output directory for results
            device: Device for computation
            
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
        """
        self.generated_images_dir = Path(generated_images_dir)
        self.real_images_dir = Path(real_images_dir)
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
        self.max_real_images = max_real_images
        self.self_similarity_samples = self_similarity_samples
        self.tsne_perplexity = tsne_perplexity
        self.tsne_n_iter = tsne_n_iter
        
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
        self.results: Dict = {}
        
        # Models (loaded on demand)
        self.xrv_model = None
        self.biovil_model = None
        self.biovil_tokenizer = None
        
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
        """Load generated and real images."""
        logger.info("Loading images...")
        
        # Load generated images
        logger.info(f"Loading generated images from {self.generated_images_dir}...")
        gen_files = sorted(self.generated_images_dir.glob("*.png")) + sorted(self.generated_images_dir.glob("*.jpg"))
        if len(gen_files) == 0:
            raise ValueError(f"No images found in {self.generated_images_dir}")
        
        for img_path in tqdm(gen_files, desc="Loading generated"):
            img = Image.open(img_path)
            self.generated_images.append(pil_to_numpy(img))
        
        logger.info(f"✓ Loaded {len(self.generated_images)} generated images")
        
        # Load real images
        logger.info(f"Loading real images from {self.real_images_dir}...")
        real_files = sorted(self.real_images_dir.glob("*.png")) + sorted(self.real_images_dir.glob("*.jpg"))
        if len(real_files) == 0:
            raise ValueError(f"No images found in {self.real_images_dir}")
        
        if self.max_real_images is not None and len(real_files) > self.max_real_images:
            logger.info(f"Limiting to {self.max_real_images} real images (out of {len(real_files)})")
            real_files = real_files[:self.max_real_images]
        
        for img_path in tqdm(real_files, desc="Loading real"):
            img = Image.open(img_path)
            self.real_images.append(pil_to_numpy(img))
        
        logger.info(f"✓ Loaded {len(self.real_images)} real images")
    
    def _ensure_xrv_model(self):
        """Load TorchXRayVision model if not already loaded."""
        if self.xrv_model is None:
            logger.info("Loading TorchXRayVision model...")
            self.xrv_model = load_torchxrayvision_model(self.device)
            logger.info("✓ TorchXRayVision model loaded")
    
    def _ensure_biovil_model(self):
        """Load BioViL model if not already loaded."""
        if self.biovil_model is None:
            logger.info("Loading BioViL model...")
            self.biovil_model, self.biovil_tokenizer = load_biovil_model(self.device)
            logger.info("✓ BioViL model loaded")
    
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
            logger.info("=" * 80)
            logger.info("Computing Novelty Metrics (SSIM)")
            logger.info("=" * 80)
            novelty = compute_novelty_metrics(
                self.generated_images,
                self.real_images,
                metric="ssim",
                show_progress=True
            )
            self.results['novelty'] = novelty
            logger.info(f"✓ Max SSIM: {novelty['max_similarity']:.4f}")
            logger.info(f"✓ P99 SSIM: {novelty['p99_similarity']:.4f}")
            logger.info(f"✓ Mean SSIM: {novelty['mean_similarity']:.4f}")
        
        # 2. Pathology confidence
        if self.compute_pathology_confidence:
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
        
        # 3. BioViL scores
        if self.compute_biovil:
            self._ensure_biovil_model()
            logger.info("=" * 80)
            logger.info("Computing BioViL Scores")
            logger.info("=" * 80)
            prompt = self.prompt_template.format(label=self.label)
            prompts = [prompt] * len(self.generated_images)
            biovil_scores = compute_biovil_scores(
                self.generated_images,
                prompts,
                self.biovil_model,
                self.biovil_tokenizer,
                self.device,
                show_progress=True
            )
            biovil_metrics = compute_score_metrics(biovil_scores)
            self.results['biovil'] = biovil_metrics
            logger.info(f"✓ Mean BioViL: {biovil_metrics['mean_score']:.3f}")
            logger.info(f"✓ Median BioViL: {biovil_metrics['median_score']:.3f}")
        
        # 4. Diversity (XRV std dev)
        if self.compute_diversity:
            self._ensure_xrv_model()
            logger.info("=" * 80)
            logger.info("Computing Diversity (XRV Std Dev)")
            logger.info("=" * 80)
            diversity = compute_diversity_metrics(
                self.generated_images,
                self.xrv_model,
                self.device,
                show_progress=True
            )
            self.results['diversity'] = diversity
            logger.info(f"✓ Overall diversity: {diversity['overall_diversity']:.4f}")
            logger.info(f"✓ Mean std: {diversity['mean_std']:.4f}")
        
        # 5. Pixel variance
        if self.compute_pixel_variance:
            logger.info("=" * 80)
            logger.info("Computing Pixel Variance")
            logger.info("=" * 80)
            pixel_var = compute_intra_class_variance(self.generated_images)
            self.results['pixel_variance'] = pixel_var
            logger.info(f"✓ Mean pixel variance: {pixel_var['mean_pixel_variance']:.2f}")
        
        # 6. Feature dispersion
        if self.compute_feature_dispersion:
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
        
        # 7. Self-similarity
        if self.compute_self_similarity:
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
        
        # 8. FMD
        if self.compute_fmd:
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
        
        # 9. t-SNE
        if self.compute_tsne:
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
                show_progress=True
            )
            self.results['tsne'] = tsne_results
            logger.info(f"✓ t-SNE overlap: {tsne_results['overlap_score']:.3f}")
            logger.info(f"✓ Mean distance: {tsne_results['mean_distance']:.3f}")
        
        logger.info("=" * 80)
        logger.info("✓ EVALUATION COMPLETE")
        logger.info("=" * 80)
        
        return self.results
    
    def save_results(self):
        """Save results to YAML file."""
        results_path = self.output_dir / f"{self.label}_evaluation_results.yaml"
        
        # Create a copy without large lists for summary
        summary = {k: v for k, v in self.results.items()}
        
        # Remove large arrays to keep file size manageable
        if 'novelty' in summary and 'nn_scores' in summary['novelty']:
            summary['novelty'] = {k: v for k, v in summary['novelty'].items() 
                                 if k not in ['nn_scores', 'nn_indices']}
        if 'biovil' in summary and 'scores' in summary['biovil']:
            summary['biovil'] = {k: v for k, v in summary['biovil'].items() 
                                if k != 'scores'}
        if 'diversity' in summary and 'pathology_stds' in summary['diversity']:
            summary['diversity'] = {k: v for k, v in summary['diversity'].items() 
                                   if k != 'pathology_stds'}
        if 'self_similarity' in summary and 'self_ssim_scores' in summary['self_similarity']:
            summary['self_similarity'] = {k: v for k, v in summary['self_similarity'].items() 
                                         if k != 'self_ssim_scores'}
        if 'tsne' in summary:
            summary['tsne'] = {k: v for k, v in summary['tsne'].items() 
                              if k not in ['tsne_embeddings', 'labels']}
        
        with open(results_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"✓ Results saved to {results_path}")
    
    def create_visualizations(self):
        """Create visualization plots."""
        import matplotlib.pyplot as plt
        
        # t-SNE visualization if computed
        if self.compute_tsne and 'tsne' in self.results:
            logger.info("Creating t-SNE visualization...")
            
            embeddings = np.array(self.results['tsne']['tsne_embeddings'])
            labels = np.array(self.results['tsne']['labels'])
            
            plt.figure(figsize=(12, 10))
            plt.scatter(
                embeddings[labels == 0, 0],
                embeddings[labels == 0, 1],
                c='blue',
                alpha=0.4,
                label=f'Real {self.label} (n={len(self.real_images)})',
                s=30,
                edgecolors='none'
            )
            plt.scatter(
                embeddings[labels == 1, 0],
                embeddings[labels == 1, 1],
                c='red',
                alpha=0.4,
                label=f'Generated {self.label} (n={len(self.generated_images)})',
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
    
    def print_summary(self):
        """Print evaluation summary."""
        logger.info("=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Label: {self.label}")
        logger.info(f"Generated images: {self.results['config']['num_generated']}")
        logger.info(f"Real images: {self.results['config']['num_real']}")
        logger.info("")
        
        if 'novelty' in self.results:
            logger.info("Novelty:")
            logger.info(f"  P99 SSIM: {self.results['novelty']['p99_similarity']:.4f} (lower = more novel)")
        
        if 'pathology' in self.results:
            logger.info("Pathology Confidence:")
            logger.info(f"  Mean: {self.results['pathology']['mean_confidence']:.3f} (higher = better)")
        
        if 'biovil' in self.results:
            logger.info("BioViL:")
            logger.info(f"  Mean: {self.results['biovil']['mean_score']:.3f} (higher = better)")
        
        if 'diversity' in self.results:
            logger.info("Diversity:")
            logger.info(f"  Overall: {self.results['diversity']['overall_diversity']:.4f} (higher = more diverse)")
        
        if 'self_similarity' in self.results:
            logger.info("Self-Similarity:")
            logger.info(f"  Mean: {self.results['self_similarity']['mean_self_ssim']:.4f} (lower = more diverse)")
        
        if 'fmd' in self.results:
            logger.info("FMD:")
            logger.info(f"  Score: {self.results['fmd']['score']:.2f} (lower = better match to real)")
        
        if 'tsne' in self.results:
            logger.info("t-SNE:")
            logger.info(f"  Overlap: {self.results['tsne']['overlap_score']:.3f} (higher = better)")
        
        logger.info("=" * 80)
