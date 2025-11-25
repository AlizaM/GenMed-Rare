"""Evaluation module for model inference and metrics computation."""

from src.eval.evaluator import ModelEvaluator, evaluate_checkpoint
from src.eval.diffusion_evaluator import DiffusionGenerationEvaluator
from src.eval.metrics import (
    # Novelty metrics
    compute_novelty_metrics,
    compute_ssim,
    compute_correlation,
    # BioViL (medical-aware text-image alignment)
    load_biovil_model,
    compute_biovil_scores,
    compute_score_metrics,
    # TorchXRayVision metrics
    load_torchxrayvision_model,
    compute_pathology_confidence,
    compute_diversity_metrics,
    # Diversity analysis
    compute_intra_class_variance,
    compute_feature_dispersion,
    compute_self_similarity,
    # Distribution metrics
    compute_fmd,
    compute_tsne_overlap,
)

__all__ = [
    # Classifier evaluation
    'ModelEvaluator',
    'evaluate_checkpoint',
    # Diffusion evaluation
    'DiffusionGenerationEvaluator',
    # Metrics
    'compute_novelty_metrics',
    'compute_ssim',
    'compute_correlation',
    'load_biovil_model',
    'compute_biovil_scores',
    'compute_score_metrics',
    'load_torchxrayvision_model',
    'compute_pathology_confidence',
    'compute_diversity_metrics',
    'compute_intra_class_variance',
    'compute_feature_dispersion',
    'compute_self_similarity',
    'compute_fmd',
    'compute_tsne_overlap',
]
