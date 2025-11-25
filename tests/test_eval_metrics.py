"""
Tests for evaluation metrics.

These tests run on CPU and validate core metric functionality including:
- Dimension handling (different image sizes)
- Grayscale vs RGB handling
- Edge cases (empty arrays, single images, etc.)

Run with: pytest tests/test_eval_metrics.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eval.metrics import (
    compute_ssim,
    compute_correlation,
    compute_novelty_metrics,
    compute_pathology_confidence,
    compute_diversity_metrics,
    compute_intra_class_variance,
    compute_feature_dispersion,
    compute_self_similarity,
)


class TestSSIM:
    """Test SSIM computation with various image dimensions."""
    
    def test_same_size_rgb(self):
        """Test SSIM with same-size RGB images."""
        img1 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        score = compute_ssim(img1, img2)
        assert 0.0 <= score <= 1.0, "SSIM should be in [0, 1]"
    
    def test_same_size_grayscale(self):
        """Test SSIM with same-size grayscale images."""
        img1 = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        
        score = compute_ssim(img1, img2)
        assert 0.0 <= score <= 1.0, "SSIM should be in [0, 1]"
    
    def test_different_sizes_rgb(self):
        """Test SSIM with different-size RGB images (should auto-resize)."""
        img1 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        
        score = compute_ssim(img1, img2)
        assert 0.0 <= score <= 1.0, "SSIM should handle different sizes"
    
    def test_different_sizes_grayscale(self):
        """Test SSIM with different-size grayscale images."""
        img1 = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
        
        score = compute_ssim(img1, img2)
        assert 0.0 <= score <= 1.0, "SSIM should handle different sizes"
    
    def test_mixed_rgb_grayscale(self):
        """Test SSIM with RGB vs grayscale."""
        img1_rgb = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        img2_gray = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        
        score = compute_ssim(img1_rgb, img2_gray)
        assert 0.0 <= score <= 1.0, "SSIM should handle RGB vs grayscale"
    
    def test_identical_images(self):
        """Test SSIM with identical images (should be ~1.0)."""
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        score = compute_ssim(img, img.copy())
        assert score > 0.99, f"Identical images should have SSIM ~1.0, got {score}"
    
    def test_very_different_sizes(self):
        """Test SSIM with very different sizes (e.g., 256 vs 1024)."""
        img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (2048, 2048, 3), dtype=np.uint8)
        
        score = compute_ssim(img1, img2)
        assert 0.0 <= score <= 1.0, "SSIM should handle large size differences"


class TestCorrelation:
    """Test correlation computation with various image dimensions."""
    
    def test_same_size(self):
        """Test correlation with same-size images."""
        img1 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        corr = compute_correlation(img1, img2)
        assert -1.0 <= corr <= 1.0, "Correlation should be in [-1, 1]"
    
    def test_different_sizes(self):
        """Test correlation with different-size images (should auto-resize)."""
        img1 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        
        corr = compute_correlation(img1, img2)
        assert -1.0 <= corr <= 1.0, "Correlation should handle different sizes"
    
    def test_identical_images(self):
        """Test correlation with identical images (should be ~1.0)."""
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        corr = compute_correlation(img, img.copy())
        assert corr > 0.99, f"Identical images should have correlation ~1.0, got {corr}"


class TestNoveltyMetrics:
    """Test novelty metrics computation."""
    
    def test_same_size_images(self):
        """Test novelty with same-size images."""
        generated = [np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(10)]
        training = [np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(20)]
        
        metrics = compute_novelty_metrics(generated, training, metric='ssim', show_progress=False)
        
        assert 'max_similarity' in metrics
        assert 'mean_similarity' in metrics
        assert 'p99_similarity' in metrics
        assert 0.0 <= metrics['max_similarity'] <= 1.0
    
    def test_different_size_images(self):
        """Test novelty with different-size images (common real-world scenario)."""
        generated = [np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(10)]
        training = [np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8) for _ in range(20)]
        
        # This should not raise an error
        metrics = compute_novelty_metrics(generated, training, metric='ssim', show_progress=False)
        
        assert 'max_similarity' in metrics
        assert 0.0 <= metrics['max_similarity'] <= 1.0
    
    def test_correlation_metric(self):
        """Test novelty using correlation instead of SSIM."""
        generated = [np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(5)]
        training = [np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(10)]
        
        metrics = compute_novelty_metrics(generated, training, metric='correlation', show_progress=False)
        
        assert 'max_similarity' in metrics
        assert -1.0 <= metrics['max_similarity'] <= 1.0


class TestDiversityMetrics:
    """Test diversity metrics (CPU-friendly, no model loading)."""
    
    def test_intra_class_variance(self):
        """Test pixel-level variance computation."""
        images = [np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(10)]
        
        metrics = compute_intra_class_variance(images)
        
        assert 'mean_pixel_variance' in metrics
        assert 'variance_per_channel' in metrics
        assert 'min_pixel_variance' in metrics
        assert 'max_pixel_variance' in metrics
        assert metrics['mean_pixel_variance'] >= 0
    
    def test_intra_class_variance_grayscale(self):
        """Test variance with grayscale images."""
        images = [np.random.randint(0, 256, (512, 512), dtype=np.uint8) for _ in range(10)]
        
        metrics = compute_intra_class_variance(images)
        
        assert 'mean_pixel_variance' in metrics
        assert metrics['mean_pixel_variance'] >= 0
    
    def test_self_similarity(self):
        """Test pairwise SSIM computation."""
        images = [np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(10)]
        
        metrics = compute_self_similarity(images, num_samples=5, show_progress=False)
        
        assert 'mean_self_ssim' in metrics
        assert 'median_self_ssim' in metrics
        assert 'std_self_ssim' in metrics
        assert 0.0 <= metrics['mean_self_ssim'] <= 1.0
    
    def test_self_similarity_different_sizes(self):
        """Test self-similarity with varied image sizes."""
        images = [
            np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8),
            np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
            np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8),
        ]
        
        # Should handle different sizes without error
        metrics = compute_self_similarity(images, num_samples=2, show_progress=False)
        
        assert 'mean_self_ssim' in metrics
        assert 0.0 <= metrics['mean_self_ssim'] <= 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_image_novelty(self):
        """Test novelty with single generated image."""
        generated = [np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)]
        training = [np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(10)]
        
        metrics = compute_novelty_metrics(generated, training, metric='ssim', show_progress=False)
        
        assert 'max_similarity' in metrics
        assert len(metrics['nn_indices']) == 1
    
    def test_very_small_images(self):
        """Test with very small images (edge case)."""
        # Note: SSIM can be negative for very small images due to edge effects
        # We just verify it computes without error
        img1 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        score = compute_ssim(img1, img2)
        assert isinstance(score, (float, np.floating))
    
    def test_single_channel_vs_multichannel(self):
        """Test handling of single-channel vs multi-channel."""
        img1 = np.random.randint(0, 256, (512, 512, 1), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        # Should handle without error
        score = compute_ssim(img1, img2)
        assert 0.0 <= score <= 1.0


class TestDataTypes:
    """Test different data types and ranges."""
    
    def test_float_images(self):
        """Test with float images [0, 1]."""
        img1 = np.random.rand(512, 512, 3).astype(np.float32) * 255
        img2 = np.random.rand(512, 512, 3).astype(np.float32) * 255
        
        score = compute_ssim(img1, img2)
        assert 0.0 <= score <= 1.0
    
    def test_uint8_images(self):
        """Test with uint8 images [0, 255]."""
        img1 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        score = compute_ssim(img1, img2)
        assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
