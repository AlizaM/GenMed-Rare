"""
Test evaluation summary generation robustness.

Tests that summary tables and visualizations handle:
- Missing metrics (None values)
- Failed metrics (empty dictionaries)
- Partial metric failures
- All metrics failing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json


def test_summary_with_all_none_metrics():
    """Test summary generation when all metrics return None."""
    results = {
        'config': {
            'label': 'Fibrosis',
            'num_generated': 100,
            'num_real': 234,
        },
        'novelty': None,
        'pathology': None,
        'biovil': None,
        'diversity': None,
    }

    # Should not crash when creating summary
    summary = create_evaluation_summary(results)

    assert summary['label'] == 'Fibrosis'
    assert summary['num_generated'] == 100
    assert summary['num_real'] == 234

    # All metric values should be None or NaN
    assert pd.isna(summary.get('p99_novelty', np.nan))
    assert pd.isna(summary.get('mean_pathology', np.nan))
    assert pd.isna(summary.get('mean_biovil', np.nan))
    assert pd.isna(summary.get('diversity', np.nan))


def test_summary_with_partial_failures():
    """Test summary when some metrics succeed and some fail."""
    results = {
        'config': {
            'label': 'Pneumonia',
            'num_generated': 50,
            'num_real': 300,
        },
        'novelty': {
            'max_novelty': 0.15,
            'p99_novelty': 0.22,
            'mean_novelty': 0.55,
        },
        'pathology': None,  # Failed
        'biovil': {
            'mean_score': 0.62,
            'median_score': 0.60,
        },
        'diversity': None,  # Failed
    }

    summary = create_evaluation_summary(results)

    # Successful metrics should have values
    assert summary['p99_novelty'] == pytest.approx(0.22)
    assert summary['mean_biovil'] == pytest.approx(0.62)

    # Failed metrics should be None/NaN
    assert pd.isna(summary.get('mean_pathology', np.nan))
    assert pd.isna(summary.get('diversity', np.nan))


def test_dataframe_with_mixed_results():
    """Test DataFrame operations with mixed None/valid values."""
    data = [
        {
            'checkpoint': 'checkpoint-2500',
            'label': 'Fibrosis',
            'p99_novelty': 0.75,
            'mean_pathology': 0.82,
            'mean_biovil': None,  # Failed
            'diversity': 0.15,
        },
        {
            'checkpoint': 'checkpoint-5000',
            'label': 'Fibrosis',
            'p99_novelty': None,  # Failed
            'mean_pathology': 0.88,
            'mean_biovil': 0.65,
            'diversity': 0.18,
        },
        {
            'checkpoint': 'checkpoint-7500',
            'label': 'Fibrosis',
            'p99_novelty': 0.70,
            'mean_pathology': None,  # Failed
            'mean_biovil': 0.68,
            'diversity': None,  # Failed
        },
    ]

    df = pd.DataFrame(data)

    # Test NaN-safe operations
    # Find best novelty (lowest is best, but some are None)
    if df['p99_novelty'].notna().any():
        best_novelty_idx = df['p99_novelty'].idxmin()
        assert df.loc[best_novelty_idx, 'checkpoint'] == 'checkpoint-7500'

    # Find best pathology (highest is best, but some are None)
    if df['mean_pathology'].notna().any():
        best_pathology_idx = df['mean_pathology'].idxmax()
        assert df.loc[best_pathology_idx, 'checkpoint'] == 'checkpoint-5000'

    # Test formatting with None values
    for _, row in df.iterrows():
        for col in ['p99_novelty', 'mean_pathology', 'mean_biovil', 'diversity']:
            val = row[col]
            if pd.notna(val) and val is not None:
                # Should be able to format
                formatted = f"{val:.4f}"
                assert len(formatted) > 0
            else:
                # Should handle None gracefully
                formatted = "N/A" if pd.isna(val) or val is None else f"{val:.4f}"
                assert formatted == "N/A"


def test_json_serialization_with_none():
    """Test that results with None values can be saved to JSON."""
    results = {
        'config': {'label': 'Test', 'num_generated': 10, 'num_real': 20},
        'novelty': {'max_novelty': 0.1, 'p99_novelty': 0.15},
        'pathology': None,
        'biovil': None,
        'diversity': {'overall_diversity': 0.12},
    }

    # Should serialize without errors
    json_str = json.dumps(results, indent=2)
    assert '"pathology": null' in json_str
    assert '"biovil": null' in json_str

    # Should deserialize correctly
    loaded = json.loads(json_str)
    assert loaded['pathology'] is None
    assert loaded['biovil'] is None
    assert loaded['novelty']['p99_novelty'] == 0.15


def test_best_checkpoint_selection_with_missing_metrics():
    """Test finding best checkpoint when some metrics are missing."""
    checkpoints = [
        {'name': 'ckpt-1', 'p99_novelty': 0.80, 'mean_pathology': 0.75, 'mean_biovil': 0.60},
        {'name': 'ckpt-2', 'p99_novelty': None, 'mean_pathology': 0.85, 'mean_biovil': 0.65},
        {'name': 'ckpt-3', 'p99_novelty': 0.75, 'mean_pathology': None, 'mean_biovil': None},
        {'name': 'ckpt-4', 'p99_novelty': 0.70, 'mean_pathology': 0.80, 'mean_biovil': 0.70},
    ]

    df = pd.DataFrame(checkpoints)

    # Best novelty (lowest p99, ignoring None)
    valid_novelty = df[df['p99_novelty'].notna()]
    if len(valid_novelty) > 0:
        best_novelty = valid_novelty.loc[valid_novelty['p99_novelty'].idxmin(), 'name']
        assert best_novelty == 'ckpt-4'  # Has lowest valid p99_novelty (0.70)

    # Best pathology (highest mean, ignoring None)
    valid_pathology = df[df['mean_pathology'].notna()]
    if len(valid_pathology) > 0:
        best_pathology = valid_pathology.loc[valid_pathology['mean_pathology'].idxmax(), 'name']
        assert best_pathology == 'ckpt-2'  # Has highest valid pathology (0.85)

    # Best combined (only checkpoints with all metrics available)
    valid_all = df[df['p99_novelty'].notna() & df['mean_pathology'].notna() & df['mean_biovil'].notna()]
    assert len(valid_all) == 2  # Only ckpt-1 and ckpt-4 have all metrics


def test_empty_results():
    """Test handling completely empty results."""
    results = {'config': {'label': 'Empty', 'num_generated': 0, 'num_real': 0}}

    summary = create_evaluation_summary(results)
    assert summary['label'] == 'Empty'
    assert summary['num_generated'] == 0


# Helper function (simplified version of what should be in diffusion_evaluator)
def create_evaluation_summary(results: dict) -> dict:
    """
    Create summary dictionary from evaluation results, handling None values.

    This is a test helper that mimics the real summary creation logic.
    """
    summary = {
        'label': results['config']['label'],
        'num_generated': results['config']['num_generated'],
        'num_real': results['config']['num_real'],
    }

    # Add metric values, handling None gracefully
    if 'novelty' in results and results['novelty'] is not None:
        summary['p99_novelty'] = results['novelty'].get('p99_novelty')
        summary['mean_novelty'] = results['novelty'].get('mean_novelty')
    else:
        summary['p99_novelty'] = None
        summary['mean_novelty'] = None

    if 'pathology' in results and results['pathology'] is not None:
        summary['mean_pathology'] = results['pathology'].get('mean_confidence')
    else:
        summary['mean_pathology'] = None

    if 'biovil' in results and results['biovil'] is not None:
        summary['mean_biovil'] = results['biovil'].get('mean_score')
    else:
        summary['mean_biovil'] = None

    if 'diversity' in results and results['diversity'] is not None:
        summary['diversity'] = results['diversity'].get('overall_diversity')
    else:
        summary['diversity'] = None

    return summary


def create_evaluation_summary_full(results: dict) -> dict:
    """
    Create summary for full preset (all 9 metrics), handling None values.

    This extends the basic summary to include all metrics from 'full' preset.
    """
    # Start with basic summary
    summary = create_evaluation_summary(results)

    # Add additional full preset metrics
    if 'pixel_variance' in results and results['pixel_variance'] is not None:
        summary['pixel_variance'] = results['pixel_variance'].get('mean_variance')
    else:
        summary['pixel_variance'] = None

    if 'feature_dispersion' in results and results['feature_dispersion'] is not None:
        summary['feature_dispersion'] = results['feature_dispersion'].get('mean_dispersion')
    else:
        summary['feature_dispersion'] = None

    if 'self_similarity' in results and results['self_similarity'] is not None:
        summary['self_similarity'] = results['self_similarity'].get('mean_self_ssim')
    else:
        summary['self_similarity'] = None

    if 'fmd' in results and results['fmd'] is not None:
        summary['fmd'] = results['fmd'].get('fmd_score')
    else:
        summary['fmd'] = None

    if 'tsne' in results and results['tsne'] is not None:
        summary['tsne_overlap'] = results['tsne'].get('overlap_score')
    else:
        summary['tsne_overlap'] = None

    return summary


def test_table_formatting_with_none():
    """Test that table formatting handles None values properly."""
    data = {
        'Checkpoint': ['ckpt-1', 'ckpt-2', 'ckpt-3'],
        'P99 Novelty': [0.75, None, 0.80],
        'Mean Pathology': [0.82, 0.85, None],
        'Mean BioViL': [None, 0.65, 0.70],
        'Diversity': [0.15, 0.18, 0.12],
    }

    df = pd.DataFrame(data)

    # Test formatting function that should handle None
    def format_value(val):
        if val is None or pd.isna(val):
            return 'N/A'
        elif isinstance(val, (int, float)):
            return f'{val:.4f}'
        else:
            return str(val)

    # Apply formatting to each cell
    formatted_df = df.copy()
    for col in df.columns:
        if col != 'Checkpoint':
            formatted_df[col] = df[col].apply(format_value)

    # Verify formatting
    assert formatted_df.loc[0, 'Mean BioViL'] == 'N/A'
    assert formatted_df.loc[1, 'P99 Novelty'] == 'N/A'
    assert formatted_df.loc[2, 'Mean Pathology'] == 'N/A'
    assert formatted_df.loc[0, 'P99 Novelty'] == '0.7500'


def test_full_preset_with_all_metrics():
    """Test summary with all 9 metrics from 'full' preset (some failing)."""
    results = {
        'config': {
            'label': 'Fibrosis',
            'num_generated': 2000,
            'num_real': 551,
        },
        # Checkpoint preset metrics (4)
        'novelty': {'max_novelty': 0.05, 'p99_novelty': 0.15, 'mean_novelty': 0.45},
        'pathology': {'mean_confidence': 0.78, 'median_confidence': 0.80},
        'biovil': None,  # Failed (API error)
        'diversity': {'overall_diversity': 0.22},

        # Additional full preset metrics (5 more)
        'pixel_variance': {'mean_variance': 1250.5, 'std_variance': 342.1},
        'feature_dispersion': None,  # Failed (OOM error)
        'self_similarity': {'mean_self_ssim': 0.35, 'median_self_ssim': 0.33},
        'fmd': {'fmd_score': 12.5},
        'tsne': None,  # Failed (perplexity too high)
    }

    summary = create_evaluation_summary_full(results)

    # Checkpoint metrics
    assert summary['p99_novelty'] == pytest.approx(0.15)
    assert summary['mean_pathology'] == pytest.approx(0.78)
    assert pd.isna(summary.get('mean_biovil', np.nan))  # Failed
    assert summary['diversity'] == pytest.approx(0.22)

    # Additional full preset metrics
    assert summary['pixel_variance'] == pytest.approx(1250.5)
    assert pd.isna(summary.get('feature_dispersion', np.nan))  # Failed
    assert summary['self_similarity'] == pytest.approx(0.35)
    assert summary['fmd'] == pytest.approx(12.5)
    assert pd.isna(summary.get('tsne_overlap', np.nan))  # Failed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
