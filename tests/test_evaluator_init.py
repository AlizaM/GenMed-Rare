#!/usr/bin/env python3
"""Test the evaluator initialization without running full inference."""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config
from src.eval.evaluator import ModelEvaluator

def test_evaluator_init():
    """Test that the evaluator can be initialized properly."""
    print("Testing ModelEvaluator initialization...")
    print("=" * 50)
    
    # Load config
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    
    # Use a dummy checkpoint path for testing
    checkpoint_path = "outputs/effusion_vs_fibrosis_baseline/checkpoints/effusion_vs_fibrosis_50_epochs_baseline/checkpoint_21.pth"
    
    print(f"Config loaded: ✓")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Checkpoint exists: {Path(checkpoint_path).exists()}")
    
    if not Path(checkpoint_path).exists():
        print("\n❌ Cannot test full initialization - checkpoint not found")
        print("This is expected if you haven't trained a model yet.")
        print("\nTo test evaluation with a real checkpoint:")
        print("1. Train a model first")
        print("2. Run: python scripts/evaluate_model.py --config configs/config.yaml --checkpoint path/to/checkpoint.pth")
        return False
    
    try:
        # Try to initialize evaluator (this will load the model)
        evaluator = ModelEvaluator(
            config=config,
            checkpoint_path=checkpoint_path,
            device='cpu'  # Use CPU for testing
        )
        
        print(f"✓ ModelEvaluator initialized successfully")
        print(f"✓ Model loaded from checkpoint")
        print(f"✓ Test data loader created")
        print(f"✓ Output directory: {evaluator.output_dir}")
        print(f"✓ Test samples: {len(evaluator.test_loader.dataset)}")
        
        # Test that we can access key components
        assert evaluator.model is not None
        assert evaluator.test_loader is not None
        assert evaluator.output_dir.exists()
        
        print("\n" + "=" * 50)
        print("✓ All initialization tests passed!")
        print("✓ Evaluator is ready to run inference")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluator_init()
    if not success:
        sys.exit(1)