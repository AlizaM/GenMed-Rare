"""
Test checkpoint loading locally on CPU.

This verifies the checkpoint loading logic works without needing GPU.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.diffusion_utils import load_pipeline

def test_checkpoint_loading(checkpoint_path):
    """Test loading checkpoint on CPU."""
    
    print("=" * 80)
    print("Testing checkpoint loading on CPU")
    print("=" * 80)
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Testing with CPU device...")
    
    try:
        # Load pipeline on CPU
        pipeline = load_pipeline(
            checkpoint_path=str(checkpoint_path),
            pretrained_model="danyalmalik/stable-diffusion-chest-xray",
            enable_attention_slicing=True,
            enable_vae_slicing=True,
            device="cpu"  # Force CPU for testing
        )
        
        print("\n" + "=" * 80)
        print("✅ SUCCESS: Checkpoint loaded successfully!")
        print("=" * 80)
        
        # Print model info
        print(f"\nModel type: {type(pipeline.unet)}")
        print(f"Number of parameters: {sum(p.numel() for p in pipeline.unet.parameters()):,}")
        
        # Check if it's a PEFT model
        if hasattr(pipeline.unet, 'print_trainable_parameters'):
            print("\nPEFT model detected!")
            pipeline.unet.print_trainable_parameters()
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ FAILED: Error loading checkpoint")
        print("=" * 80)
        print(f"\nError: {e}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_checkpoint_loading_local.py <checkpoint_path>")
        print("\nExample:")
        print("  python scripts/test_checkpoint_loading_local.py outputs/checkpoint-6500")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    success = test_checkpoint_loading(checkpoint_path)
    
    sys.exit(0 if success else 1)
