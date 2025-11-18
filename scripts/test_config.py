#!/usr/bin/env python3
"""Test the new auto-generation configuration features."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config

def test_config_auto_generation():
    """Test that paths are auto-generated correctly."""
    print("🧪 Testing Config Auto-Generation")
    print("=" * 50)
    
    # Load config
    config = load_config("configs/config.yaml")
    
    print(f"📋 Classes:")
    print(f"  Positive (rare):  {config.data.class_positive}")
    print(f"  Negative (common): {config.data.class_negative}")
    print()
    
    print(f"🎯 Auto-generated paths:")
    print(f"  Experiment name:   {config.experiment.name}")
    print(f"  Processed dir:     {config.data.processed_dir}")
    print(f"  Checkpoint dir:    {config.training.checkpoint_dir}")
    print(f"  Log dir:           {config.training.log_dir}")
    print()
    
    print(f"📁 Expected directory structure:")
    print(f"  outputs/")
    print(f"  ├── {config.experiment.name}/")
    print(f"  │   ├── checkpoints/{config.experiment.name}/")
    print(f"  │   ├── logs/{config.experiment.name}/")
    print(f"  │   ├── evaluation/")
    print(f"  │   ├── config.yaml (copied)")
    print(f"  │   └── dataset.csv (copied)")
    print(f"  └── data/processed/{config.data.processed_dir.name}/")
    print()
    
    # Test creating directories
    try:
        checkpoint_dir, log_dir = config.create_dirs()
        print(f"✅ Directories created successfully!")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Log dir: {log_dir}")
        print(f"  Experiment dir: outputs/{config.experiment.name}")
        
        # Check if files were copied
        experiment_dir = Path("outputs") / config.experiment.name
        config_copy = experiment_dir / "config.yaml"
        dataset_copy = experiment_dir / "dataset.csv"
        
        if config_copy.exists():
            print(f"  ✅ Configuration copied to experiment folder")
        if dataset_copy.exists():
            print(f"  ✅ Dataset copied to experiment folder")
        elif Path(config.data.processed_dir / "dataset.csv").exists():
            print(f"  ℹ️  Dataset exists but not yet copied (will copy during training)")
        else:
            print(f"  ⚠️  Dataset not found - you may need to run data preprocessing first")
            
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
    
    print()
    print("🔄 To switch classes, just edit configs/config.yaml:")
    print("   class_positive: 'Fibrosis'    # Change this")  
    print("   class_negative: 'Effusion'    # And this")
    print("   Everything else updates automatically!")

if __name__ == "__main__":
    test_config_auto_generation()