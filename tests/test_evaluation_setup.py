#!/usr/bin/env python3
"""Test script to verify evaluation setup is working correctly."""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.dataset import ChestXrayDataset
import torch

def test_data_loading():
    """Test that test data loading works correctly."""
    print("Testing evaluation data loading setup...")
    print("=" * 50)
    
    # Load config
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    
    print(f"Config loaded from: {config_path}")
    print(f"Processed data directory: {config.data.processed_dir}")
    
    # Test dataset path
    dataset_path = Path(config.data.processed_dir) / "dataset.csv"
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset exists: {dataset_path.exists()}")
    
    if not dataset_path.exists():
        print("ERROR: Dataset file not found!")
        return False
    
    try:
        # Create test dataset
        test_dataset = ChestXrayDataset(
            csv_path=dataset_path,
            config=config,
            split='test'
        )
        
        print(f"Test dataset created successfully!")
        print(f"Number of test samples: {len(test_dataset)}")
        
        # Check class distribution
        labels = [test_dataset.df.iloc[i]['label'] for i in range(len(test_dataset))]
        class_counts = {0: labels.count(0), 1: labels.count(1)}
        
        print(f"Class distribution:")
        print(f"  {config.data.class_negative} (0): {class_counts[0]} samples")
        print(f"  {config.data.class_positive} (1): {class_counts[1]} samples")
        
        # Test loading a sample
        sample_image, sample_label = test_dataset[0]
        print(f"Sample loaded successfully:")
        print(f"  Image shape: {sample_image.shape}")
        print(f"  Label: {sample_label} ({config.data.class_negative if sample_label == 0 else config.data.class_positive})")
        
        # Create dataloader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0  # Use 0 for testing
        )
        
        print(f"DataLoader created successfully!")
        print(f"Number of batches: {len(test_loader)}")
        
        # Test loading a batch
        batch_images, batch_labels = next(iter(test_loader))
        print(f"Batch loaded successfully:")
        print(f"  Batch images shape: {batch_images.shape}")
        print(f"  Batch labels shape: {batch_labels.shape}")
        print(f"  Batch labels: {batch_labels.tolist()}")
        
        print("=" * 50)
        print("✓ All tests passed! Evaluation setup is working correctly.")
        print(f"✓ Test data source: {dataset_path}")
        print(f"✓ Total test samples: {len(test_dataset)}")
        print(f"✓ Data ready for evaluation!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if not success:
        sys.exit(1)