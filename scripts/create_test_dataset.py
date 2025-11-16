#!/usr/bin/env python3
"""Create a small test dataset for quick training validation."""
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def create_test_dataset(
    input_csv: str = "data/processed/effusion_fibrosis/dataset.csv",
    output_csv: str = "data/processed/effusion_fibrosis/dataset_test.csv",
    train_samples: int = 100,
    val_samples: int = 30,
    test_samples: int = 30
):
    """
    Create a small test dataset by sampling from the full dataset.
    
    Args:
        input_csv: Path to full dataset CSV
        output_csv: Path to save test dataset CSV
        train_samples: Number of training samples per class
        val_samples: Number of validation samples per class
        test_samples: Number of test samples per class
    """
    print("=" * 80)
    print("Creating Test Dataset")
    print("=" * 80)
    
    # Load full dataset
    df = pd.read_csv(input_csv)
    print(f"\nFull dataset: {len(df)} images")
    print(f"  Train: {(df['split'] == 'train').sum()}")
    print(f"  Val: {(df['split'] == 'val').sum()}")
    print(f"  Test: {(df['split'] == 'test').sum()}")
    
    test_dfs = []
    
    # Sample from each split, maintaining class balance
    for split_name, n_samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        df_split = df[df['split'] == split_name]
        
        # Sample equally from each class
        df_class_0 = df_split[df_split['label'] == 0].sample(n=n_samples, random_state=42)
        df_class_1 = df_split[df_split['label'] == 1].sample(n=n_samples, random_state=42)
        
        test_dfs.append(pd.concat([df_class_0, df_class_1]))
        
        print(f"\n{split_name.upper()}: Sampled {len(df_class_0) + len(df_class_1)} images")
        print(f"  Class 0 (Effusion): {len(df_class_0)}")
        print(f"  Class 1 (Fibrosis): {len(df_class_1)}")
    
    # Combine all splits
    df_test = pd.concat(test_dfs, ignore_index=True)
    
    # Save test dataset
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print(f"Test dataset saved: {output_path}")
    print(f"Total images: {len(df_test)}")
    print(f"  Train: {(df_test['split'] == 'train').sum()}")
    print(f"  Val: {(df_test['split'] == 'val').sum()}")
    print(f"  Test: {(df_test['split'] == 'test').sum()}")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create test dataset')
    parser.add_argument('--input', type=str, default='data/processed/effusion_fibrosis/dataset.csv')
    parser.add_argument('--output', type=str, default='data/processed/effusion_fibrosis/dataset_test.csv')
    parser.add_argument('--train-samples', type=int, default=100, help='Samples per class for train')
    parser.add_argument('--val-samples', type=int, default=30, help='Samples per class for val')
    parser.add_argument('--test-samples', type=int, default=30, help='Samples per class for test')
    
    args = parser.parse_args()
    
    create_test_dataset(
        args.input,
        args.output,
        args.train_samples,
        args.val_samples,
        args.test_samples
    )
