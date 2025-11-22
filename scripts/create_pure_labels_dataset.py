#!/usr/bin/env python3
"""
Create Pure Labels Dataset

This script extracts images with only pure Fibrosis and pure Effusion labels
from the raw NIH Chest X-ray dataset for binary classification experiments.

Pure labels = single pathology only (no mixed labels with '|')

Usage:
    python scripts/create_pure_labels_dataset.py [--target-classes fibrosis,effusion] [--output-dir path]
"""

import os
import sys
import pandas as pd
import shutil
import argparse
from pathlib import Path
from collections import Counter
from zipfile import ZipFile


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create dataset with only pure labels")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/raw/Data_Entry_2017.csv",
        help="Path to input CSV file (default: data/raw/Data_Entry_2017.csv)"
    )
    parser.add_argument(
        "--input-archive",
        type=str,
        default="data/raw/archive.zip",
        help="Path to input archive with images (default: data/raw/archive.zip)"
    )
    parser.add_argument(
        "--target-classes",
        type=str,
        default="Fibrosis,Effusion",
        help="Comma-separated list of target pure classes (default: Fibrosis,Effusion)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pure_labels",
        help="Output directory for pure labels dataset (default: data/pure_labels)"
    )
    parser.add_argument(
        "--train-val-list",
        type=str,
        default="data/raw/train_val_list.txt",
        help="Path to train/val split file (default: data/raw/train_val_list.txt)"
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default="data/raw/test_list.txt",
        help="Path to test split file (default: data/raw/test_list.txt)"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy actual image files (requires more space and time)"
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum number of images per class (for balanced datasets)"
    )
    
    return parser.parse_args()


def load_split_files(train_val_path, test_path):
    """Load train/val and test split files."""
    train_val_images = set()
    test_images = set()
    
    if Path(train_val_path).exists():
        with open(train_val_path, 'r') as f:
            train_val_images = set(line.strip() for line in f if line.strip())
        print(f"✓ Loaded {len(train_val_images):,} train/val images")
    else:
        print(f"⚠️ Train/val list not found: {train_val_path}")
    
    if Path(test_path).exists():
        with open(test_path, 'r') as f:
            test_images = set(line.strip() for line in f if line.strip())
        print(f"✓ Loaded {len(test_images):,} test images")
    else:
        print(f"⚠️ Test list not found: {test_path}")
    
    return train_val_images, test_images


def analyze_pure_labels(df, target_classes):
    """Analyze and filter for pure labels."""
    print("\n" + "="*60)
    print("ANALYZING PURE LABELS")
    print("="*60)
    
    # Filter for pure labels only (no '|' in Finding Labels)
    pure_labels = df[~df['Finding Labels'].str.contains(r'\|', na=False)]
    print(f"Total images with pure labels: {len(pure_labels):,}")
    
    # Count pure target classes
    target_counts = {}
    target_dfs = {}
    
    for class_name in target_classes:
        class_df = pure_labels[pure_labels['Finding Labels'] == class_name]
        target_counts[class_name] = len(class_df)
        target_dfs[class_name] = class_df
        print(f"Pure {class_name}: {len(class_df):,} images")
    
    return target_dfs, target_counts


def balance_classes(target_dfs, max_per_class):
    """Balance classes to have equal number of samples."""
    if max_per_class is None:
        return target_dfs
    
    print(f"\n🔧 Balancing classes to max {max_per_class:,} samples each...")
    
    balanced_dfs = {}
    for class_name, df in target_dfs.items():
        if len(df) > max_per_class:
            # Sample randomly
            balanced_df = df.sample(n=max_per_class, random_state=42)
            print(f"  {class_name}: {len(df):,} → {len(balanced_df):,} (sampled)")
        else:
            balanced_df = df.copy()
            print(f"  {class_name}: {len(df):,} (unchanged)")
        
        balanced_dfs[class_name] = balanced_df
    
    return balanced_dfs


def create_split_datasets(target_dfs, train_val_images, test_images):
    """Split datasets into train/val and test based on official splits."""
    print("\n" + "="*60)
    print("CREATING TRAIN/VAL AND TEST SPLITS")
    print("="*60)
    
    train_val_data = []
    test_data = []
    
    for class_name, df in target_dfs.items():
        class_train_val = df[df['Image Index'].isin(train_val_images)]
        class_test = df[df['Image Index'].isin(test_images)]
        
        print(f"\n{class_name}:")
        print(f"  Train/Val: {len(class_train_val):,} images")
        print(f"  Test: {len(class_test):,} images")
        print(f"  Total: {len(df):,} images")
        
        # Add split column
        class_train_val = class_train_val.copy()
        class_test = class_test.copy()
        class_train_val['Split'] = 'train_val'
        class_test['Split'] = 'test'
        
        train_val_data.append(class_train_val)
        test_data.append(class_test)
    
    # Combine all classes
    train_val_df = pd.concat(train_val_data, ignore_index=True) if train_val_data else pd.DataFrame()
    test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
    
    return train_val_df, test_df


def copy_images_from_archive(image_list, archive_path, output_dir):
    """Copy images from ZIP archive to output directory."""
    if not Path(archive_path).exists():
        print(f"⚠️ Archive not found: {archive_path}")
        return False
    
    print(f"\n📦 Extracting {len(image_list)} images from archive...")
    
    with ZipFile(archive_path, 'r') as zip_ref:
        # Get list of files in archive
        archive_files = set(zip_ref.namelist())
        
        copied_count = 0
        for img_name in image_list:
            # Look for the image in different possible paths within the archive
            possible_paths = [
                img_name,
                f"images/{img_name}",
                f"images_001/images/{img_name}",
                f"images_002/images/{img_name}",
                f"images_003/images/{img_name}",
                f"images_004/images/{img_name}",
                f"images_005/images/{img_name}",
                f"images_006/images/{img_name}",
                f"images_007/images/{img_name}",
                f"images_008/images/{img_name}",
                f"images_009/images/{img_name}",
                f"images_010/images/{img_name}",
                f"images_011/images/{img_name}",
                f"images_012/images/{img_name}",
            ]
            
            found = False
            for path in possible_paths:
                if path in archive_files:
                    # Extract to output directory
                    zip_ref.extract(path, output_dir)
                    
                    # Move to root of output directory if in subdirectory
                    extracted_path = Path(output_dir) / path
                    final_path = Path(output_dir) / img_name
                    
                    if extracted_path != final_path:
                        extracted_path.rename(final_path)
                        
                        # Clean up empty directories
                        try:
                            extracted_path.parent.rmdir()
                        except:
                            pass
                    
                    copied_count += 1
                    found = True
                    break
            
            if not found:
                print(f"⚠️ Image not found in archive: {img_name}")
        
        print(f"✓ Copied {copied_count}/{len(image_list)} images")
        return copied_count > 0


def main():
    """Main function."""
    args = parse_args()
    
    # Parse target classes
    target_classes = [cls.strip() for cls in args.target_classes.split(',')]
    
    print("🔬 PURE LABELS DATASET CREATOR")
    print("=" * 60)
    print(f"Input CSV: {args.input_csv}")
    print(f"Target classes: {', '.join(target_classes)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max per class: {args.max_per_class or 'No limit'}")
    print(f"Copy images: {args.copy_images}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CSV data
    print(f"\n📊 Loading data from {args.input_csv}...")
    if not Path(args.input_csv).exists():
        print(f"❌ Input CSV not found: {args.input_csv}")
        return False
    
    df = pd.read_csv(args.input_csv)
    print(f"✓ Loaded {len(df):,} total images")
    
    # Load split files
    train_val_images, test_images = load_split_files(args.train_val_list, args.test_list)
    
    # Analyze and filter for pure labels
    target_dfs, target_counts = analyze_pure_labels(df, target_classes)
    
    if not any(target_counts.values()):
        print("❌ No pure labels found for target classes!")
        return False
    
    # Balance classes if requested
    if args.max_per_class:
        target_dfs = balance_classes(target_dfs, args.max_per_class)
    
    # Create train/val and test splits
    if train_val_images and test_images:
        train_val_df, test_df = create_split_datasets(target_dfs, train_val_images, test_images)
    else:
        # Combine all data if no split files
        combined_df = pd.concat(target_dfs.values(), ignore_index=True)
        train_val_df = combined_df
        test_df = pd.DataFrame()
    
    # Save datasets
    print(f"\n💾 Saving datasets to {output_dir}...")
    
    if not train_val_df.empty:
        train_val_path = output_dir / "pure_labels_train_val.csv"
        train_val_df.to_csv(train_val_path, index=False)
        print(f"✓ Train/val dataset: {train_val_path} ({len(train_val_df):,} images)")
    
    if not test_df.empty:
        test_path = output_dir / "pure_labels_test.csv"
        test_df.to_csv(test_path, index=False)
        print(f"✓ Test dataset: {test_path} ({len(test_df):,} images)")
    
    # Combined dataset
    combined_df = pd.concat([train_val_df, test_df], ignore_index=True) if not test_df.empty else train_val_df
    combined_path = output_dir / "pure_labels_combined.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"✓ Combined dataset: {combined_path} ({len(combined_df):,} images)")
    
    # Copy images if requested
    if args.copy_images:
        all_images = set(combined_df['Image Index'].tolist())
        copy_images_from_archive(all_images, args.input_archive, str(output_dir))
    
    # Print summary
    print(f"\n📈 DATASET SUMMARY")
    print("=" * 60)
    for class_name in target_classes:
        class_count = len(combined_df[combined_df['Finding Labels'] == class_name])
        print(f"{class_name}: {class_count:,} pure images")
    
    print(f"\n✅ Pure labels dataset created successfully!")
    print(f"📁 Output directory: {output_dir}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)