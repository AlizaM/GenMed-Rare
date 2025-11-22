#!/usr/bin/env python3
"""
Create Pure Class Folders

This script creates separate folders for pure Fibrosis, pure Pneumonia, and healthy images
from the NIH Chest X-ray dataset. Images are extracted and organized into class-specific folders.

Usage:
    python scripts/create_pure_class_folders.py [--healthy-count 1200] [--copy-images]
"""

import os
import sys
import pandas as pd
import shutil
import argparse
from pathlib import Path
from collections import Counter
from zipfile import ZipFile
import random


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create folders with pure class images")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/raw/Data_Entry_2017.csv",
        help="Path to input CSV file (default: data/raw/Data_Entry_2017.csv)"
    )
    parser.add_argument(
        "--train-val-list",
        type=str,
        default="data/raw/train_val_list.txt",
        help="Path to train_val_list.txt file (default: data/raw/train_val_list.txt)"
    )
    parser.add_argument(
        "--input-archive",
        type=str,
        default="data/raw/archive.zip",
        help="Path to input archive with images (default: data/raw/archive.zip)"
    )
    parser.add_argument(
        "--interim-dir",
        type=str,
        default="data/interim",
        help="Path to interim directory with organized images (default: data/interim)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pure_class_folders",
        help="Output directory for class folders (default: data/pure_class_folders)"
    )
    parser.add_argument(
        "--healthy-count",
        type=int,
        default=1000,
        help="Number of healthy (No Finding) images to include (default: 1000)"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy actual image files from archive (requires archive.zip)"
    )
    parser.add_argument(
        "--create-csv-only",
        action="store_true",
        help="Only create CSV files, don't copy images"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling healthy images (default: 42)"
    )
    
    return parser.parse_args()


def find_image_in_archive(zip_ref, img_name, archive_files):
    """Find image in various possible paths within the archive."""
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
    
    for path in possible_paths:
        if path in archive_files:
            return path
    return None


def extract_images_for_class(zip_ref, archive_files, image_list, output_folder):
    """Extract images for a specific class to the output folder."""
    output_folder.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    missing_count = 0
    
    for img_name in image_list:
        archive_path = find_image_in_archive(zip_ref, img_name, archive_files)
        
        if archive_path:
            # Extract to temporary location
            zip_ref.extract(archive_path, "/tmp")
            
            # Move to final destination
            temp_path = Path("/tmp") / archive_path
            final_path = output_folder / img_name
            
            shutil.move(str(temp_path), str(final_path))
            
            # Clean up empty directories
            try:
                temp_path.parent.rmdir()
            except:
                pass
            
            copied_count += 1
        else:
            print(f"⚠️ Image not found in archive: {img_name}")
            missing_count += 1
    
    return copied_count, missing_count


def analyze_and_extract_classes(df, train_val_images, healthy_count, seed=42):
    """Analyze data and extract pure classes from train/val set only."""
    print("\n" + "="*60)
    print("ANALYZING AND EXTRACTING PURE CLASSES (TRAIN/VAL ONLY)")
    print("="*60)
    
    # Filter to train/val images only
    original_count = len(df)
    df_filtered = df[df['Image Index'].isin(train_val_images)]
    print(f"Original dataset: {original_count:,} images")
    print(f"Train/val subset: {len(df_filtered):,} images")
    print(f"Test images excluded: {original_count - len(df_filtered):,} images")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Extract pure classes from train/val
    pure_fibrosis = df_filtered[df_filtered['Finding Labels'] == 'Fibrosis']
    pure_pneumonia = df_filtered[df_filtered['Finding Labels'] == 'Pneumonia']
    
    print(f"\n🔍 Pure classes found in train/val:")
    print(f"Pure Fibrosis: {len(pure_fibrosis):,} images")
    print(f"Pure Pneumonia: {len(pure_pneumonia):,} images")
    
    # Get healthy images from diffusion dataset
    print(f"\n📊 Loading healthy images from diffusion dataset...")
    diffusion_csv_path = Path("data/diffusion_data/diffusion_dataset_balanced.csv")
    
    if diffusion_csv_path.exists():
        diffusion_df = pd.read_csv(diffusion_csv_path)
        print(f"Loaded diffusion dataset: {len(diffusion_df):,} images")
        
        # Get healthy images from diffusion dataset
        diffusion_healthy = diffusion_df[diffusion_df['Finding Labels'] == 'No Finding']
        print(f"Healthy images in diffusion dataset: {len(diffusion_healthy):,}")
        
        # Sample the requested number of healthy images
        if len(diffusion_healthy) >= healthy_count:
            pure_healthy_sampled = diffusion_healthy.sample(n=healthy_count, random_state=seed)
            print(f"Healthy images sampled: {len(pure_healthy_sampled):,} (from {len(diffusion_healthy):,} available)")
        else:
            pure_healthy_sampled = diffusion_healthy
            print(f"Using all available healthy images: {len(pure_healthy_sampled):,}")
        
        # Convert diffusion format to match main dataset format (already compatible)
        healthy_for_output = pure_healthy_sampled[['Image Index', 'Finding Labels']].copy()
        
    else:
        print(f"❌ Diffusion dataset not found at: {diffusion_csv_path}")
        print(f"Falling back to train/val healthy images...")
        pure_healthy = df_filtered[df_filtered['Finding Labels'] == 'No Finding']
        print(f"Pure Healthy (No Finding): {len(pure_healthy):,} images")
        
        if len(pure_healthy) > healthy_count:
            healthy_for_output = pure_healthy.sample(n=healthy_count, random_state=seed)
            print(f"Healthy images sampled: {len(healthy_for_output):,} (from {len(pure_healthy):,} available)")
        else:
            healthy_for_output = pure_healthy
            print(f"Using all available healthy images: {len(healthy_for_output):,}")
    
    return {
        'Fibrosis': pure_fibrosis,
        'Pneumonia': pure_pneumonia,
        'Healthy': healthy_for_output
    }


def create_class_csvs(class_data, output_dir):
    """Create CSV files for each class."""
    print(f"\n💾 Creating class CSV files in {output_dir}...")
    
    csv_paths = {}
    for class_name, df in class_data.items():
        if len(df) > 0:
            csv_path = output_dir / f"{class_name.lower()}_images.csv"
            df.to_csv(csv_path, index=False)
            csv_paths[class_name] = csv_path
            print(f"✓ {class_name}: {csv_path} ({len(df):,} images)")
        else:
            print(f"⚠️ {class_name}: No images found, skipping CSV creation")
    
    return csv_paths


def copy_images_from_interim(class_data, output_dir, interim_dir):
    """Copy images from interim directory structure to class-specific folders."""
    interim_path = Path(interim_dir)
    
    if not interim_path.exists():
        print(f"❌ Interim directory not found: {interim_dir}")
        return False
    
    print(f"\n📦 Copying images from {interim_dir}...")
    
    # Check for train_val and test directories
    train_val_dir = interim_path / "train_val"
    test_dir = interim_path / "test"
    
    # Check for diffusion data (for healthy images)
    diffusion_balanced_dir = Path("data/diffusion_data/diffusion_data_balanced")
    
    if not train_val_dir.exists():
        print(f"❌ Train/val directory not found: {train_val_dir}")
        return False
    
    print(f"Found train_val directory: {train_val_dir}")
    if diffusion_balanced_dir.exists():
        print(f"Found diffusion balanced directory: {diffusion_balanced_dir}")
    
    total_copied = 0
    total_missing = 0
    
    for class_name, df in class_data.items():
        if len(df) == 0:
            continue
            
        print(f"\n📁 Processing {class_name}...")
        
        # Create output folder for this class
        output_class_folder = output_dir / class_name.lower()
        output_class_folder.mkdir(parents=True, exist_ok=True)
        
        image_list = df['Image Index'].tolist()
        copied_count = 0
        missing_count = 0
        
        if class_name == 'Healthy':
            # For healthy images, use diffusion balanced data
            if diffusion_balanced_dir.exists():
                for img_name in image_list:
                    source_file = diffusion_balanced_dir / img_name
                    if source_file.exists():
                        dest_file = output_class_folder / img_name
                        shutil.copy2(source_file, dest_file)
                        copied_count += 1
                    else:
                        missing_count += 1
            else:
                print(f"⚠️ Diffusion balanced directory not found, skipping healthy images")
                missing_count = len(image_list)
        else:
            # For pathology images, use train_val directory
            # Map class names to interim folder names
            class_mapping = {
                'Fibrosis': 'Fibrosis',
                'Pneumonia': 'Pneumonia'
            }
            
            interim_class_name = class_mapping.get(class_name, class_name)
            source_path = train_val_dir / interim_class_name
            
            if source_path.exists():
                for img_name in image_list:
                    source_file = source_path / img_name
                    if source_file.exists():
                        dest_file = output_class_folder / img_name
                        shutil.copy2(source_file, dest_file)
                        copied_count += 1
                    else:
                        missing_count += 1
            else:
                print(f"⚠️ Source directory not found: {source_path}")
                missing_count = len(image_list)
        
        print(f"✓ {class_name}: {copied_count:,} images copied, {missing_count:,} missing")
        total_copied += copied_count
        total_missing += missing_count
    
    print(f"\n📊 Total: {total_copied:,} images copied, {total_missing:,} missing")
    return total_copied > 0


def copy_images_to_folders(class_data, output_dir, archive_path):
    """Copy images to class-specific folders."""
    if not Path(archive_path).exists():
        print(f"❌ Archive not found: {archive_path}")
        print("Please ensure archive.zip is available or use --create-csv-only flag")
        return False
    
    print(f"\n📦 Extracting images from {archive_path}...")
    
    with ZipFile(archive_path, 'r') as zip_ref:
        # Get list of files in archive
        archive_files = set(zip_ref.namelist())
        print(f"Archive contains {len(archive_files):,} files")
        
        total_copied = 0
        total_missing = 0
        
        for class_name, df in class_data.items():
            if len(df) == 0:
                continue
                
            print(f"\n📁 Processing {class_name}...")
            class_folder = output_dir / class_name.lower()
            image_list = df['Image Index'].tolist()
            
            copied, missing = extract_images_for_class(
                zip_ref, archive_files, image_list, class_folder
            )
            
            print(f"✓ {class_name}: {copied:,} images copied, {missing:,} missing")
            total_copied += copied
            total_missing += missing
        
        print(f"\n📊 Total: {total_copied:,} images copied, {total_missing:,} missing")
        return total_copied > 0


def main():
    """Main function."""
    args = parse_args()
    
    print("📁 PURE CLASS FOLDERS CREATOR")
    print("=" * 60)
    print(f"Input CSV: {args.input_csv}")
    print(f"Train/val list: {args.train_val_list}")
    print(f"Input Archive: {args.input_archive}")
    print(f"Output directory: {args.output_dir}")
    print(f"Healthy images: {args.healthy_count:,}")
    print(f"Copy images: {args.copy_images}")
    print(f"Create CSV only: {args.create_csv_only}")
    print(f"Random seed: {args.seed}")
    
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
    
    # Load train/val split
    print(f"\n📋 Loading train/val split from {args.train_val_list}...")
    if not Path(args.train_val_list).exists():
        print(f"❌ Train/val list not found: {args.train_val_list}")
        return False
    
    with open(args.train_val_list, 'r') as f:
        train_val_images = set(line.strip() for line in f if line.strip())
    
    print(f"✓ Loaded {len(train_val_images):,} train/val images")
    
    # Analyze and extract pure classes
    class_data = analyze_and_extract_classes(df, train_val_images, args.healthy_count, args.seed)
    
    # Create CSV files for each class
    csv_paths = create_class_csvs(class_data, output_dir)
    
    # Copy images if requested
    if args.copy_images and not args.create_csv_only:
        # Try archive first, then fall back to interim directory
        if Path(args.input_archive).exists():
            print(f"\n🗃️ Using archive: {args.input_archive}")
            success = copy_images_to_folders(class_data, output_dir, args.input_archive)
        elif Path(args.interim_dir).exists():
            print(f"\n📁 Using interim directory: {args.interim_dir}")
            success = copy_images_from_interim(class_data, output_dir, args.interim_dir)
        else:
            print(f"❌ Neither archive ({args.input_archive}) nor interim directory ({args.interim_dir}) found")
            success = False
            
        if not success:
            print("❌ Image copying failed")
            return False
    elif args.create_csv_only:
        print("\n⏭️ Skipping image copying (--create-csv-only flag)")
    else:
        print("\n⏭️ Skipping image copying (use --copy-images to copy files)")
    
    # Print final summary
    print(f"\n📈 FINAL SUMMARY")
    print("=" * 60)
    total_images = 0
    for class_name, df in class_data.items():
        count = len(df)
        total_images += count
        if count > 0:
            folder_path = output_dir / class_name.lower()
            csv_path = output_dir / f"{class_name.lower()}_images.csv"
            print(f"📁 {class_name}:")
            print(f"   Images: {count:,}")
            print(f"   CSV: {csv_path}")
            if args.copy_images and not args.create_csv_only:
                print(f"   Folder: {folder_path}/")
    
    print(f"\n✅ Total: {total_images:,} images organized")
    print(f"📁 Output directory: {output_dir}")
    
    # Show folder structure
    print(f"\n📂 Directory Structure:")
    print(f"{output_dir}/")
    for class_name in class_data.keys():
        if len(class_data[class_name]) > 0:
            print(f"├── {class_name.lower()}_images.csv")
            if args.copy_images and not args.create_csv_only:
                print(f"├── {class_name.lower()}/")
                sample_images = class_data[class_name]['Image Index'].head(3).tolist()
                for i, img in enumerate(sample_images):
                    prefix = "│   ├──" if i < len(sample_images) - 1 else "│   └──"
                    print(f"{prefix} {img}")
                if len(class_data[class_name]) > 3:
                    print(f"│   └── ... and {len(class_data[class_name]) - 3:,} more images")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)