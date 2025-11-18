"""
Prepare dataset for diffusion model training.

This script:
1. Counts images containing the target rare label (N)
2. Randomly samples N images from each of the other pathology labels
3. Removes duplicates from combined dataset
4. Extracts images from archive.zip to data/diffusion_data/
5. Creates CSV and summary statistics
6. Zips the final dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import zipfile
import shutil
import argparse
from tqdm import tqdm

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Target rare label to use as sampling baseline
TARGET_RARE_LABEL = "Fibrosis"  # Can be changed to any label

# Paths
RAW_DATA_DIR = Path("data/raw")
CSV_PATH = RAW_DATA_DIR / "Data_Entry_2017.csv"
TRAIN_VAL_LIST = RAW_DATA_DIR / "train_val_list.txt"
TEST_LIST = RAW_DATA_DIR / "test_list.txt"
ARCHIVE_PATH = RAW_DATA_DIR / "archive.zip"
OUTPUT_DIR = Path("data/diffusion_data")
OUTPUT_CSV = OUTPUT_DIR / "diffusion_dataset.csv"
SUMMARY_CSV = OUTPUT_DIR / "diffusion_dataset_summary.csv"
OUTPUT_ZIP = Path("data/diffusion_data.zip")

# All 15 possible labels in NIH dataset (14 pathologies + No Finding)
ALL_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding"
]

def load_train_val_list():
    """Load the train_val split file."""
    print(f"Loading train/val split from {TRAIN_VAL_LIST}...")
    with open(TRAIN_VAL_LIST, 'r') as f:
        train_val_images = set(line.strip() for line in f)
    print(f"Train/val images: {len(train_val_images)}")
    return train_val_images

def load_data():
    """Load the NIH dataset CSV and filter to train_val split only."""
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Total images in full dataset: {len(df)}")
    
    # Load train_val split
    train_val_images = load_train_val_list()
    
    # Filter to only train_val images
    df = df[df['Image Index'].isin(train_val_images)].copy()
    df = df.reset_index(drop=True)
    print(f"Images in train/val split: {len(df)}")
    
    return df

def count_images_per_label(df):
    """Count how many images contain each label (including multi-label)."""
    label_counts = {}
    for label in ALL_LABELS:
        # Count images where the label appears in Finding Labels
        count = df['Finding Labels'].str.contains(label, regex=False, na=False).sum()
        label_counts[label] = count
    return label_counts

def sample_images_per_label(df, target_count):
    """
    Sample target_count images for each label.
    Returns a dictionary mapping label -> list of sampled image indices.
    """
    sampled_indices = {}
    
    for label in ALL_LABELS:
        # Get all images containing this label
        mask = df['Finding Labels'].str.contains(label, regex=False, na=False)
        label_images = df[mask].index.tolist()
        
        # Sample target_count images (or all if fewer available)
        n_to_sample = min(target_count, len(label_images))
        sampled = np.random.choice(label_images, size=n_to_sample, replace=False)
        sampled_indices[label] = sampled.tolist()
        
        print(f"{label}: {len(label_images)} available, sampled {n_to_sample}")
    
    return sampled_indices

def combine_and_deduplicate(df, sampled_indices):
    """
    Combine all sampled indices and remove duplicates.
    Returns DataFrame with unique images.
    """
    # Combine all sampled indices
    all_indices = []
    for label, indices in sampled_indices.items():
        all_indices.extend(indices)
    
    print(f"\nTotal sampled (with duplicates): {len(all_indices)}")
    
    # Remove duplicates
    unique_indices = list(set(all_indices))
    print(f"Unique images after deduplication: {len(unique_indices)}")
    
    # Create DataFrame with unique images
    final_df = df.loc[unique_indices].copy()
    final_df = final_df.reset_index(drop=True)
    
    return final_df

def generate_statistics(final_df):
    """Generate statistics about the final dataset."""
    stats = {}
    
    # Count images per label in final dataset
    label_counts = {}
    for label in ALL_LABELS:
        count = final_df['Finding Labels'].str.contains(label, regex=False, na=False).sum()
        label_counts[label] = count
    
    # Count multi-label images
    multi_label_count = final_df['Finding Labels'].str.contains('\|', regex=True).sum()
    single_label_count = len(final_df) - multi_label_count
    
    stats['total_images'] = len(final_df)
    stats['single_label_images'] = single_label_count
    stats['multi_label_images'] = multi_label_count
    stats['label_counts'] = label_counts
    
    return stats

def save_statistics(stats, output_path):
    """Save statistics to CSV."""
    # Create summary DataFrame
    summary_data = {
        'Metric': ['Total Images', 'Single-Label Images', 'Multi-Label Images'],
        'Count': [stats['total_images'], stats['single_label_images'], stats['multi_label_images']]
    }
    
    # Add label counts
    for label, count in stats['label_counts'].items():
        summary_data['Metric'].append(f"Images with {label}")
        summary_data['Count'].append(count)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    print(f"\nStatistics saved to {output_path}")
    
    # Print summary
    print("\n=== Dataset Statistics ===")
    print(f"Total Images: {stats['total_images']}")
    print(f"Single-Label Images: {stats['single_label_images']}")
    print(f"Multi-Label Images: {stats['multi_label_images']}")
    print("\nImages per Label:")
    for label, count in stats['label_counts'].items():
        print(f"  {label}: {count}")

def extract_images_from_archive(final_df, archive_path, output_dir):
    """Extract selected images from archive.zip to output directory."""
    print(f"\nExtracting images from {archive_path}...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images to extract
    images_to_extract = set(final_df['Image Index'].tolist())
    
    # Open the archive
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        # Get all file names in archive
        all_files = zip_ref.namelist()
        
        # Filter for image files we need
        files_to_extract = []
        for file in all_files:
            filename = Path(file).name
            if filename in images_to_extract:
                files_to_extract.append(file)
        
        print(f"Found {len(files_to_extract)} images to extract (out of {len(images_to_extract)} requested)")
        
        # Extract files with progress bar
        for file in tqdm(files_to_extract, desc="Extracting images"):
            # Extract to output directory (flatten structure)
            filename = Path(file).name
            source = zip_ref.open(file)
            target = output_dir / filename
            with open(target, 'wb') as f:
                f.write(source.read())
    
    print(f"Images extracted to {output_dir}")

def create_zip(source_dir, output_zip):
    """Create zip file of the output directory."""
    print(f"\nCreating zip file: {output_zip}...")
    
    # Remove existing zip if it exists
    if output_zip.exists():
        output_zip.unlink()
    
    # Create zip
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in tqdm(list(source_dir.rglob('*')), desc="Zipping files"):
            if file.is_file():
                # Add file to zip with relative path
                arcname = file.relative_to(source_dir.parent)
                zipf.write(file, arcname)
    
    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"Zip file created: {output_zip} ({zip_size_mb:.2f} MB)")

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare dataset for diffusion model training")
    parser.add_argument(
        "--target-label",
        type=str,
        default=TARGET_RARE_LABEL,
        choices=ALL_LABELS,
        help=f"Target rare label to use as sampling baseline (default: {TARGET_RARE_LABEL})"
    )
    args = parser.parse_args()
    
    target_label = args.target_label
    
    print("="*60)
    print("Preparing Diffusion Model Dataset")
    print("="*60)
    print(f"Target rare label: {target_label}")
    
    # Load data
    df = load_data()
    
    # Count images per label
    print("\n=== Initial Label Counts ===")
    label_counts = count_images_per_label(df)
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    
    # Get target label count as baseline
    target_count = label_counts[target_label]
    print(f"\n=== Sampling Strategy ===")
    print(f"{target_label} count: {target_count}")
    print(f"Will sample {target_count} images from each label (or all available if fewer)")
    
    # Sample images for each label
    print("\n=== Sampling Images ===")
    sampled_indices = sample_images_per_label(df, target_count)
    
    # Combine and deduplicate
    print("\n=== Combining and Deduplicating ===")
    final_df = combine_and_deduplicate(df, sampled_indices)
    
    # Generate statistics
    stats = generate_statistics(final_df)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save final dataset CSV
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset CSV saved to {OUTPUT_CSV}")
    
    # Save statistics
    save_statistics(stats, SUMMARY_CSV)
    
    # Extract images from archive
    extract_images_from_archive(final_df, ARCHIVE_PATH, OUTPUT_DIR)
    
    # Create zip file
    create_zip(OUTPUT_DIR, OUTPUT_ZIP)
    
    print("\n" + "="*60)
    print("✓ Dataset preparation complete!")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Dataset CSV: {OUTPUT_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Zip file: {OUTPUT_ZIP}")

if __name__ == "__main__":
    main()
