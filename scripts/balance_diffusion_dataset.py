"""
Balance the diffusion dataset to have approximately N images per label.

This script:
1. Loads the existing diffusion_dataset.csv
2. Identifies labels with <= N images (keeps all)
3. For labels with > N images, removes images until reaching N to N+50 range
4. Creates a balanced CSV (copy of original)
5. Filters the actual image files based on the balanced CSV
6. Creates a new zip file with the balanced dataset
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

# Paths
DATA_DIR = Path("data/diffusion_data")
INPUT_CSV = DATA_DIR / "diffusion_dataset.csv"
BALANCED_CSV = DATA_DIR / "diffusion_dataset_balanced.csv"
BALANCED_SUMMARY_CSV = DATA_DIR / "diffusion_dataset_balanced_summary.csv"
BALANCED_ZIP = Path("data/diffusion_data_balanced.zip")

# All 15 possible labels in NIH dataset
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

# Target range parameters
TARGET_N = None  # Will be set from target label
MAX_OVERFLOW = 50  # Allow up to N+50 images per label


def load_dataset():
    """Load the existing diffusion dataset."""
    print(f"Loading dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total images: {len(df)}")
    return df


def count_images_per_label(df):
    """Count how many images contain each label."""
    label_counts = {}
    for label in ALL_LABELS:
        count = df['Finding Labels'].str.contains(label, regex=False, na=False).sum()
        label_counts[label] = count
    return label_counts


def identify_target_n(label_counts, target_label):
    """Identify the target N from the specified label."""
    target_n = label_counts[target_label]
    print(f"\nTarget label '{target_label}' has {target_n} images")
    print(f"Will balance other labels to: {target_n} to {target_n + MAX_OVERFLOW} images")
    return target_n


def classify_labels(label_counts, target_n):
    """Classify labels into those needing balancing vs those already balanced."""
    labels_ok = {}  # Labels with <= target_n images
    labels_to_reduce = {}  # Labels with > target_n images
    
    for label, count in label_counts.items():
        if count <= target_n:
            labels_ok[label] = count
        else:
            labels_to_reduce[label] = count
    
    print(f"\n=== Label Classification ===")
    print(f"Labels already balanced (<= {target_n}): {len(labels_ok)}")
    for label, count in sorted(labels_ok.items()):
        print(f"  {label}: {count}")
    
    print(f"\nLabels needing reduction (> {target_n}): {len(labels_to_reduce)}")
    for label, count in sorted(labels_to_reduce.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count} → target: {target_n}-{target_n + MAX_OVERFLOW}")
    
    return labels_ok, labels_to_reduce


def balance_dataset(df, labels_to_reduce, labels_ok, target_n):
    """
    Remove images iteratively to balance labels.
    
    Strategy:
    - Start with all images
    - For each over-represented label, randomly remove images
    - PROTECT images that contain any label with count <= target_n
    - Continue until all labels are in range [target_n, target_n + MAX_OVERFLOW]
    """
    print(f"\n=== Balancing Dataset ===")
    
    # Work with a copy
    balanced_df = df.copy()
    current_counts = count_images_per_label(balanced_df)
    
    # Identify protected labels (those with <= target_n)
    protected_labels = set(labels_ok.keys())
    print(f"\nProtected labels (will not remove images containing these): {protected_labels}")
    
    # Track images to remove
    images_to_remove = set()
    
    # Identify protected images (contain any protected label)
    def is_protected(finding_labels):
        """Check if an image contains any protected label."""
        for label in protected_labels:
            if label in str(finding_labels):
                return True
        return False
    
    protected_mask = balanced_df['Finding Labels'].apply(is_protected)
    protected_images = set(balanced_df[protected_mask]['Image Index'])
    print(f"Protected images: {len(protected_images)}")
    
    # Iteratively reduce over-represented labels
    max_iterations = 100
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Find labels still over the limit
        over_limit = {label: count for label, count in current_counts.items() 
                     if count > target_n + MAX_OVERFLOW}
        
        if not over_limit:
            print(f"Converged after {iteration} iterations")
            break
        
        # Find the most over-represented label
        worst_label = max(over_limit.items(), key=lambda x: x[1])[0]
        worst_count = over_limit[worst_label]
        
        # Calculate how many to remove
        excess = worst_count - target_n
        
        # Get all remaining images with this label that are NOT protected
        mask = balanced_df['Finding Labels'].str.contains(worst_label, regex=False, na=False)
        candidate_images = balanced_df[
            mask & 
            ~balanced_df['Image Index'].isin(images_to_remove) &
            ~balanced_df['Image Index'].isin(protected_images)
        ]
        
        if len(candidate_images) == 0:
            print(f"Warning: No more removable images for {worst_label} (all remaining are protected)")
            break
        
        # Randomly select images to remove
        # Remove more aggressively if we're far over the limit
        n_to_remove = min(excess // 2 + 1, len(candidate_images))
        images_to_drop = candidate_images.sample(n=n_to_remove, random_state=RANDOM_SEED + iteration)
        
        # Add to removal set
        for img in images_to_drop['Image Index']:
            images_to_remove.add(img)
        
        # Recalculate counts after removal
        temp_df = balanced_df[~balanced_df['Image Index'].isin(images_to_remove)]
        current_counts = count_images_per_label(temp_df)
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: {len(images_to_remove)} images marked for removal")
    
    # Apply removals
    balanced_df = balanced_df[~balanced_df['Image Index'].isin(images_to_remove)]
    balanced_df = balanced_df.reset_index(drop=True)
    
    print(f"\nRemoved {len(images_to_remove)} images")
    print(f"Balanced dataset size: {len(balanced_df)}")
    
    return balanced_df


def generate_statistics(df):
    """Generate statistics about the balanced dataset."""
    stats = {}
    
    # Count images per label
    label_counts = count_images_per_label(df)
    
    # Count multi-label images
    multi_label_count = df['Finding Labels'].str.contains('\\|', regex=True).sum()
    single_label_count = len(df) - multi_label_count
    
    stats['total_images'] = len(df)
    stats['single_label_images'] = single_label_count
    stats['multi_label_images'] = multi_label_count
    stats['label_counts'] = label_counts
    
    return stats


def save_statistics(stats, output_path):
    """Save statistics to CSV."""
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
    print("\n=== Balanced Dataset Statistics ===")
    print(f"Total Images: {stats['total_images']}")
    print(f"Single-Label Images: {stats['single_label_images']}")
    print(f"Multi-Label Images: {stats['multi_label_images']}")
    print("\nImages per Label:")
    for label, count in stats['label_counts'].items():
        print(f"  {label}: {count}")


def filter_images(balanced_df, original_df):
    """
    Remove image files that are not in the balanced dataset.
    """
    print("\n=== Filtering Image Files ===")
    
    # Get sets of image filenames
    balanced_images = set(balanced_df['Image Index'])
    original_images = set(original_df['Image Index'])
    
    # Images to remove
    images_to_remove = original_images - balanced_images
    
    print(f"Images to keep: {len(balanced_images)}")
    print(f"Images to remove: {len(images_to_remove)}")
    
    if len(images_to_remove) == 0:
        print("No images to remove!")
        return
    
    # Remove image files
    removed_count = 0
    not_found_count = 0
    
    for img_name in tqdm(images_to_remove, desc="Removing images"):
        img_path = DATA_DIR / img_name
        if img_path.exists():
            img_path.unlink()
            removed_count += 1
        else:
            not_found_count += 1
    
    print(f"✓ Removed {removed_count} image files")
    if not_found_count > 0:
        print(f"  (Warning: {not_found_count} files were already missing)")


def create_balanced_zip():
    """Create zip file of the balanced dataset."""
    print(f"\n=== Creating Balanced Zip File ===")
    print(f"Output: {BALANCED_ZIP}")
    
    # Remove existing zip if it exists
    if BALANCED_ZIP.exists():
        BALANCED_ZIP.unlink()
    
    # Create zip
    with zipfile.ZipFile(BALANCED_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add CSV files
        zipf.write(BALANCED_CSV, BALANCED_CSV.relative_to(DATA_DIR.parent))
        zipf.write(BALANCED_SUMMARY_CSV, BALANCED_SUMMARY_CSV.relative_to(DATA_DIR.parent))
        
        # Add all image files from the directory
        image_files = list(DATA_DIR.glob('*.png'))
        for img_file in tqdm(image_files, desc="Zipping files"):
            arcname = img_file.relative_to(DATA_DIR.parent)
            zipf.write(img_file, arcname)
    
    zip_size_mb = BALANCED_ZIP.stat().st_size / (1024 * 1024)
    print(f"✓ Zip file created: {BALANCED_ZIP} ({zip_size_mb:.2f} MB)")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Balance diffusion dataset")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes: delete removed images and create new zip file"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Balancing Diffusion Dataset")
    print("="*60)
    
    # Load dataset
    original_df = load_dataset()
    
    # Count initial distribution
    print("\n=== Initial Label Counts ===")
    initial_counts = count_images_per_label(original_df)
    for label, count in initial_counts.items():
        print(f"{label}: {count}")
    
    # Identify target N from Fibrosis (or could be made configurable)
    target_label = "Fibrosis"
    target_n = identify_target_n(initial_counts, target_label)
    
    # Classify labels
    labels_ok, labels_to_reduce = classify_labels(initial_counts, target_n)
    
    if not labels_to_reduce:
        print("\n✓ Dataset already balanced! No changes needed.")
        return
    
    # Balance the dataset
    balanced_df = balance_dataset(original_df, labels_to_reduce, labels_ok, target_n)
    
    # Generate statistics
    stats = generate_statistics(balanced_df)
    
    # Save balanced CSV
    balanced_df.to_csv(BALANCED_CSV, index=False)
    print(f"\n✓ Balanced dataset CSV saved to {BALANCED_CSV}")
    
    # Save statistics
    save_statistics(stats, BALANCED_SUMMARY_CSV)
    
    # Apply changes if requested
    if args.apply:
        print("\n" + "="*60)
        print("Applying Changes: Removing Images and Creating Zip")
        print("="*60)
        
        # Filter images
        filter_images(balanced_df, original_df)
        
        # Create new zip
        create_balanced_zip()
        
        print("\n" + "="*60)
        print("✓ All changes applied successfully!")
        print("="*60)
        print(f"Balanced zip file: {BALANCED_ZIP}")
    else:
        print("\n" + "="*60)
        print("✓ Balancing complete (CSV only)")
        print("="*60)
        print(f"To apply changes (delete images and create zip), run:")
        print(f"  python scripts/balance_diffusion_dataset.py --apply")
    
    print(f"\nOriginal dataset: {len(original_df)} images")
    print(f"Balanced dataset: {len(balanced_df)} images")
    print(f"To be removed: {len(original_df) - len(balanced_df)} images")
    print(f"\nBalanced CSV: {BALANCED_CSV}")
    print(f"Summary CSV: {BALANCED_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
