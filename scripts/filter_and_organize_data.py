#!/usr/bin/env python3
"""
Filter and organize chest X-ray images by specified labels.

This script extracts images from the NIH Chest X-ray dataset archive,
filters them by configurable labels, and organizes them into train/val
and test directories with label-based subdirectories.
"""

import os
import zipfile
import pandas as pd
from pathlib import Path
from typing import List, Set, Dict
import shutil
from tqdm import tqdm
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Input paths
    'archive_path': 'data/raw/archive.zip',
    'csv_filename': 'Data_Entry_2017.csv',
    'train_val_list_filename': 'train_val_list.txt',
    'test_list_filename': 'test_list.txt',
    
    # Output paths
    'output_base': 'data/interim',
    'train_val_dir': 'train_val',
    'test_dir': 'test',
    'filtered_csv_name': 'filtered_data_entry.csv',
    
    # Filtering configuration
    'target_labels': ['Hernia', 'Pneumonia', 'Fibrosis', 'Effusion'],
    
    # CSV column names
    'image_column': 'Image Index',
    'label_column': 'Finding Labels',
    'label_separator': '|',
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_csv_from_zip(zip_path: str, csv_filename: str) -> pd.DataFrame:
    """Load CSV file from zip archive without extracting."""
    logger.info(f"Loading {csv_filename} from archive...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(csv_filename) as csv_file:
            df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} entries from CSV")
    return df


def load_image_list_from_zip(zip_path: str, list_filename: str) -> Set[str]:
    """Load image list file from zip archive."""
    logger.info(f"Loading {list_filename} from archive...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(list_filename) as list_file:
            image_list = set(line.decode('utf-8').strip() for line in list_file)
    logger.info(f"Loaded {len(image_list)} images from {list_filename}")
    return image_list


def filter_by_labels(df: pd.DataFrame, target_labels: List[str], 
                     label_column: str, label_separator: str) -> pd.DataFrame:
    """Filter dataframe to keep only rows with at least one target label."""
    logger.info(f"Filtering by labels: {target_labels}")
    
    def has_target_label(label_string):
        if pd.isna(label_string):
            return False
        labels = set(label_string.split(label_separator))
        return bool(labels.intersection(set(target_labels)))
    
    filtered_df = df[df[label_column].apply(has_target_label)].copy()
    logger.info(f"Filtered dataset: {len(filtered_df)} / {len(df)} entries")
    return filtered_df


def get_image_labels(label_string: str, target_labels: List[str], 
                     label_separator: str) -> List[str]:
    """Extract target labels from label string."""
    if pd.isna(label_string):
        return []
    labels = set(label_string.split(label_separator))
    return [label for label in target_labels if label in labels]


def find_image_in_zip(zip_ref: zipfile.ZipFile, image_name: str) -> str:
    """Find the full path of an image in the zip archive."""
    # Images are in folders like images_001/images/, images_002/images/, etc.
    for file_info in zip_ref.filelist:
        if file_info.filename.endswith(image_name):
            return file_info.filename
    return None


def create_output_directories(base_path: Path, split_name: str, 
                              target_labels: List[str]) -> Dict[str, Path]:
    """Create output directory structure."""
    split_path = base_path / split_name
    label_dirs = {}
    
    for label in target_labels:
        label_dir = split_path / label
        label_dir.mkdir(parents=True, exist_ok=True)
        label_dirs[label] = label_dir
        logger.info(f"Created directory: {label_dir}")
    
    return label_dirs


def extract_and_organize_images(zip_path: str, filtered_df: pd.DataFrame,
                                image_list: Set[str], output_dirs: Dict[str, Path],
                                target_labels: List[str], image_column: str,
                                label_column: str, label_separator: str):
    """Extract relevant images from zip and organize by labels."""
    logger.info(f"Extracting and organizing images...")
    
    # Filter to only images in the split list
    split_df = filtered_df[filtered_df[image_column].isin(image_list)]
    logger.info(f"Processing {len(split_df)} images for this split")
    
    extracted_count = 0
    skipped_count = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Extracting images"):
            image_name = row[image_column]
            labels = get_image_labels(row[label_column], target_labels, label_separator)
            
            # Find image in zip
            image_path_in_zip = find_image_in_zip(zip_ref, image_name)
            
            if image_path_in_zip is None:
                logger.warning(f"Image not found in archive: {image_name}")
                skipped_count += 1
                continue
            
            # Extract and copy to all relevant label directories
            for label in labels:
                output_path = output_dirs[label] / image_name
                
                # Extract image data and write to destination
                with zip_ref.open(image_path_in_zip) as source:
                    with open(output_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
            
            extracted_count += 1
    
    logger.info(f"Extracted {extracted_count} images, skipped {skipped_count}")


def save_filtered_csv(df: pd.DataFrame, output_path: Path):
    """Save filtered CSV to disk."""
    df.to_csv(output_path, index=False)
    logger.info(f"Saved filtered CSV to: {output_path}")


def print_statistics(filtered_df: pd.DataFrame, train_val_list: Set[str],
                     test_list: Set[str], target_labels: List[str],
                     image_column: str, label_column: str, label_separator: str):
    """Print dataset statistics."""
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    
    train_val_df = filtered_df[filtered_df[image_column].isin(train_val_list)]
    test_df = filtered_df[filtered_df[image_column].isin(test_list)]
    
    logger.info(f"Total filtered images: {len(filtered_df)}")
    logger.info(f"Train/Val images: {len(train_val_df)}")
    logger.info(f"Test images: {len(test_df)}")
    logger.info("")
    
    for label in target_labels:
        total_count = filtered_df[label_column].str.contains(label, regex=False, na=False).sum()
        train_val_count = train_val_df[label_column].str.contains(label, regex=False, na=False).sum()
        test_count = test_df[label_column].str.contains(label, regex=False, na=False).sum()
        
        logger.info(f"{label}: {total_count} total ({train_val_count} train/val, {test_count} test)")
    
    logger.info("=" * 60)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("Starting data filtering and organization process")
    logger.info(f"Target labels: {CONFIG['target_labels']}")
    
    # Convert paths to Path objects
    archive_path = Path(CONFIG['archive_path'])
    output_base = Path(CONFIG['output_base'])
    
    # Check if archive exists
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    
    # Step 1: Load data from archive
    df = load_csv_from_zip(str(archive_path), CONFIG['csv_filename'])
    train_val_list = load_image_list_from_zip(str(archive_path), CONFIG['train_val_list_filename'])
    test_list = load_image_list_from_zip(str(archive_path), CONFIG['test_list_filename'])
    
    # Step 2: Filter by labels
    filtered_df = filter_by_labels(
        df, 
        CONFIG['target_labels'],
        CONFIG['label_column'],
        CONFIG['label_separator']
    )
    
    # Step 3: Print statistics
    print_statistics(
        filtered_df,
        train_val_list,
        test_list,
        CONFIG['target_labels'],
        CONFIG['image_column'],
        CONFIG['label_column'],
        CONFIG['label_separator']
    )
    
    # Step 4: Create output directories
    logger.info("Creating output directory structure...")
    train_val_dirs = create_output_directories(
        output_base,
        CONFIG['train_val_dir'],
        CONFIG['target_labels']
    )
    test_dirs = create_output_directories(
        output_base,
        CONFIG['test_dir'],
        CONFIG['target_labels']
    )
    
    # Step 5: Extract and organize train/val images
    logger.info("Processing train/val split...")
    extract_and_organize_images(
        str(archive_path),
        filtered_df,
        train_val_list,
        train_val_dirs,
        CONFIG['target_labels'],
        CONFIG['image_column'],
        CONFIG['label_column'],
        CONFIG['label_separator']
    )
    
    # Step 6: Extract and organize test images
    logger.info("Processing test split...")
    extract_and_organize_images(
        str(archive_path),
        filtered_df,
        test_list,
        test_dirs,
        CONFIG['target_labels'],
        CONFIG['image_column'],
        CONFIG['label_column'],
        CONFIG['label_separator']
    )
    
    # Step 7: Save filtered CSV
    filtered_csv_path = output_base / CONFIG['filtered_csv_name']
    save_filtered_csv(filtered_df, filtered_csv_path)
    
    logger.info("=" * 60)
    logger.info("PROCESS COMPLETED SUCCESSFULLY")
    logger.info(f"Output directory: {output_base}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
