"""Data preprocessing module for binary classification."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_split_files(config: Config) -> dict:
    """
    Load train_val_list.txt and test_list.txt to create split mapping.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary mapping image_name -> split ('train_val' or 'test')
    """
    logger.info("Loading split files...")
    
    # Get data directory from train_val_dir (data/interim/train_val -> data)
    data_dir = config.data.train_val_dir.parent.parent
    train_val_list = data_dir / 'raw' / 'train_val_list.txt'
    test_list = data_dir / 'raw' / 'test_list.txt'
    
    split_map = {}
    
    # Load train_val images
    with open(train_val_list, 'r') as f:
        for line in f:
            image_name = line.strip()
            if image_name:
                split_map[image_name] = 'train_val'
    
    # Load test images
    with open(test_list, 'r') as f:
        for line in f:
            image_name = line.strip()
            if image_name:
                split_map[image_name] = 'test'
    
    logger.info(f"Loaded {len([s for s in split_map.values() if s == 'train_val'])} train_val images")
    logger.info(f"Loaded {len([s for s in split_map.values() if s == 'test'])} test images")
    
    return split_map


def filter_and_prepare_binary_dataset(config: Config, split_map: dict) -> pd.DataFrame:
    """
    Filter CSV for binary classification task across both splits.
    
    Steps:
    1. Load interim CSV
    2. Add split column based on split_map
    3. Keep only images with at least one target class
    4. Remove images labeled with BOTH classes
    5. Strip out non-target labels
    6. Create binary label column (0=negative, 1=positive)
    7. Add image paths
    8. Remove images where file doesn't exist
    
    Args:
        config: Configuration object
        split_map: Dictionary mapping image_name -> split
        
    Returns:
        Filtered dataframe with all splits
    """
    logger.info("Loading interim CSV...")
    df = pd.read_csv(config.data.interim_csv)
    logger.info(f"Loaded {len(df)} entries from {config.data.interim_csv}")
    
    # Add split column
    df['original_split'] = df['Image Index'].map(split_map)
    
    class_pos = config.data.class_positive
    class_neg = config.data.class_negative
    label_col = 'Finding Labels'
    
    # Function to check if row has target class
    def has_class(labels_str, target_class):
        if pd.isna(labels_str):
            return False
        labels = set(labels_str.split('|'))
        return target_class in labels
    
    # Keep only rows with at least one target class
    mask_has_target = df[label_col].apply(
        lambda x: has_class(x, class_pos) or has_class(x, class_neg)
    )
    df_filtered = df[mask_has_target].copy()
    logger.info(f"After filtering for target classes: {len(df_filtered)} entries")
    
    # Remove images with BOTH classes
    mask_both = df_filtered[label_col].apply(
        lambda x: has_class(x, class_pos) and has_class(x, class_neg)
    )
    df_filtered = df_filtered[~mask_both].copy()
    logger.info(f"After removing dual-labeled images: {len(df_filtered)} entries")
    
    # Create clean label column (only target classes)
    def clean_labels(labels_str):
        """Keep only target class labels."""
        if pd.isna(labels_str):
            return ""
        labels = labels_str.split('|')
        target_labels = [l for l in labels if l in [class_pos, class_neg]]
        return '|'.join(target_labels)
    
    df_filtered['Finding Labels'] = df_filtered[label_col].apply(clean_labels)
    
    # Create binary label (0=negative, 1=positive)
    df_filtered['label'] = df_filtered['Finding Labels'].apply(
        lambda x: 1 if class_pos in x else 0
    )
    
    # Add image paths based on original split
    def get_image_path(row):
        """Find image path in organized directories."""
        image_name = row['Image Index']
        split_type = row['original_split']
        
        # Check both target class directories
        for label_class in [class_pos, class_neg]:
            if split_type == 'train_val':
                path = config.data.train_val_dir / label_class / image_name
            elif split_type == 'test':
                path = config.data.test_dir / label_class / image_name
            else:
                continue
            
            if path.exists():
                return str(path)
        return None
    
    df_filtered['image_path'] = df_filtered.apply(get_image_path, axis=1)
    
    # Remove rows where image doesn't exist
    initial_count = len(df_filtered)
    df_filtered = df_filtered[df_filtered['image_path'].notna()].copy()
    removed = initial_count - len(df_filtered)
    if removed > 0:
        logger.warning(f"Removed {removed} entries with missing image files")
    
    logger.info(f"Final filtered dataset: {len(df_filtered)} entries")
    
    return df_filtered


def split_train_val_and_report_stats(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Split train_val into train and val, add to test, and report statistics.
    
    Args:
        df: Filtered dataframe with original_split column
        config: Configuration object
        
    Returns:
        DataFrame with final 'split' column (train/val/test)
    """
    class_pos = config.data.class_positive
    class_neg = config.data.class_negative
    
    # Separate train_val and test
    df_train_val = df[df['original_split'] == 'train_val'].copy()
    df_test = df[df['original_split'] == 'test'].copy()
    df_test['split'] = 'test'
    
    logger.info(f"\n{'='*80}")
    logger.info("SPLITTING TRAIN_VAL INTO TRAIN AND VAL")
    logger.info(f"{'='*80}")
    logger.info(f"train_val images: {len(df_train_val)}")
    logger.info(f"Split ratio: {config.data.train_val_split:.2f} train, {1-config.data.train_val_split:.2f} val")
    
    # Set random seed for reproducibility
    np.random.seed(config.experiment.seed)
    
    # Stratified split
    train_df, val_df = train_test_split(
        df_train_val,
        train_size=config.data.train_val_split,
        stratify=df_train_val['label'] if config.data.stratified else None,
        random_state=config.experiment.seed
    )
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    # Combine all splits
    df_final = pd.concat([train_df, val_df, df_test], ignore_index=True)
    
    logger.info(f"\n{'='*80}")
    logger.info("DATASET STATISTICS")
    logger.info(f"{'='*80}")
    
    # Overall statistics
    total = len(df_final)
    logger.info(f"\nTotal images: {total}")
    logger.info(f"  {class_neg} (label=0): {(df_final['label'] == 0).sum()} ({(df_final['label'] == 0).sum()/total*100:.1f}%)")
    logger.info(f"  {class_pos} (label=1): {(df_final['label'] == 1).sum()} ({(df_final['label'] == 1).sum()/total*100:.1f}%)")
    
    # Per-split statistics
    for split_name in ['train', 'val', 'test']:
        df_split = df_final[df_final['split'] == split_name]
        split_total = len(df_split)
        neg_count = (df_split['label'] == 0).sum()
        pos_count = (df_split['label'] == 1).sum()
        
        logger.info(f"\n{split_name.upper()} split: {split_total} images ({split_total/total*100:.1f}% of total)")
        logger.info(f"  {class_neg} (label=0): {neg_count} ({neg_count/split_total*100:.1f}% of {split_name})")
        logger.info(f"  {class_pos} (label=1): {pos_count} ({pos_count/split_total*100:.1f}% of {split_name})")
    
    return df_final


def save_processed_csv(df: pd.DataFrame, config: Config):
    """
    Save unified processed CSV for training and generate summary statistics.
    
    Args:
        df: Final dataframe with all splits
        config: Configuration object
    """
    # Create output directory
    config.data.processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Select columns to save
    columns_to_save = ['Image Index', 'image_path', 'Finding Labels', 'label', 'split']
    
    # Save unified CSV
    unified_path = config.data.processed_dir / 'dataset.csv'
    df[columns_to_save].to_csv(unified_path, index=False)
    
    # Generate summary statistics
    class_pos = config.data.class_positive
    class_neg = config.data.class_negative
    total = len(df)
    
    summary_data = []
    for split_name in ['train', 'val', 'test']:
        df_split = df[df['split'] == split_name]
        split_total = len(df_split)
        neg_count = (df_split['label'] == 0).sum()
        pos_count = (df_split['label'] == 1).sum()
        
        summary_data.append({
            'split': split_name,
            'total_images': split_total,
            'percent_of_dataset': round(split_total / total * 100, 2),
            f'{class_neg}_count': neg_count,
            f'{class_neg}_percent': round(neg_count / split_total * 100, 2),
            f'{class_pos}_count': pos_count,
            f'{class_pos}_percent': round(pos_count / split_total * 100, 2),
        })
    
    # Add overall totals
    summary_data.append({
        'split': 'TOTAL',
        'total_images': total,
        'percent_of_dataset': 100.0,
        f'{class_neg}_count': (df['label'] == 0).sum(),
        f'{class_neg}_percent': round((df['label'] == 0).sum() / total * 100, 2),
        f'{class_pos}_count': (df['label'] == 1).sum(),
        f'{class_pos}_percent': round((df['label'] == 1).sum() / total * 100, 2),
    })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to experiment-specific outputs directory
    outputs_dir = Path('outputs') / config.experiment.name
    outputs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = outputs_dir / 'dataset_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Saved unified dataset: {unified_path}")
    logger.info(f"Saved summary statistics: {summary_path}")
    logger.info(f"{'='*80}")


def preprocess_data(config_path: str):
    """
    Main preprocessing pipeline.
    
    Args:
        config_path: Path to configuration YAML file
    """
    logger.info("=" * 80)
    logger.info("DATA PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Binary classification: {config.data.class_negative} (0) vs {config.data.class_positive} (1)")
    
    # Load split files
    split_map = load_split_files(config)
    
    # Filter and prepare dataset for both train_val and test
    df_filtered = filter_and_prepare_binary_dataset(config, split_map)
    
    # Split train_val into train/val and combine with test
    df_final = split_train_val_and_report_stats(df_filtered, config)
    
    # Save unified CSV
    save_processed_csv(df_final, config)
    
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


def crop_border_and_resize(image: np.ndarray, crop_pixels: int = 10, target_size: int = 512) -> np.ndarray:
    """
    Crop border pixels from all sides and resize back to target size.

    This removes artifacts (e.g., white borders) from generated images while maintaining
    a consistent zoom level across real and generated images.

    The crop is applied PROPORTIONALLY based on image size relative to 512px.
    For example, if crop_pixels=10:
    - 512x512 image: crops 10px from each side (2% of image)
    - 1024x1024 image: crops 20px from each side (2% of image)

    This ensures both generated (512x512) and real (1024x1024) images have
    the same effective zoom after cropping.

    Args:
        image: Input image as numpy array (H, W, C) or (H, W), dtype=uint8 in range [0, 255]
        crop_pixels: Number of pixels to crop from each side at 512px scale (default: 10)
        target_size: Target size for output image (default: 512 for 512×512)

    Returns:
        Cropped and resized image as numpy array, dtype=uint8 in range [0, 255]

    Example:
        # Remove white borders from generated images
        >>> from PIL import Image
        >>> img = np.array(Image.open('generated_image.png'))  # 512×512×3
        >>> img_clean = crop_border_and_resize(img, crop_pixels=10, target_size=512)
        >>> img_clean.shape  # Still 512×512×3, but with borders removed

    Note:
        This function maintains the original value range [0, 255] and dtype (uint8).
        No normalization is applied - use this before evaluation metrics.
    """
    from PIL import Image

    # Validate input
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")

    # Convert to PIL for easy cropping and resizing
    pil_img = Image.fromarray(image)

    # Get original dimensions
    width, height = pil_img.size

    # Scale crop pixels proportionally to image size (relative to 512px base)
    # This ensures same zoom level regardless of input resolution
    scale_factor = width / 512.0  # Assume square images
    actual_crop = int(crop_pixels * scale_factor)

    # Crop from all sides
    # Box format: (left, upper, right, lower)
    crop_box = (
        actual_crop,           # left
        actual_crop,           # upper
        width - actual_crop,   # right
        height - actual_crop   # lower
    )

    cropped_img = pil_img.crop(crop_box)

    # Resize back to target size
    # Use LANCZOS for high-quality downsampling (medical images need quality preservation)
    resized_img = cropped_img.resize((target_size, target_size), Image.LANCZOS)

    # Convert back to numpy array
    result = np.array(resized_img)

    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess data for binary classification')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    preprocess_data(args.config)
