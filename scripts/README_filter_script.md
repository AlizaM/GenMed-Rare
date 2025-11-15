# Data Filtering and Organization Script

## Overview
This script filters and organizes the NIH Chest X-ray dataset by specified pathology labels. It extracts only relevant images from the archive and organizes them into train/val and test directories with label-based subdirectories.

## Features
- **Selective extraction**: Extracts only images matching target labels (no need to extract the entire 42GB archive)
- **Multi-label support**: Handles images with multiple labels (separated by `|`)
- **Organized output**: Creates separate directories for each label within train/val and test splits
- **Filtered CSV**: Generates a filtered CSV containing only relevant entries
- **Progress tracking**: Shows extraction progress with tqdm progress bars
- **Statistics**: Displays dataset statistics after filtering

## Configuration

Edit the `CONFIG` dictionary in the script to customize:

```python
CONFIG = {
    # Target labels to filter (only these labels will have folders created)
    'target_labels': ['Hernia', 'Pneumonia', 'Fibrosis', 'Effusion'],
    
    # Input paths (relative to project root)
    'archive_path': 'data/raw/archive.zip',
    
    # Output paths
    'output_base': 'data/interim',
    'train_val_dir': 'train_val',
    'test_dir': 'test',
    'filtered_csv_name': 'filtered_data_entry.csv',
}
```

## Output Structure

```
data/interim/
├── filtered_data_entry.csv          # Filtered CSV with only target labels
├── train_val/
│   ├── Hernia/
│   │   ├── 00000003_000.png
│   │   └── ...
│   ├── Pneumonia/
│   │   └── ...
│   ├── Fibrosis/
│   │   └── ...
│   └── Effusion/
│       └── ...
└── test/
    ├── Hernia/
    ├── Pneumonia/
    ├── Fibrosis/
    └── Effusion/
```

**Note**: Images with multiple labels (e.g., "Hernia|Infiltration") will be copied to ALL relevant target label folders. For example, an image labeled "Hernia|Pneumonia" will appear in both the `Hernia/` and `Pneumonia/` subdirectories.

## Usage

### Prerequisites
Ensure required packages are installed:
```bash
pip install pandas tqdm
```

### Run the script
From the project root directory:
```bash
python scripts/filter_and_organize_data.py
```

## How It Works

1. **Load data**: Reads CSV and train/val/test split lists directly from the zip archive
2. **Filter**: Filters the CSV to keep only rows with at least one target label
3. **Statistics**: Displays counts for each label in train/val and test splits
4. **Create directories**: Creates output folder structure with label subdirectories
5. **Extract images**: Selectively extracts only relevant images from the zip archive
6. **Organize**: Places each image in the appropriate split and label folder(s)
7. **Save CSV**: Exports the filtered CSV for reference

## Dataset Information

- **Source**: NIH Chest X-ray Dataset
- **CSV columns**:
  - `Image Index`: Image filename
  - `Finding Labels`: Pipe-separated labels (e.g., "Hernia|Infiltration")
- **Split files**: `train_val_list.txt` and `test_list.txt` define the data splits
- **Image location**: Images are stored in `images_XXX/images/` folders within the archive

## Customization

To change target labels, modify the `target_labels` list in the `CONFIG` dictionary:
```python
'target_labels': ['Cardiomegaly', 'Edema', 'Consolidation'],
```

To change output location:
```python
'output_base': 'data/processed',
```
