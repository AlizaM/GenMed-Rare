# Training Pipeline for Effusion vs Fibrosis Classification

This directory contains a complete training pipeline for binary medical image classification using PyTorch and Swin Transformer.

## Overview

The pipeline consists of the following components:

1. **Configuration Management** (`src/utils/config.py`)
   - YAML-based configuration with dataclass conversion
   - Type-safe configuration handling

2. **Data Handler** (`src/data/data_handler.py`)
   - Loads and processes data from `data/interim/filtered_data_entry.csv`
   - Filters to only Effusion and Fibrosis classes
   - Removes images with both labels
   - Creates stratified 80/20 train/validation split
   - Saves processed CSV with image paths and split information

3. **Dataset & Transforms** (`src/data/dataset.py`, `src/data/transforms.py`)
   - PyTorch Dataset for loading medical images
   - Training augmentations: rotation, brightness/contrast adjustment, Gaussian noise, normalization
   - Validation transforms: resize and normalization only

4. **Model** (`src/models/classifier.py`)
   - Swin Transformer-based classifier
   - Configurable variant (tiny, small, base)
   - Pretrained on ImageNet with custom classification head

5. **Training** (`src/train/trainer.py`)
   - Complete training loop with metrics tracking
   - Checkpoint saving (best model + periodic)
   - Early stopping support
   - Learning rate scheduling (ReduceLROnPlateau)
   - TensorBoard logging

6. **Metrics** (`src/utils/metrics.py`)
   - Accuracy, Precision, Recall, F1-score, AUC-ROC
   - Confusion matrix computation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure you have the filtered data at `data/interim/filtered_data_entry.csv` and images at `data/interim/train_val/`.

### 3. Configure Training

Edit `configs/train_effusion_fibrosis.yaml` to customize:
- Model architecture (Swin variant: tiny, small, base)
- Training hyperparameters (learning rate, batch size, epochs)
- Data augmentation settings
- Logging and checkpointing options

### 4. Run Training

```bash
python scripts/train.py --config configs/train_effusion_fibrosis.yaml
```

#### Command-line Options

- `--config`: Path to configuration file (default: `configs/train_effusion_fibrosis.yaml`)
- `--skip-processing`: Skip data processing step if processed CSV already exists
- `--resume`: Path to checkpoint to resume training from

### 5. Monitor Training

Training progress is logged to:
- Console output
- `training.log` file
- TensorBoard logs in `outputs/logs/`

View TensorBoard:
```bash
tensorboard --logdir outputs/logs
```

## Directory Structure

```
GenMed-Rare/
├── configs/
│   └── train_effusion_fibrosis.yaml    # Training configuration
├── data/
│   ├── interim/
│   │   ├── filtered_data_entry.csv     # Input data
│   │   └── train_val/                   # Training images
│   └── processed/
│       └── effusion_fibrosis_processed.csv  # Processed data (generated)
├── outputs/
│   ├── checkpoints/                     # Model checkpoints
│   │   ├── best_model.pth
│   │   └── checkpoint_epoch_*.pth
│   └── logs/                            # TensorBoard logs
├── scripts/
│   └── train.py                         # Training entry point
└── src/
    ├── data/
    │   ├── data_handler.py              # Data processing
    │   ├── dataset.py                   # PyTorch Dataset
    │   └── transforms.py                # Augmentations
    ├── models/
    │   └── classifier.py                # Model definition
    ├── train/
    │   └── trainer.py                   # Training loop
    └── utils/
        ├── config.py                    # Configuration management
        └── metrics.py                   # Metrics calculation
```

## Configuration

Key configuration sections:

### Experiment Settings
```yaml
experiment:
  name: "effusion_vs_fibrosis"
  seed: 42                              # Random seed for reproducibility
  output_dir: "outputs"
  checkpoint_dir: "outputs/checkpoints"
  log_dir: "outputs/logs"
```

### Data Settings
```yaml
data:
  class_1: "Effusion"                   # First class
  class_2: "Fibrosis"                   # Second class
  train_split: 0.8                      # 80% train, 20% val
  stratify: true                        # Stratified split
  img_size: 224                         # Image size for model input
```

### Model Settings
```yaml
model:
  name: "swin_transformer"
  variant: "base"                       # Options: tiny, small, base
  pretrained: true                      # Use ImageNet pretrained weights
  num_classes: 2                        # Binary classification
```

### Training Settings
```yaml
training:
  epochs: 100
  batch_size: 16
  lr: 1e-4
  optimizer: "adamw"
  scheduler: "reduce_on_plateau"
  early_stopping: true
  early_stopping_patience: 15
```

### Augmentation Settings
```yaml
augmentation:
  train:
    rotation_degrees: 10
    brightness: 0.2
    contrast: 0.2
    gaussian_noise_std: 0.01
    horizontal_flip: false
    vertical_flip: false
```

## Features

- **Stratified Splitting**: Maintains class distribution in train/val splits
- **Class Weighting**: Handles class imbalance with weighted loss
- **Data Augmentation**: Sensible augmentations for medical images
- **Checkpoint Management**: Saves best model and periodic checkpoints
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Comprehensive Metrics**: Tracks accuracy, precision, recall, F1, AUC-ROC
- **TensorBoard Logging**: Visualize training progress
- **Reproducibility**: Fixed random seeds for consistent results

## Metrics Tracked

During training, the following metrics are computed and logged:

- **Loss**: Cross-entropy loss (with class weighting)
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score (binary)
- **Recall**: Recall score (binary)
- **F1-score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

## Output Files

After training, you'll find:

1. **Checkpoints** in `outputs/checkpoints/`:
   - `best_model.pth`: Best model based on validation loss
   - `checkpoint_epoch_*.pth`: Periodic checkpoints every N epochs

2. **Logs** in `outputs/logs/`:
   - TensorBoard event files for visualization

3. **Processed Data** in `data/processed/`:
   - `effusion_fibrosis_processed.csv`: Preprocessed dataset with split information

## Resuming Training

To resume from a checkpoint:

```bash
python scripts/train.py --config configs/train_effusion_fibrosis.yaml --resume outputs/checkpoints/checkpoint_epoch_50.pth
```

## Customization

### Using a Different Model Variant

Edit `configs/train_effusion_fibrosis.yaml`:
```yaml
model:
  variant: "tiny"  # Options: tiny, small, base
```

### Adjusting Learning Rate

```yaml
training:
  lr: 5e-5  # Lower learning rate for fine-tuning
```

### Changing Augmentation

```yaml
augmentation:
  train:
    rotation_degrees: 15  # More aggressive rotation
    brightness: 0.3       # Stronger brightness variation
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Use a smaller model variant (tiny instead of base)

### Training Too Slow
- Use a smaller model variant
- Reduce `num_workers` if CPU-bound
- Reduce image size (though this may hurt performance)

### Class Imbalance
The pipeline automatically handles class imbalance with:
- Stratified splitting
- Class-weighted loss function

## Notes

- Images are expected in `data/interim/train_val/` directory
- The pipeline removes images with both Effusion and Fibrosis labels
- Other labels on images are removed but the images are kept
- Random seed is set for reproducibility
- TensorBoard logs training and validation metrics every epoch
