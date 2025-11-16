# Training Pipeline for Binary Classification

## Overview
Complete PyTorch training pipeline for Effusion vs Fibrosis binary classification using Swin Transformer.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data
Convert interim data into training-ready format:
```bash
python src/data/preprocess.py --config configs/train_effusion_fibrosis.yaml
```

This will:
- Filter images to only Effusion and Fibrosis
- Remove images labeled with both classes
- Create binary labels (0=Effusion, 1=Fibrosis)
- Perform stratified 80/20 train/val split
- Save CSVs to `outputs/processed_data/`

### 3. Train Model
```bash
python scripts/train_classifier.py --config configs/train_effusion_fibrosis.yaml
```

### 4. Monitor Training
Open TensorBoard to view metrics:
```bash
tensorboard --logdir=outputs/logs
```

Then navigate to http://localhost:6006

## Configuration

All settings are in `configs/train_effusion_fibrosis.yaml`:

- **Experiment**: Name, description, random seed
- **Data**: Paths, class labels, split ratio
- **Model**: Swin variant (tiny/small/base), pretrained weights
- **Training**: Batch size, epochs, learning rate, optimizer
- **Augmentation**: Rotation, brightness, contrast, Gaussian noise
- **Metrics**: Tracking accuracy, F1, AUC-ROC, etc.

### Key Model Options

```yaml
model:
  variant: "swin_base_patch4_window7_224"
  # Options: swin_tiny_patch4_window7_224
  #          swin_small_patch4_window7_224  
  #          swin_base_patch4_window7_224
```

### Medical Imaging Augmentations

```yaml
augmentation:
  train:
    rotation_degrees: 10      # Slight rotation only
    brightness: 0.2
    contrast: 0.2
    gaussian_noise_std: 0.01
```

**Note**: NO horizontal/vertical flips (orientation matters in medical imaging!)

## Output Structure

```
outputs/
├── processed_data/
│   ├── train.csv              # Training data
│   ├── val.csv                # Validation data
│   └── combined_train_val.csv # Combined reference
├── checkpoints/
│   └── effusion_vs_fibrosis_baseline/
│       ├── best_checkpoint.pth
│       └── latest_checkpoint.pth
└── logs/
    └── effusion_vs_fibrosis_baseline/
        └── events.out.tfevents...
```

## Metrics Tracked

- **Loss**: CrossEntropyLoss
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive class precision
- **Recall**: Positive class recall (sensitivity)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: True/false positives and negatives

## Features

✅ **Stratified splitting** - Preserves class balance  
✅ **Early stopping** - Prevents overfitting  
✅ **Learning rate scheduling** - ReduceLROnPlateau  
✅ **Checkpointing** - Saves best and latest models  
✅ **TensorBoard logging** - Real-time monitoring  
✅ **Reproducibility** - Fixed random seeds  
✅ **Medical-safe augmentations** - No orientation flips  
✅ **Mixed precision training** - Optional AMP support  

## Resume Training

To resume from a checkpoint:
```bash
python scripts/train_classifier.py \
    --config configs/train_effusion_fibrosis.yaml \
    --resume outputs/checkpoints/effusion_vs_fibrosis_baseline/latest_checkpoint.pth
```

## Troubleshooting

### CUDA out of memory
Reduce batch size in config:
```yaml
training:
  batch_size: 16  # Instead of 32
```

Or use smaller model:
```yaml
model:
  variant: "swin_tiny_patch4_window7_224"
```

### Slow training
Enable mixed precision:
```yaml
training:
  use_amp: true
```

### Class imbalance
The dataset class provides `get_class_weights()` for weighted loss (not currently used but available).

## Advanced Options

### Freeze Backbone
Only train the classification head:
```yaml
model:
  freeze_backbone: true
```

### Change Classes
Edit the config to classify different diseases:
```yaml
data:
  class_positive: "Pneumonia"  # Rare class
  class_negative: "Hernia"     # Common class
```

Then re-run preprocessing and training.

## Next Steps

After training, you can:
1. Evaluate on test set (create `scripts/evaluate.py`)
2. Train generative models to augment rare class
3. Compare performance with synthetic data augmentation
