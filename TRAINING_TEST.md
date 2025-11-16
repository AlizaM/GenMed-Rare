# Quick Training Test Setup

## What's Been Created

### 1. Test Configuration (`configs/config_test.yaml`)
- **Model**: Swin Tiny (smaller, faster than Swin Base)
- **Batch size**: 8 (small for quick iterations)
- **Epochs**: 3 (just for testing)
- **Workers**: 2 (fewer for testing)
- **Outputs**: `outputs/test_training_pipeline/`

### 2. Test Dataset Creator (`scripts/create_test_dataset.py`)
Creates a small balanced dataset:
- **Train**: 200 images (100 per class)
- **Val**: 60 images (30 per class)  
- **Test**: 60 images (30 per class)
- **Total**: 320 images (perfectly balanced)

### 3. Training Test Script (`scripts/test_training.py`)
Runs quick training and verifies:
- Model loading
- Data loading
- Training loop
- Checkpoint saving
- TensorBoard logging

## How to Use

### Quick Test (Already Running)
```bash
# 1. Create test dataset
python scripts/create_test_dataset.py

# 2. Run training test
python scripts/test_training.py
```

### Custom Test Dataset Size
```bash
python scripts/create_test_dataset.py \
    --train-samples 50 \
    --val-samples 15 \
    --test-samples 15
```

### Full Training (Production)
```bash
# Use the full dataset and main config
python scripts/train_classifier.py --config configs/config.yaml
```

## What's Being Tested

✅ Configuration loading  
✅ Data pipeline (dataset + dataloaders)  
✅ Model creation (Swin Transformer)  
✅ Optimizer setup (AdamW)  
✅ Learning rate scheduler (ReduceLROnPlateau)  
✅ Training loop  
✅ Validation loop  
✅ Metrics computation  
✅ Checkpoint saving (best + latest)  
✅ TensorBoard logging  
✅ Early stopping logic  

## Current Training Test Status

Training is running on CPU (takes ~3-5 minutes for 3 epochs on 200 images).

Check progress:
```bash
# View live logs
tail -f test_training.log

# Check TensorBoard
tensorboard --logdir=outputs/test_training_pipeline/logs
```

## After Test Completes

Verify checkpoints exist:
```bash
ls -lh outputs/test_training_pipeline/checkpoints/
```

Should see:
- `best_checkpoint.pth`
- `latest_checkpoint.pth`
