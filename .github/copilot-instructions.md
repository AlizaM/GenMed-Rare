# GenMed-Rare AI Agent Instructions

## Project Overview
Medical image classification improvement through generative model augmentation. Binary classification experiments comparing rare disease detection with/without synthetic data augmentation using NIH Chest X-ray dataset.

## Core Architecture

### Data Flow (Multi-Stage Pipeline)
1. **Raw → Interim** (`scripts/filter_and_organize_data.py`): Extract NIH archive → filter by target labels → organize into `data/interim/{train_val,test}/{Label}/`
2. **Interim → Processed** (planned): Binary classification prep → stratified split → `outputs/processed_data/{train,val}.csv`
3. **Training** (planned): PyTorch classifier training with TensorBoard logging → checkpoints to `outputs/checkpoints/`
4. **Generation** (planned): Train generative models → synthetic samples to `outputs/samples/`
5. **Evaluation** (planned): Compare classifier performance across augmentation percentages

### Critical Design Decisions
- **Multi-label handling**: Images can have multiple pathologies (pipe-separated: `"Hernia|Pneumonia"`). Images appear in ALL relevant label folders during organization.
- **Binary classification focus**: Train one rare class vs one common class at a time (e.g., Fibrosis vs Effusion). Set via `configs/config.py` `CLASS_RARE` and `CLASS_COMMON`.
- **No horizontal/vertical flips**: Medical imaging constraint—orientation matters. Only use rotation (±10°), brightness/contrast, Gaussian noise.
- **Stratified splits required**: Use `STRATIFIED=True` to preserve class balance in 80/20 train/val splits.

## Code Conventions

### Path Management
**Always use `pathlib.Path`** instead of `os.path`. Example:
```python
# Good
DATA_DIR = Path("data/interim")
csv_path = DATA_DIR / "filtered_data_entry.csv"

# Bad
import os
csv_path = os.path.join("data/interim", "filtered_data_entry.csv")
```

### Configuration Pattern
Central config in `configs/config.py` as class attributes (not YAML for training). Access via:
```python
from configs.config import Config
config = Config()
config.create_dirs()  # Always call before training
```

### Multi-Label CSV Structure
- `Image Index`: Filename (e.g., `00000003_000.png`)
- `Finding Labels`: Pipe-separated labels (e.g., `"Effusion|Consolidation"`)
- Filter logic: Keep images with **at least one** target label, remove non-target labels from string

## Key Workflows

### Data Preparation
```bash
# 1. Download NIH dataset to data/raw/archive.zip (manual or Kaggle CLI)
# 2. Extract and filter (creates data/interim/)
python scripts/filter_and_organize_data.py

# Output: data/interim/{train_val,test}/{Hernia,Pneumonia,Fibrosis,Effusion}/
#         data/interim/filtered_data_entry.csv
```

### Training Setup (Planned)
```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Configure binary classification in configs/config.py
# Set CLASS_RARE and CLASS_COMMON

# 3. Preprocess for training (creates train/val split CSVs)
python src/data/preprocess.py  # TODO: implement

# 4. Train classifier
python scripts/train_classifier.py  # TODO: implement

# 5. Monitor with TensorBoard
tensorboard --logdir=outputs/logs
```

## Critical Implementation Notes

### For Dataset Classes
- Convert grayscale X-rays to RGB (3 channels) for pretrained models: `Image.open(path).convert('RGB')`
- Apply ImageNet normalization: `MEAN=[0.485,0.456,0.406]`, `STD=[0.229,0.224,0.225]`
- Training augmentations: `RandomRotation(10)`, `ColorJitter(brightness=0.2, contrast=0.2)`, custom `AddGaussianNoise(std=0.01)`
- **Never** use `RandomHorizontalFlip` or `RandomVerticalFlip`

### For Training Scripts
- Set `RANDOM_SEED=42` for reproducibility: call `torch.manual_seed()`, `np.random.seed()`, `torch.backends.cudnn.deterministic=True`
- Save both `best_checkpoint.pth` (best val accuracy) and `latest_checkpoint.pth`
- Use `ReduceLROnPlateau(patience=5, factor=0.5)` scheduler
- Implement early stopping with `EARLY_STOPPING_PATIENCE=10`
- Log to TensorBoard: train/val loss, accuracy, precision, recall, F1, AUC-ROC

### For Model Creation
- Default backbone: `resnet50` (configurable: resnet18/34/50/101)
- Always use pretrained ImageNet weights: `models.resnet50(pretrained=True)`
- Replace final layer: `model.fc = nn.Linear(num_features, 2)` for binary classification

## File Organization Rules

### When creating new modules:
- **Datasets**: `src/data/dataset.py` (PyTorch Dataset classes)
- **Models**: `src/models/classifier.py` (model builders), `src/models/gan.py`, etc.
- **Training**: `src/train/trainer.py` (training loops), `src/train/metrics.py`
- **Entry points**: `scripts/train_classifier.py`, `scripts/generate.py`, `scripts/evaluate.py`
- **Configs**: Per-experiment YAMLs in `configs/`, shared Python config in `configs/config.py`

### Directory Creation
Always use `Path.mkdir(parents=True, exist_ok=True)` or `Config.create_dirs()`

## Experiment Protocol

Goal: Compare classifier performance with 0%, 10%, 25%, 50% synthetic augmentation.

1. Train baseline classifier (0% augmentation)
2. Train generative model on rare class
3. Generate synthetic samples at each percentage
4. Train classifiers with augmented datasets
5. Evaluate on fixed test set (never augmented)
6. Report mean/std across 3-5 random seeds
7. Statistical tests (t-test) comparing to baseline

## Common Pitfalls

- **Don't** assume images are in flat directories—use label subfolders from `data/interim/`
- **Don't** use `os.path`—use `pathlib.Path` throughout
- **Don't** apply horizontal/vertical flips to medical images
- **Don't** forget to set random seeds before splitting data
- **Don't** augment the test set—only train/val
- **Don't** train on images with both target classes simultaneously (binary only)

## Dependencies
Core: `torch`, `torchvision`, `pandas`, `Pillow`, `PyYAML`, `tqdm`, `scikit-image`, `scikit-learn`
