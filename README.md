# GenMed-Rare

A PyTorch-based medical computer vision project focused on improving rare disease classification through synthetic data augmentation. The pipeline integrates two complementary generative approaches — Stable Diffusion with LoRA fine-tuning and a conditional GAN (cGAN) — to generate synthetic chest X-rays for rare pathology classes, with comprehensive binary and three-class classification experiments on the NIH ChestX-ray14 dataset.

## Project Overview

This project addresses the challenge of medical image classification for rare diseases by synthesizing additional training data using generative models. The pipeline compares classification performance on rare cases (e.g., Fibrosis) vs common cases (e.g., Effusion) with and without synthetic data augmentation from both diffusion models and a conditional GAN, evaluating their downstream impact on rare-class sensitivity.

## Current Implementation Status - Diffusion based augmentation (- Aliza) ✅

### ✅ Prior-based Diffusion Training
- **Medical Model**: `danyalmalik/stable-diffusion-chest-xray` (pre-trained on chest X-rays)
- **Custom VAE**: `stabilityai/sd-vae-ft-mse` (fine-tuned for detail preservation)
- **Prior-based Learning**: Each pathology image paired with healthy "prior" images
- **LoRA Fine-tuning**: 13.3M trainable parameters with PEFT integration
- **Pure Class Data**: 551 fibrosis + 234 pneumonia + 1,000 healthy images (train/val only)
- **Conditional Training**: "a chest x-ray with fibrosis" vs "a chest x-ray"
- **Robust Checkpointing**: Frequent checkpoints (every 250 steps) with validation safety

### ✅ Complete Diffusion Pipeline  
- **Training**: HuggingFace Diffusers with mixed precision, gradient checkpointing
- **Resume Support**: Full checkpoint resume functionality with automatic latest detection
- **Validation**: Real-time sample generation with error handling
- **Output**: Production-ready LoRA adapters for medical image synthesis
- **Memory Optimized**: Works on 8GB VRAM with batch size optimizations

### ✅ Pure Class Dataset Organization
- **Data Extraction**: Pure single-pathology images from NIH dataset
- **Train/Val Split**: Respects official NIH train_val_list.txt (excludes test images)
- **Healthy Images**: Curated from diffusion dataset (verified "No Finding" labels)
- **Folder Structure**: Organized by pathology class for easy access

### ✅ Classification Pipeline
- **Dataset**: NIH Chest X-ray (filtered for target pathologies)
- **Task**: Binary classification (configurable rare vs common disease pairs)
- **Model**: Swin Transformer with medical-safe augmentations
- **Training**: PyTorch with TensorBoard logging, early stopping, and robust checkpointing
- **Evaluation**: Comprehensive metrics including precision, recall, F1, and AUC-ROC

### 🔬 Research Focus
- **Synthetic Augmentation**: Generate rare pathology cases using diffusion models
- **Performance Analysis**: Compare classification metrics with 0%, 10%, 25%, 50% synthetic augmentation
- **Medical Safety**: Orientation-preserving augmentations (no horizontal/vertical flips)
- **Statistical Rigor**: Multi-seed experiments with statistical significance testing

## GAN-Based Rare-Class Augmentation (- Pujitha) ✅

A complete adversarial augmentation pipeline has been added to complement the diffusion model by offering a more controlled, anatomy-preserving synthetic data strategy for rare pathologies.

Conditional GAN Training for Rare Pathologies
	•	Architecture: Lightweight convolutional cGAN
	•	Conditioning: Class labels appended as channels
	•	Training Dataset: Pure Fibrosis and Pneumonia cohorts (same as diffusion)
	•	Output:
	•	400 synthetic Fibrosis images
	•	400 synthetic Pneumonia images
	•	Total of 800 rare-class synthetic images for downstream augmentation

Stabilization Techniques
	•	One-sided label smoothing (y_real = 0.9)
	•	Gradient clipping to prevent exploding gradients
	•	Early stopping based on discriminator plateau
	•	Mode-collapse monitoring via feature diversity tracking
	•	Training logs confirm stable adversarial dynamics

Feature-Space Validation
	•	ResNet-50 embedding + t-SNE visualization
	•	GAN samples partially overlap with real data manifold
	•	Indicates realistic pathological support without drifting too far from the rare-class distribution

Three-Class Downstream Evaluation

A three-class Swin-T classifier (Fibrosis / Pneumonia / Effusion) was trained under three regimes:
	1.	Baseline Cross-Entropy
	2.	Tempered Class-Weighted Cross-Entropy
	3.	GAN + Class-Weighted Cross-Entropy

Key Results
	•	Fibrosis recall improves significantly:
	•	0.2644 → 0.3563 (with class weighting)
	•	0.3563 → 0.4552 (with GAN augmentation)
	•	Pneumonia recall remains more than double the baseline
	•	Macro F1 remains competitive (0.5223)
	•	Accuracy reduction modest (0.8252 → 0.7755)

These results demonstrate that carefully regularized cGAN augmentation can meaningfully improve rare-class sensitivity, especially for Fibrosis.

## Repository Structure

```
GenMed-Rare/
├── configs/                    # YAML configuration files
│   ├── config.yaml            # Main training configuration (Effusion vs Fibrosis)
│   ├── config_diffusion.yaml  # Original diffusion model training
│   ├── config_diffusion_fibrosis.yaml # NEW: Prior-based fibrosis training (20 epochs)
│   ├── config_diffusion_test.yaml     # NEW: Quick test config (1 epoch, fast validation)
│   └── config_test.yaml       # Quick test configuration (small dataset)
├── data/
│   ├── raw/                   # Raw data (download separately)
│   │   ├── archive.zip        # NIH Chest X-ray archive (112,120 images)
│   │   ├── Data_Entry_2017.csv # Original labels CSV
│   │   ├── train_val_list.txt # Train/val split file
│   │   └── test_list.txt      # Test split file
│   ├── interim/               # Organized data (created by scripts/filter_and_organize_data.py)
│   │   ├── train_val/         # Training + validation images organized by label
│   │   │   ├── Hernia/
│   │   │   ├── Pneumonia/
│   │   │   ├── Fibrosis/
│   │   │   └── Effusion/
│   │   ├── test/              # Test images organized by label
│   │   │   ├── Hernia/
│   │   │   ├── Pneumonia/
│   │   │   ├── Fibrosis/
│   │   │   └── Effusion/
│   │   └── filtered_data_entry.csv  # Filtered labels CSV
│   ├── processed/             # Preprocessed data (created by src/data/preprocess.py)
│   │   └── effusion_fibrosis/
│   │       ├── dataset.csv         # Unified dataset with split column
│   │       └── dataset_test.csv    # Small test dataset (320 images)
│   ├── pure_class_folders/    # Pure single-pathology datasets (train/val only)
│   │   ├── fibrosis_images.csv    # 551 pure fibrosis images metadata
│   │   ├── fibrosis/              # Pure fibrosis image files
│   │   ├── pneumonia_images.csv   # 234 pure pneumonia images metadata  
│   │   ├── pneumonia/             # Pure pneumonia image files
│   │   ├── healthy_images.csv     # 1,000 healthy images metadata
│   │   └── healthy/               # Healthy image files (from diffusion dataset)
│   └── diffusion_data/        # Diffusion training data
│       └── diffusion_data_balanced/  # Balanced dataset for diffusion training
│           ├── diffusion_dataset_balanced.csv  # Image metadata
│           └── *.png          # Chest X-ray images
├── outputs/                   # Training outputs (experiment-specific)
│   ├── <experiment_name>/     # e.g., effusion_vs_fibrosis_baseline/ (classification)
│   │   ├── checkpoints/       # Model checkpoints
│   │   │   ├── best_checkpoint.pth
│   │   │   └── latest_checkpoint.pth
│   │   ├── logs/              # TensorBoard logs
│   │   └── dataset_summary.csv # Dataset statistics
│   └── diffusion_models/     # Diffusion training outputs
│       └── <experiment_name>/ # e.g., sd15_lora_fibrosis/
│           ├── checkpoints/   # LoRA model checkpoints
│           ├── logs/          # Training logs
│           └── samples/       # Generated validation images
├── src/                       # Source code
│   ├── config/               # Configuration management
│   │   ├── __init__.py
│   │   └── config_manager.py # Dataclass-based config with type safety
│   ├── data/                 # Data pipeline
│   │   ├── __init__.py
│   │   ├── dataset.py        # PyTorch Dataset with medical augmentations
│   │   ├── diffusion_dataset.py # Diffusion training dataset
│   │   └── preprocess.py     # Binary classification data preparation
│   ├── models/               # Model definitions
│   │   ├── __init__.py
│   │   └── classifier.py     # Swin Transformer classifier
│   └── train/                # Training utilities
│       ├── __init__.py
│       └── trainer.py        # Training loop with metrics, logging, checkpointing
├── scripts/                   # Executable scripts
│   ├── filter_and_organize_data.py  # Extract & organize NIH dataset
│   ├── train_classifier.py          # Classification training entry point
│   ├── train_diffusion.py           # Original diffusion model training
│   ├── train_diffusion_prior.py     # NEW: Prior-based diffusion training for pathologies
│   ├── create_pure_class_folders.py # NEW: Create pure class datasets from NIH data
│   ├── test_training_diffusion.py   # Diffusion pipeline validation
│   ├── test_resume_demo.py          # Checkpoint resume demonstration
│   ├── create_test_dataset.py       # Create small test dataset
│   └── test_training.py             # Quick classification training test
│   └── diagnose_missing_images.py   # Debug missing image files
    |__ gan_implementation.py        # Complete GAN based implementation of the project
├── tests/                     # Comprehensive pytest test suite (174 tests)
│   ├── test_config.py        # Configuration management tests
│   ├── test_dataloader.py    # Dataset and dataloader tests
│   ├── test_diffusion_dataset.py    # Diffusion dataset tests
│   ├── test_diffusion_training.py   # Diffusion training pipeline tests  
│   ├── test_diffusion_resume.py     # Checkpoint resume functionality tests (21 tests)
│   ├── test_evaluation.py    # Model evaluation tests
│   ├── test_environment.py   # Environment and dependency tests
│   └── test_trainer_logging.py      # Training and logging tests
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── TRAINING_TEST.md           # Quick training test documentation
```

## Data Setup

### Option 1: Pre-processed Data (Recommended for Quick Start)

Skip the NIH dataset download and use pre-processed data directly:

#### For Classification Training:
- **Filtered NIH Dataset** (`data/interim/`)
  - Pre-filtered for Hernia, Pneumonia, Fibrosis, Effusion
  - Already organized into train_val/ and test/ directories
  - **Size**: ~14,627 images
  - **Download**: https://drive.google.com/file/d/1xBZDLDPtgVHFpFE79ouLlF4xFp1_oXZv/view?usp=drive_link
  - Extract to: `data/interim/`

#### For Diffusion Training:
- **Balanced Diffusion Dataset** (`data/diffusion_data/`)
  - Balanced across all 15 pathology labels
  - Includes `diffusion_dataset_balanced.csv` metadata
  - **Size**: 10,541 images
  - **Download**: https://drive.google.com/file/d/171Rqd1T97BEnMJ9DPPnxXJ-F85YNJZbR/view?usp=drive_link
  - Extract to: `data/diffusion_data/diffusion_data_balanced/`

### Option 2: Full NIH Dataset (For Custom Filtering)

Download the full NIH Chest X-ray dataset and process it yourself:

1. **`archive.zip`** - NIH Chest X-ray dataset (112,120 PNG images, ~45GB)
   - Download from: [Kaggle NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data)
   - Place at: `data/raw/archive.zip`

2. **`Data_Entry_2017.csv`** - Original labels file
   - Included in the archive or download separately
   - Place at: `data/raw/Data_Entry_2017.csv`

3. **`train_val_list.txt`** and **`test_list.txt`** - Official train/test split files
   - Defines which images belong to train_val vs test sets
   - Place at: `data/raw/train_val_list.txt` and `data/raw/test_list.txt`

Then run the processing scripts:

```bash
# Extract and organize NIH dataset (creates data/interim/)
python scripts/filter_and_organize_data.py

# Create balanced diffusion dataset (creates data/diffusion_data/)
python scripts/prepare_diffusion_dataset.py
python scripts/balance_diffusion_dataset.py --apply
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

#### Option A: Using Pre-processed Data
If you downloaded the pre-processed datasets:

```bash
# For classification: Extract filtered data to data/interim/
# For diffusion: Extract balanced data to data/diffusion_data/diffusion_data_balanced/

# Then run binary classification preprocessing
python src/data/preprocess.py --config configs/config.yaml
```

#### Option B: Processing from Raw NIH Dataset
If you downloaded the full NIH dataset:

```bash
# Step 1: Extract and organize NIH dataset (creates data/interim/)
python scripts/filter_and_organize_data.py

# Step 2: Preprocess for binary classification (creates data/processed/)
python src/data/preprocess.py --config configs/config.yaml

# Step 3 (Optional): Create balanced diffusion dataset
python scripts/prepare_diffusion_dataset.py
python scripts/balance_diffusion_dataset.py --apply
```

### 3. Training

#### Classification Training

```bash
# Quick test with small dataset (recommended first)
python scripts/create_test_dataset.py  # Creates 320-image test dataset
python scripts/test_training.py        # Runs 3-epoch test (~5 minutes on CPU)

# Full training
python scripts/train_classifier.py --config configs/config.yaml

# Monitor with TensorBoard
tensorboard --logdir=outputs/<experiment_name>/logs
```

#### Prior-Based Diffusion Training (NEW)

Train pathology-specific diffusion models using healthy images as "priors":

```bash
# 1. Create pure class datasets from NIH data
python scripts/create_pure_class_folders.py --copy-images

# This creates:
# data/pure_class_folders/fibrosis/     (551 pure fibrosis images)
# data/pure_class_folders/pneumonia/    (234 pure pneumonia images)  
# data/pure_class_folders/healthy/      (1,000 healthy images)

# 2. Train fibrosis generator with prior-based learning
python scripts/train_diffusion_prior.py --config configs/config_diffusion_fibrosis.yaml

# 3. Resume training from checkpoint
python scripts/train_diffusion_prior.py --config configs/config_diffusion_fibrosis.yaml --resume-latest

# 4. Quick test (fast validation of setup)
python scripts/train_diffusion_prior.py --config configs/config_diffusion_test.yaml

# 5. Monitor progress  
tensorboard --logdir=outputs/diffusion_fibrosis_prior/logs
```

**Prior-Based Training Features:**
- ✅ **Medical Model**: `danyalmalik/stable-diffusion-chest-xray` (chest X-ray specific)
- ✅ **Custom VAE**: `stabilityai/sd-vae-ft-mse` (fine-grained detail preservation)
- ✅ **Prior Learning**: Each fibrosis image paired with different healthy "priors"
- ✅ **Conditioning**: "a chest x-ray with fibrosis" vs "a chest x-ray"
- ✅ **Data Repeats**: 10× repetitions (551 → 5,510 training samples)
- ✅ **Robust Checkpointing**: Every 250 steps with validation safety
- ✅ **Error Handling**: Validation failures won't stop training
- ✅ **Memory Optimized**: Batch size 6 for 8GB VRAM

**Training Progress:**
- **Duration**: ~31 hours for full 20 epochs (27,560 steps)
- **Checkpoints**: Auto-saved every 250 steps (110 total checkpoints)  
- **Validation**: Generated images every 1,000 steps (4 samples each)
- **Resume Support**: Full state preservation with step-exact continuation

#### Standard Diffusion Training

**Prerequisites**: 
- Balanced diffusion dataset in: `data/diffusion_data/diffusion_data_balanced/`
- CSV metadata: `data/diffusion_data/diffusion_data_balanced/diffusion_dataset_balanced.csv`
- **GPU Requirements**: 8GB+ VRAM recommended (RTX 2070 or better)

```bash
# Quick diffusion pipeline validation (recommended first)
python scripts/test_training_diffusion.py --no-training  # Validates setup without training

# Train Stable Diffusion LoRA model from scratch
python scripts/train_diffusion.py --config configs/config_diffusion.yaml

# Resume training from latest checkpoint (if available)
python scripts/train_diffusion.py --config configs/config_diffusion.yaml --resume-latest

# Resume from specific checkpoint
python scripts/train_diffusion.py --config configs/config_diffusion.yaml \
    --resume outputs/diffusion_models/sd15_lora_fibrosis/checkpoints/checkpoint-step-5000

# List available checkpoints and verify integrity
python scripts/test_resume_demo.py

# Monitor training progress
tensorboard --logdir=outputs/diffusion_models/sd15_lora_fibrosis/logs
```

**Checkpoint Resume Features:**
- ✅ **Automatic Latest Detection**: `--resume-latest` finds and resumes from most recent checkpoint
- ✅ **Manual Selection**: `--resume path/to/checkpoint` resumes from specific checkpoint  
- ✅ **Step Tracking**: Properly continues from exact training step (not just epoch)
- ✅ **LoRA Loading**: Correctly loads PEFT adapter weights and training state
- ✅ **Validation**: Comprehensive checkpoint integrity verification
- ✅ **Error Handling**: Robust handling of corrupted or missing checkpoints

**Current Checkpoint Status:**
- **Latest Available**: Step 7,000 (if previous training exists)
- **Total Checkpoints**: 16 valid checkpoints available for resume
- **Checkpoint Types**: Both step-based and epoch-based supported

**Training Configuration:**
- **Model**: Stable Diffusion 1.5 + LoRA adapters (16 rank, 32 alpha)
- **Optimization**: AdamW with cosine annealing, gradient accumulation
- **Memory**: Mixed precision (fp16), gradient checkpointing for efficiency  
- **Validation**: Real-time sample generation, FID score tracking
- **Safety**: Medical-appropriate data augmentations (no orientation flips)

### 4. Testing & Validation

```bash
# Run comprehensive test suite (174 tests)
pytest tests/ -v

# Test specific components
pytest tests/test_diffusion_resume.py -v     # Checkpoint resume functionality (21 tests)
pytest tests/test_diffusion_training.py -v   # Diffusion training pipeline (14 tests) 
pytest tests/test_diffusion_dataset.py -v    # Diffusion dataset (17 tests)
pytest tests/test_config.py -v               # Configuration management
pytest tests/test_dataloader.py -v           # Dataset and dataloader functionality
pytest tests/test_evaluation.py -v           # Model evaluation metrics

# Quick validation scripts
python scripts/test_training_diffusion.py --no-training  # Diffusion setup validation
python scripts/test_training.py                          # Classification training test
python scripts/test_resume_demo.py                       # Checkpoint resume demo
```
## GAN Training Quick Start
```bash
python scripts/gan_implementation.py \
    --epochs 25 \
    --batch-size 32 \
    --output-dir outputs/cgan_rare_aug/
```

### Synthetic GAN augmented images will be saved to - outputs/cgan_rare_aug/samples/


## Key Features

### 🚀 Production-Ready Diffusion Training
- **Stable Diffusion 1.5** with LoRA fine-tuning for efficient medical image generation
- **Checkpoint Resume**: Robust training continuation from any checkpoint
- **Memory Optimized**: Mixed precision, gradient checkpointing, gradient accumulation
- **Medical Safety**: Orientation-preserving augmentations for medical imaging standards
- **Comprehensive Logging**: TensorBoard integration with loss tracking and sample generation

### 📊 Advanced Classification Pipeline  
- **Swin Transformer**: State-of-the-art vision transformer for medical image classification
- **Medical Augmentations**: Rotation, brightness/contrast, Gaussian noise (no harmful flips)
- **Multi-Class Support**: Configurable binary classification for any rare vs common disease pair
- **Robust Training**: Early stopping, learning rate scheduling, comprehensive metrics

### 🔬 Research Infrastructure
- **Experiment Management**: Auto-generated experiment names with timestamp and configuration
- **Reproducibility**: Deterministic training with configurable random seeds
- **Statistical Analysis**: Multi-seed experiment support for rigorous evaluation
- **Data Integrity**: Comprehensive validation and error checking throughout pipeline

### ✅ Quality Assurance
- **174 Test Cases**: Comprehensive pytest coverage for all components
- **CI/CD Ready**: Automated testing for model training, data loading, and configuration
- **Error Handling**: Robust exception handling with informative error messages
- **Documentation**: Extensive inline documentation and configuration examples

## Research Applications

This codebase supports several research directions in medical AI:

1. **Synthetic Data Augmentation**: Quantify improvement from generative model augmentation
2. **Rare Disease Classification**: Address class imbalance through targeted data synthesis  
3. **Medical Image Quality**: Evaluate synthetic vs real medical image utility
4. **Transfer Learning**: Study domain adaptation between synthetic and real medical images
5. **Fairness & Bias**: Analyze model performance across different patient demographics

## Performance Benchmarks

### Classification Baseline Results
- **Effusion vs Fibrosis**: 92.5% accuracy, 0.89 F1-score (baseline without augmentation)
- **Training Time**: ~45 minutes for full training on RTX 2070 with Max-Q Design
- **Memory Usage**: ~6GB GPU memory for batch_size=32

### Diffusion Training Performance  
- **Training Speed**: ~3-4 hours per epoch (10,541 images) on RTX 2070 with Max-Q Design
- **Memory Usage**: ~7.5GB GPU memory with mixed precision and gradient checkpointing
- **Checkpoint Size**: ~16MB per LoRA checkpoint (vs ~5GB for full model fine-tuning)
- **Generation Speed**: ~2-3 seconds per 512×512 image

## Dependencies

Core requirements (see `requirements.txt` for complete list):

```python
# Core ML Libraries
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
diffusers>=0.21.0
accelerate>=0.24.0
peft>=0.6.0

# Data & Visualization  
pandas>=2.0.0
numpy>=1.24.0
pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Training & Logging
tensorboard>=2.14.0
tqdm>=4.65.0
pyyaml>=6.0.0
timm>=0.9.0

# Testing & Validation
pytest>=7.4.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
```

## Quick Start: Fibrosis Generator Training

To immediately start training a fibrosis-specific diffusion model:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Create pure class datasets
python scripts/create_pure_class_folders.py --copy-images
# ✅ Creates 551 fibrosis + 234 pneumonia + 1,000 healthy images

# 3. Start fibrosis training (full 20 epochs)
python scripts/train_diffusion_prior.py --config configs/config_diffusion_fibrosis.yaml

# 4. Monitor progress
tensorboard --logdir=outputs/diffusion_fibrosis_prior/logs
```

**Expected Training:**
- **Duration**: ~31 hours (27,560 steps)
- **First checkpoint**: Step 250 (~17 minutes) 
- **First validation**: Step 1,000 (~1.1 hours)
- **Memory usage**: ~6-7GB VRAM
- **Output**: LoRA weights in `outputs/diffusion_fibrosis_prior/lora_weights/`

**Resume Training:**
```bash
# Resume from latest checkpoint (if interrupted)
python scripts/train_diffusion_prior.py --config configs/config_diffusion_fibrosis.yaml --resume-latest

# Resume from specific step
python scripts/train_diffusion_prior.py --config configs/config_diffusion_fibrosis.yaml --resume-step 5000
```

---

## Contributing

1. **Code Quality**: Follow PEP 8, add type hints, maintain test coverage
2. **Medical Safety**: Preserve image orientation, validate augmentation appropriateness
3. **Testing**: Add tests for new features, maintain 90%+ coverage
4. **Documentation**: Update README and docstrings for new components
5. **Configuration**: Use YAML configs for new experiments, avoid hardcoded parameters

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{genmed_rare_2025,
  title={Improving Rare Disease Classification through Synthetic Data Augmentation with Stable Diffusion},
  author={Your Name},
  journal={Medical Image Analysis},
  year={2025},
  note={GenMed-Rare: A PyTorch framework for medical image synthesis and classification}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NIH Clinical Center**: For providing the Chest X-ray dataset
- **HuggingFace**: For the diffusers library and model hosting  
- **PyTorch Team**: For the deep learning framework
- **Medical AI Community**: For advancing responsible AI in healthcare


