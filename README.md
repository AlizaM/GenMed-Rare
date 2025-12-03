# GenMed-Rare

A medical computer vision project for improving rare disease classification through synthetic data augmentation. Uses Stable Diffusion with LoRA fine-tuning and conditional GAN to generate synthetic chest X-rays for rare pathology classes from the NIH ChestX-ray14 dataset.

## Project Structure

```
GenMed-Rare/
├── configs/                    # YAML configuration files
├── data/
│   ├── raw/                    # NIH dataset (user-provided)
│   ├── interim/                # Organized by pathology
│   ├── processed/              # Binary classification datasets
│   └── pure_class_folders/     # Single-label images for diffusion
├── src/
│   ├── config/                 # Configuration management
│   ├── data/                   # Dataset and preprocessing
│   ├── models/                 # Model definitions
│   ├── train/                  # Training utilities
│   └── eval/                   # Evaluation metrics
├── scripts/                    # Executable scripts
├── outputs/                    # Training outputs and checkpoints
└── tests/                      # Test suite
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Pre-processed Data (Recommended)

Download pre-processed datasets:
- **Classification**: [Filtered NIH Dataset](https://drive.google.com/file/d/1xBZDLDPtgVHFpFE79ouLlF4xFp1_oXZv/view?usp=drive_link) → Extract to `data/interim/`
- **Diffusion**: [Balanced Dataset](https://drive.google.com/file/d/171Rqd1T97BEnMJ9DPPnxXJ-F85YNJZbR/view?usp=drive_link) → Extract to `data/diffusion_data/diffusion_data_balanced/`

### From Raw NIH Dataset

```bash
# Organize NIH dataset
python scripts/filter_and_organize_data.py

# Preprocess for binary classification
python src/data/preprocess.py --config configs/config.yaml

# Create pure class folders for diffusion training
python scripts/create_pure_class_folders.py --copy-images
```

## Classification

Train a Swin Transformer for binary classification (e.g., Fibrosis vs Effusion):

```bash
# Train classifier
python scripts/train_classifier.py --config configs/config.yaml

# Evaluate model
python scripts/evaluate_model.py --checkpoint outputs/.../best_checkpoint.pth --config configs/config.yaml

# Monitor training
tensorboard --logdir=outputs/<experiment_name>/logs
```

## Image Generation (Diffusion)

Train pathology-specific generators using prior-based learning:

```bash
# Train fibrosis generator
python scripts/train_diffusion_prior.py --config configs/config_diffusion_fibrosis.yaml

# Resume from checkpoint
python scripts/train_diffusion_prior.py --config configs/config_diffusion_fibrosis.yaml --resume-latest

# Generate synthetic images
python scripts/generate_xrays.py \
    --checkpoint outputs/.../checkpoint-6500 \
    --output-dir outputs/generated_images \
    --num-images 100 \
    --pathology Fibrosis
```

## Evaluation

```bash
# Evaluate generated images (novelty, pathology confidence, diversity)
python scripts/evaluate_checkpoints.py --config configs/config_eval_fibrosis.yaml

# Comprehensive evaluation
python scripts/evaluate_diffusion_generation.py \
    --generated-dir outputs/.../images \
    --real-dir data/pure_class_folders/fibrosis \
    --label Fibrosis \
    --preset full
```

## GAN-Based Rare-Class Augmentation (- Pujitha)

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

### GAN Training

```bash
python scripts/gan_implementation.py \
    --epochs 25 \
    --batch-size 32 \
    --output-dir outputs/cgan_rare_aug/
```

Synthetic GAN augmented images will be saved to `outputs/cgan_rare_aug/samples/`

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Test specific component
pytest tests/test_config.py -v
```

## License

MIT License
