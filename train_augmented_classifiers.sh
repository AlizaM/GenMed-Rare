#!/bin/bash
#SBATCH --job-name=train_aug_clf
#SBATCH --partition=gpunodes
#SBATCH --constraint=RTX_A4500
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=8:00:00
#SBATCH --mail-user=alizat@cs.toronto.edu
#SBATCH --mail-type=ALL

# ============================================================================
# Train Classifiers on Augmented Data (Fibrosis + Pneumonia)
# ============================================================================
# This script trains TWO classifiers sequentially:
# 1. Effusion vs Fibrosis (with augmented Fibrosis images)
# 2. Effusion vs Pneumonia (with augmented Pneumonia images)
#
# Strategy: Copy entire project + data to scratch, maintaining relative paths.
# This ensures all CSV paths resolve correctly.
# ============================================================================

set -e  # Exit on any error

# ============================================================================
# CONFIGURATION
# ============================================================================
USER_HOME_BASE="/u/alizat"
PROJECT_NAME="GenMed-Rare"
PROJECT_DIR="$USER_HOME_BASE/cv_project/$PROJECT_NAME"

# Track venv location on scratch
VENV_LOCATION_FILE="$PROJECT_DIR/.scratch_venv_location"

# HDD storage base (for persistent outputs)
W_STORAGE_BASE="/w/20251/alizat"

# Scratch space for fast I/O
SCRATCH_BASE="/scratch/ssd004/scratch/alizat"

# Job-specific directories
JOB_ID="${SLURM_JOB_ID:-local}"
SCRATCH_PROJECT_DIR="$SCRATCH_BASE/classifier_training_${JOB_ID}/$PROJECT_NAME"
OUTPUT_DIR="$W_STORAGE_BASE/classifier_outputs/augmented_training_${JOB_ID}"

# Source augmented data paths (on /w/)
FIBROSIS_AUGMENTED_SRC="$W_STORAGE_BASE/data/augmented_data/fibrosis"
PNEUMONIA_AUGMENTED_SRC="$W_STORAGE_BASE/data/augmented_data/pneumonia"

# Config files (relative to project)
FIBROSIS_CONFIG="configs/config_augmented_fibrosis.yaml"
PNEUMONIA_CONFIG="configs/config_augmented_pneumonia.yaml"

# Training script (relative to project)
TRAINING_SCRIPT="scripts/train_classifier.py"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================
echo "============================================================================"
echo "Training Augmented Classifiers (Fibrosis + Pneumonia)"
echo "============================================================================"
echo "Job ID: $JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Check venv exists
if [ ! -f "$VENV_LOCATION_FILE" ]; then
    echo "ERROR: Venv location file not found: $VENV_LOCATION_FILE"
    exit 1
fi

VENV_DIR=$(cat "$VENV_LOCATION_FILE")
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Venv not found at $VENV_DIR"
    exit 1
fi
echo "✓ Venv verified at: $VENV_DIR"

# Check config files exist in home project
if [ ! -f "$PROJECT_DIR/$FIBROSIS_CONFIG" ]; then
    echo "ERROR: Fibrosis config not found: $PROJECT_DIR/$FIBROSIS_CONFIG"
    exit 1
fi
if [ ! -f "$PROJECT_DIR/$PNEUMONIA_CONFIG" ]; then
    echo "ERROR: Pneumonia config not found: $PROJECT_DIR/$PNEUMONIA_CONFIG"
    exit 1
fi
echo "✓ Config files verified"

# Check augmented data directories on /w/
if [ ! -d "$FIBROSIS_AUGMENTED_SRC" ]; then
    echo "ERROR: Fibrosis augmented data not found: $FIBROSIS_AUGMENTED_SRC"
    echo "Please run create_augmented_dataset.py first"
    exit 1
fi
if [ ! -d "$PNEUMONIA_AUGMENTED_SRC" ]; then
    echo "ERROR: Pneumonia augmented data not found: $PNEUMONIA_AUGMENTED_SRC"
    echo "Please run create_augmented_dataset.py first"
    exit 1
fi
echo "✓ Augmented data directories verified"
echo ""

# ============================================================================
# GPU INFORMATION
# ============================================================================
echo "GPU Information:"
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv,noheader
echo ""

# ============================================================================
# COPY PROJECT TO SCRATCH
# ============================================================================
echo "==> Copying project to scratch for fast I/O..."

# Create scratch directory
mkdir -p "$SCRATCH_PROJECT_DIR"

# Copy project code (excluding large files and caches)
echo "  Copying project code..."
rsync -a --exclude='.git' \
         --exclude='__pycache__' \
         --exclude='*.pyc' \
         --exclude='.venv' \
         --exclude='outputs' \
         --exclude='*.pth' \
         --exclude='*.pt' \
         "$PROJECT_DIR/" "$SCRATCH_PROJECT_DIR/"
echo "    ✓ Project code copied"

# Copy interim data (real images)
echo "  Copying interim data (real images)..."
if [ -d "$PROJECT_DIR/data/interim" ]; then
    mkdir -p "$SCRATCH_PROJECT_DIR/data"
    cp -r "$PROJECT_DIR/data/interim" "$SCRATCH_PROJECT_DIR/data/"
    INTERIM_COUNT=$(find "$SCRATCH_PROJECT_DIR/data/interim" -name "*.png" 2>/dev/null | wc -l)
    echo "    ✓ $INTERIM_COUNT interim images copied"
else
    echo "    WARNING: No interim data found at $PROJECT_DIR/data/interim"
fi

# Copy processed data (for val/test splits)
echo "  Copying processed data (val/test CSVs)..."
if [ -d "$PROJECT_DIR/data/processed" ]; then
    cp -r "$PROJECT_DIR/data/processed" "$SCRATCH_PROJECT_DIR/data/"
    echo "    ✓ Processed data copied"
fi

# Copy augmented data from /w/ to scratch data folder
echo "  Copying augmented Fibrosis data..."
mkdir -p "$SCRATCH_PROJECT_DIR/data/augmented_data"
cp -r "$FIBROSIS_AUGMENTED_SRC" "$SCRATCH_PROJECT_DIR/data/augmented_data/"
FIB_IMG_COUNT=$(find "$SCRATCH_PROJECT_DIR/data/augmented_data/fibrosis" -name "*.png" 2>/dev/null | wc -l)
echo "    ✓ $FIB_IMG_COUNT Fibrosis images copied"

echo "  Copying augmented Pneumonia data..."
cp -r "$PNEUMONIA_AUGMENTED_SRC" "$SCRATCH_PROJECT_DIR/data/augmented_data/"
PNEU_IMG_COUNT=$(find "$SCRATCH_PROJECT_DIR/data/augmented_data/pneumonia" -name "*.png" 2>/dev/null | wc -l)
echo "    ✓ $PNEU_IMG_COUNT Pneumonia images copied"

echo ""
echo "Scratch disk usage:"
du -sh "$SCRATCH_PROJECT_DIR"
echo ""

# ============================================================================
# ACTIVATE VIRTUAL ENVIRONMENT
# ============================================================================
echo "==> Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Python: $(which python)"
python --version
echo ""

# ============================================================================
# PREPARE DATASET CSVs
# ============================================================================
echo "==> Preparing dataset CSV files..."

# Function to prepare combined dataset.csv
prepare_dataset() {
    local LABEL=$1
    local AUG_DIR=$2
    local ORIG_CSV=$3

    python3 << EOF
import pandas as pd
from pathlib import Path

label = "$LABEL"
aug_dir = Path("$AUG_DIR")
orig_csv = Path("$ORIG_CSV")

print(f"  Preparing {label} dataset...")

# Load augmented training data
aug_csv = aug_dir / "train_augmented.csv"
if not aug_csv.exists():
    print(f"    ERROR: {aug_csv} not found")
    exit(1)

df_aug = pd.read_csv(aug_csv)
print(f"    Loaded {len(df_aug)} augmented samples")

# Load original dataset for val/test
if orig_csv.exists():
    df_orig = pd.read_csv(orig_csv)
    df_val_test = df_orig[df_orig['split'].isin(['val', 'test'])].copy()
    print(f"    Loaded {len(df_val_test)} val/test samples from original")
else:
    print(f"    ERROR: Original CSV not found: {orig_csv}")
    exit(1)

# Combine: augmented train + original val/test
df_combined = pd.concat([df_aug, df_val_test], ignore_index=True)

# Save as dataset.csv (expected by training script)
output_csv = aug_dir / "dataset.csv"
df_combined.to_csv(output_csv, index=False)

print(f"    ✓ Combined dataset saved: {output_csv}")
print(f"      Total: {len(df_combined)} samples")
print(f"      Train: {len(df_combined[df_combined['split'] == 'train'])}")
print(f"      Val: {len(df_combined[df_combined['split'] == 'val'])}")
print(f"      Test: {len(df_combined[df_combined['split'] == 'test'])}")
EOF
}

# Prepare Fibrosis dataset
prepare_dataset "Fibrosis" \
    "$SCRATCH_PROJECT_DIR/data/augmented_data/fibrosis" \
    "$SCRATCH_PROJECT_DIR/data/processed/effusion_fibrosis/dataset.csv"

echo ""

# Prepare Pneumonia dataset
prepare_dataset "Pneumonia" \
    "$SCRATCH_PROJECT_DIR/data/augmented_data/pneumonia" \
    "$SCRATCH_PROJECT_DIR/data/processed/effusion_pneumonia/dataset.csv"

echo ""

# ============================================================================
# CREATE OUTPUT DIRECTORY
# ============================================================================
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================================================
# VERIFY PYTHON ENVIRONMENT
# ============================================================================
echo "==> Verifying Python environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
python -c "from torch.utils.tensorboard import SummaryWriter; print('TensorBoard: OK')"
echo ""

# ============================================================================
# SAVE ENVIRONMENT INFO
# ============================================================================
cat > "$OUTPUT_DIR/environment_info.txt" << EOF
Job Information:
  Job ID: $JOB_ID
  Node: $(hostname)
  Start Time: $(date)

GPU:
$(nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv)

Python: $(python --version 2>&1)

Key Packages:
$(pip show torch timm tensorboard 2>/dev/null | grep -E "^(Name|Version)")

SLURM Configuration:
  Partition: ${SLURM_JOB_PARTITION:-N/A}
  GPUs: ${SLURM_GPUS:-N/A}
  CPUs: ${SLURM_CPUS_PER_TASK:-N/A}

Data Locations:
  Scratch project: $SCRATCH_PROJECT_DIR
  Fibrosis augmented: $SCRATCH_PROJECT_DIR/data/augmented_data/fibrosis
  Pneumonia augmented: $SCRATCH_PROJECT_DIR/data/augmented_data/pneumonia
  Output: $OUTPUT_DIR
EOF
echo "✓ Environment info saved"
echo ""

# ============================================================================
# TRAIN FIBROSIS CLASSIFIER
# ============================================================================
echo "============================================================================"
echo "Training Fibrosis Classifier (Effusion vs Fibrosis)"
echo "============================================================================"
echo "Start time: $(date)"
echo ""

# Change to scratch project directory (so relative paths work)
cd "$SCRATCH_PROJECT_DIR"

# Create runtime config with correct paths
FIBROSIS_RUNTIME_CONFIG="$OUTPUT_DIR/config_fibrosis_runtime.yaml"
python3 << EOF
import yaml

with open('$FIBROSIS_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# Use relative paths (we're running from scratch project dir)
config['data']['processed_dir'] = 'data/augmented_data/fibrosis'
config['data']['interim_csv'] = 'data/interim/filtered_data_entry.csv'
config['data']['train_val_dir'] = 'data/interim/train_val'
config['data']['test_dir'] = 'data/interim/test'

# Output to /w/ for persistence (absolute paths)
config['training']['checkpoint_dir'] = '$OUTPUT_DIR/fibrosis/checkpoints'
config['training']['log_dir'] = '$OUTPUT_DIR/fibrosis/logs'

# Ensure experiment name is set correctly
config['experiment']['name'] = 'effusion_vs_fibrosis_augmented'

with open('$FIBROSIS_RUNTIME_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✓ Fibrosis runtime config created")
EOF

echo "Config: $FIBROSIS_RUNTIME_CONFIG"
echo ""

# Run training
python "$TRAINING_SCRIPT" --config "$FIBROSIS_RUNTIME_CONFIG" \
    2>&1 | tee "$OUTPUT_DIR/fibrosis_training.log"

FIBROSIS_EXIT_CODE=$?

echo ""
if [ $FIBROSIS_EXIT_CODE -eq 0 ]; then
    echo "✓ Fibrosis classifier training completed successfully!"
    echo "  Checkpoints: $OUTPUT_DIR/fibrosis/checkpoints"
    echo "  TensorBoard logs: $OUTPUT_DIR/fibrosis/logs"
else
    echo "✗ Fibrosis classifier training failed with exit code $FIBROSIS_EXIT_CODE"
fi
echo "End time: $(date)"
echo ""

# ============================================================================
# TRAIN PNEUMONIA CLASSIFIER
# ============================================================================
echo "============================================================================"
echo "Training Pneumonia Classifier (Effusion vs Pneumonia)"
echo "============================================================================"
echo "Start time: $(date)"
echo ""

# Create runtime config with correct paths
PNEUMONIA_RUNTIME_CONFIG="$OUTPUT_DIR/config_pneumonia_runtime.yaml"
python3 << EOF
import yaml

with open('$PNEUMONIA_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# Use relative paths (we're running from scratch project dir)
config['data']['processed_dir'] = 'data/augmented_data/pneumonia'
config['data']['interim_csv'] = 'data/interim/filtered_data_entry.csv'
config['data']['train_val_dir'] = 'data/interim/train_val'
config['data']['test_dir'] = 'data/interim/test'

# Output to /w/ for persistence (absolute paths)
config['training']['checkpoint_dir'] = '$OUTPUT_DIR/pneumonia/checkpoints'
config['training']['log_dir'] = '$OUTPUT_DIR/pneumonia/logs'

# Ensure experiment name is set correctly
config['experiment']['name'] = 'effusion_vs_pneumonia_augmented'

with open('$PNEUMONIA_RUNTIME_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✓ Pneumonia runtime config created")
EOF

echo "Config: $PNEUMONIA_RUNTIME_CONFIG"
echo ""

# Run training
python "$TRAINING_SCRIPT" --config "$PNEUMONIA_RUNTIME_CONFIG" \
    2>&1 | tee "$OUTPUT_DIR/pneumonia_training.log"

PNEUMONIA_EXIT_CODE=$?

echo ""
if [ $PNEUMONIA_EXIT_CODE -eq 0 ]; then
    echo "✓ Pneumonia classifier training completed successfully!"
    echo "  Checkpoints: $OUTPUT_DIR/pneumonia/checkpoints"
    echo "  TensorBoard logs: $OUTPUT_DIR/pneumonia/logs"
else
    echo "✗ Pneumonia classifier training failed with exit code $PNEUMONIA_EXIT_CODE"
fi
echo "End time: $(date)"
echo ""

# ============================================================================
# CLEANUP SCRATCH
# ============================================================================
echo "==> Cleaning up scratch space..."
rm -rf "$SCRATCH_BASE/classifier_training_${JOB_ID}"
echo "✓ Scratch space cleaned"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "============================================================================"
echo "Training Complete"
echo "============================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Fibrosis Classifier:"
echo "  Exit code: $FIBROSIS_EXIT_CODE"
echo "  Log: $OUTPUT_DIR/fibrosis_training.log"
if [ $FIBROSIS_EXIT_CODE -eq 0 ]; then
    echo "  Checkpoints: $OUTPUT_DIR/fibrosis/checkpoints/"
    echo "  TensorBoard: tensorboard --logdir=$OUTPUT_DIR/fibrosis/logs"
fi
echo ""
echo "Pneumonia Classifier:"
echo "  Exit code: $PNEUMONIA_EXIT_CODE"
echo "  Log: $OUTPUT_DIR/pneumonia_training.log"
if [ $PNEUMONIA_EXIT_CODE -eq 0 ]; then
    echo "  Checkpoints: $OUTPUT_DIR/pneumonia/checkpoints/"
    echo "  TensorBoard: tensorboard --logdir=$OUTPUT_DIR/pneumonia/logs"
fi
echo ""

# Overall status
if [ $FIBROSIS_EXIT_CODE -eq 0 ] && [ $PNEUMONIA_EXIT_CODE -eq 0 ]; then
    echo "✓ Both classifiers trained successfully!"
    OVERALL_EXIT=0
elif [ $FIBROSIS_EXIT_CODE -eq 0 ] || [ $PNEUMONIA_EXIT_CODE -eq 0 ]; then
    echo "⚠ One classifier training failed"
    OVERALL_EXIT=1
else
    echo "✗ Both classifier trainings failed"
    OVERALL_EXIT=1
fi

echo ""
echo "Output disk usage:"
du -sh "$OUTPUT_DIR"
echo ""
echo "============================================================================"

exit $OVERALL_EXIT
