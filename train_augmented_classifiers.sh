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
# Data layout:
#   /u/alizat/cv_project/GenMed-Rare/  <- CODE only (home)
#   /w/20251/alizat/data/              <- ALL DATA (interim, processed, augmented)
#
# Strategy:
#   1. Find scratch space dynamically
#   2. Copy DATA from /w/ to scratch ONCE (reuse if exists)
#   3. Copy CODE from home to scratch EVERY TIME (fresh copy)
#   4. Symlink data folder so relative paths work
#   5. Run training, output to /w/
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

# HDD storage base (where data lives + outputs go)
W_STORAGE_BASE="/w/20251/alizat"
W_DATA_DIR="$W_STORAGE_BASE/data"

# Config files (relative to project)
FIBROSIS_CONFIG="configs/config_augmented_fibrosis.yaml"
PNEUMONIA_CONFIG="configs/config_augmented_pneumonia.yaml"

# Training script (relative to project)
TRAINING_SCRIPT="scripts/train_classifier.py"

# Job ID for unique directories
JOB_ID="${SLURM_JOB_ID:-local}"

# Output directory (persistent on /w/)
OUTPUT_DIR="$W_STORAGE_BASE/classifier_outputs/augmented_training_${JOB_ID}"

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

# Check data directories on /w/
if [ ! -d "$W_DATA_DIR/interim" ]; then
    echo "ERROR: Interim data not found: $W_DATA_DIR/interim"
    exit 1
fi
echo "✓ Interim data verified on /w/"

if [ ! -d "$W_DATA_DIR/processed" ]; then
    echo "ERROR: Processed data not found: $W_DATA_DIR/processed"
    exit 1
fi
echo "✓ Processed data verified on /w/"

if [ ! -d "$W_DATA_DIR/augmented_data/fibrosis" ]; then
    echo "ERROR: Fibrosis augmented data not found: $W_DATA_DIR/augmented_data/fibrosis"
    exit 1
fi
if [ ! -d "$W_DATA_DIR/augmented_data/pneumonia" ]; then
    echo "ERROR: Pneumonia augmented data not found: $W_DATA_DIR/augmented_data/pneumonia"
    exit 1
fi
echo "✓ Augmented data verified on /w/"
echo ""

# ============================================================================
# GPU INFORMATION
# ============================================================================
echo "GPU Information:"
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv,noheader
echo ""

# ============================================================================
# FIND SCRATCH SPACE
# ============================================================================
echo "==> Finding scratch space..."
SCRATCH_BASE="/scratch/scratch-space"
LATEST_SCRATCH_DIR=$(ls -td $SCRATCH_BASE/expires-* 2>/dev/null | head -1)

if [ -z "$LATEST_SCRATCH_DIR" ]; then
    echo "ERROR: No scratch directory found in $SCRATCH_BASE"
    exit 1
fi

echo "    Found scratch: $LATEST_SCRATCH_DIR"

# Scratch directory for data only
SCRATCH_DATA_DIR="$LATEST_SCRATCH_DIR/classifier_data"

# ============================================================================
# COPY DATA TO SCRATCH (ONCE - reuse if exists)
# ============================================================================
echo "==> Setting up data on scratch..."

if [ -d "$SCRATCH_DATA_DIR/interim" ] && [ -d "$SCRATCH_DATA_DIR/processed" ] && [ -d "$SCRATCH_DATA_DIR/augmented_data" ]; then
    echo "    ✓ Data already exists on scratch, reusing..."
    INTERIM_COUNT=$(find "$SCRATCH_DATA_DIR/interim" -name "*.png" 2>/dev/null | wc -l)
    echo "      Interim: $INTERIM_COUNT images"
else
    echo "    Copying data from /w/ to scratch (first time)..."
    mkdir -p "$SCRATCH_DATA_DIR"

    # Copy interim data
    echo "      Copying interim data..."
    cp -r "$W_DATA_DIR/interim" "$SCRATCH_DATA_DIR/"
    INTERIM_COUNT=$(find "$SCRATCH_DATA_DIR/interim" -name "*.png" 2>/dev/null | wc -l)
    echo "        ✓ $INTERIM_COUNT interim images"

    # Copy processed data
    echo "      Copying processed data..."
    cp -r "$W_DATA_DIR/processed" "$SCRATCH_DATA_DIR/"
    echo "        ✓ Processed CSVs copied"

    # Copy augmented data
    echo "      Copying augmented data..."
    cp -r "$W_DATA_DIR/augmented_data" "$SCRATCH_DATA_DIR/"
    FIB_COUNT=$(find "$SCRATCH_DATA_DIR/augmented_data/fibrosis" -name "*.png" 2>/dev/null | wc -l)
    PNEU_COUNT=$(find "$SCRATCH_DATA_DIR/augmented_data/pneumonia" -name "*.png" 2>/dev/null | wc -l)
    echo "        ✓ $FIB_COUNT Fibrosis + $PNEU_COUNT Pneumonia images"
fi
echo ""

# ============================================================================
# SYMLINK DATA IN HOME PROJECT (run code from home, data from scratch)
# ============================================================================
echo "==> Setting up data symlink in home project..."

# Backup existing data folder if it's a real directory (not a symlink)
if [ -d "$PROJECT_DIR/data" ] && [ ! -L "$PROJECT_DIR/data" ]; then
    echo "    WARNING: $PROJECT_DIR/data is a real directory, backing up..."
    mv "$PROJECT_DIR/data" "$PROJECT_DIR/data_backup_${JOB_ID}"
fi

# Remove existing symlink if present
if [ -L "$PROJECT_DIR/data" ]; then
    rm "$PROJECT_DIR/data"
fi

# Create symlink from home project/data -> scratch data
ln -s "$SCRATCH_DATA_DIR" "$PROJECT_DIR/data"
echo "    ✓ Symlinked: $PROJECT_DIR/data -> $SCRATCH_DATA_DIR"

echo ""
echo "Scratch disk usage:"
du -sh "$SCRATCH_DATA_DIR" 2>/dev/null || true
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
    "$SCRATCH_DATA_DIR/augmented_data/fibrosis" \
    "$SCRATCH_DATA_DIR/processed/effusion_fibrosis/dataset.csv"

echo ""

# Prepare Pneumonia dataset
prepare_dataset "Pneumonia" \
    "$SCRATCH_DATA_DIR/augmented_data/pneumonia" \
    "$SCRATCH_DATA_DIR/processed/effusion_pneumonia/dataset.csv"

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

Data Sources:
  Code: $PROJECT_DIR (home - run from here)
  Data: $W_DATA_DIR (/w/ - original location)

Layout:
  Scratch Data: $SCRATCH_DATA_DIR (fast I/O)
  Symlink: $PROJECT_DIR/data -> $SCRATCH_DATA_DIR

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

# Run from home project directory (code is here, data is symlinked to scratch)
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"
echo ""

# Verify data symlink works
ls -la "$PROJECT_DIR/data" | head -1
echo ""

# Create runtime config with correct paths
FIBROSIS_RUNTIME_CONFIG="$OUTPUT_DIR/config_fibrosis_runtime.yaml"
python3 << EOF
import yaml

with open('$FIBROSIS_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# Use relative paths (symlink makes data/ available)
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
python scripts/train_classifier.py --config "$FIBROSIS_RUNTIME_CONFIG" \
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

# Use relative paths (symlink makes data/ available)
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
python scripts/train_classifier.py --config "$PNEUMONIA_RUNTIME_CONFIG" \
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
# CLEANUP (keep data symlink and scratch data for reuse)
# ============================================================================
echo "==> Scratch data preserved for reuse at: $SCRATCH_DATA_DIR"
echo "    Data symlink remains: $PROJECT_DIR/data -> $SCRATCH_DATA_DIR"
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
