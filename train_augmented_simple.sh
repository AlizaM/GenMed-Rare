#!/bin/bash
#SBATCH --job-name=train_aug_clf
#SBATCH --partition=gpunodes
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --mail-user=alizat@cs.toronto.edu
#SBATCH --mail-type=ALL

# ============================================================================
# Train Augmented Classifiers
# ============================================================================
# Uses data directly from /w/ (no scratch copy)
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_DIR="/u/alizat/cv_project/GenMed-Rare"
W_DATA_DIR="/w/20251/alizat/data"
VENV_LOCATION_FILE="$PROJECT_DIR/.scratch_venv_location"

JOB_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="/w/20251/alizat/classifier_outputs/augmented_${JOB_ID}"

echo "============================================================================"
echo "Training Augmented Classifiers"
echo "============================================================================"
echo "Job ID: $JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# ============================================================================
# CHECK VENV
# ============================================================================
if [ ! -f "$VENV_LOCATION_FILE" ]; then
    echo "ERROR: Venv location file not found: $VENV_LOCATION_FILE"
    exit 1
fi
VENV_DIR=$(cat "$VENV_LOCATION_FILE")
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Venv not found at $VENV_DIR"
    exit 1
fi
echo "Venv: $VENV_DIR"

# GPU info
echo ""
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader
echo ""

# ============================================================================
# ACTIVATE ENVIRONMENT
# ============================================================================
source "$VENV_DIR/bin/activate"
echo "Python: $(which python)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

# ============================================================================
# VERIFY DATA
# ============================================================================
echo ""
echo "Using data from: $W_DATA_DIR"
IMAGE_COUNT=$(find "$W_DATA_DIR/interim" -name "*.png" 2>/dev/null | wc -l)
echo "Total images: $IMAGE_COUNT"

# Check CSV exists in home
CSV_PATH="$PROJECT_DIR/data/processed/effusion_fibrosis/dataset.csv"
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: CSV not found: $CSV_PATH"
    exit 1
fi
echo "CSV: $CSV_PATH"

# Show CSV stats
echo ""
echo "CSV stats:"
python3 -c "
import pandas as pd
df = pd.read_csv('$CSV_PATH')
print(f'Total rows: {len(df)}')
print(f'Train: {len(df[df.split==\"train\"])}')
print(f'  Effusion: {len(df[(df.split==\"train\") & (df.label==0)])}')
print(f'  Fibrosis: {len(df[(df.split==\"train\") & (df.label==1)])}')
if 'source' in df.columns:
    print(f'  Generated: {len(df[(df.split==\"train\") & (df.source==\"generated\")])}')
"
echo ""

# ============================================================================
# TRAIN FIBROSIS CLASSIFIER
# ============================================================================
echo "============================================================================"
echo "Training Fibrosis Classifier"
echo "============================================================================"

python scripts/train_classifier.py \
    --config configs/config_augmented_fibrosis.yaml \
    --data-root "$W_DATA_DIR" \
    2>&1 | tee "$OUTPUT_DIR/fibrosis_training.log"

FIBROSIS_EXIT=$?
echo "Fibrosis exit code: $FIBROSIS_EXIT"

# ============================================================================
# TRAIN PNEUMONIA CLASSIFIER
# ============================================================================
echo ""
echo "============================================================================"
echo "Training Pneumonia Classifier"
echo "============================================================================"

python scripts/train_classifier.py \
    --config configs/config_augmented_pneumonia.yaml \
    --data-root "$W_DATA_DIR" \
    2>&1 | tee "$OUTPUT_DIR/pneumonia_training.log"

PNEUMONIA_EXIT=$?
echo "Pneumonia exit code: $PNEUMONIA_EXIT"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================================"
echo "Done"
echo "============================================================================"
echo "Fibrosis: $FIBROSIS_EXIT"
echo "Pneumonia: $PNEUMONIA_EXIT"
echo "Logs: $OUTPUT_DIR"

if [ $FIBROSIS_EXIT -eq 0 ] && [ $PNEUMONIA_EXIT -eq 0 ]; then
    exit 0
else
    exit 1
fi
