#!/bin/bash
#SBATCH --job-name=eval_fibrosis
#SBATCH --partition=gpunodes
#SBATCH --constraint=RTX_A4500
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --mail-user=alizat@cs.toronto.edu
#SBATCH --mail-type=ALL

set -e

# ============================================================================
# MINIMAL SLURM JOB SCRIPT FOR CHECKPOINT EVALUATION
# 
# All configuration is in: configs/config_eval_fibrosis.yaml
# This script just activates the environment and runs the Python script.
# ============================================================================

echo "==> Starting checkpoint evaluation job"
echo "==> Job ID: $SLURM_JOB_ID"
echo "==> Node: $(hostname)"
echo "==> GPU: $CUDA_VISIBLE_DEVICES"

# Project paths
PROJECT_DIR="/u/alizat/cv_project/GenMed-Rare"
VENV_LOCATION_FILE="$PROJECT_DIR/.scratch_venv_location"

# Read venv location
if [ -f "$VENV_LOCATION_FILE" ]; then
    VENV_DIR=$(cat "$VENV_LOCATION_FILE")
    echo "==> Using venv: $VENV_DIR"
else
    echo "ERROR: Venv location file not found: $VENV_LOCATION_FILE"
    exit 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Enable CUDA debugging for better error messages
export CUDA_LAUNCH_BLOCKING=1

# Run evaluation
# All paths and configuration are in the YAML file
cd "$PROJECT_DIR"

python scripts/evaluate_checkpoints.py \
    --config configs/config_eval_diffusion.yaml \
    --preset checkpoint \
    --min-images 100

# Preset options:
#   --preset checkpoint  (default: novelty, pathology, biovil, diversity - FAST)
#   --preset diversity   (diversity-focused metrics for mode collapse detection)
#   --preset full        (all 9 metrics including FMD and t-SNE - EXPENSIVE)
#
# Other options:
#   --min-images 100     (skip generation if >= 100 images exist, default: 100)
#   --num-images 200     (override config to generate 200 images)

echo "==> Evaluation complete!"
