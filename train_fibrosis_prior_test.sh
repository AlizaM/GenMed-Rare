#!/bin/bash
#SBATCH --job-name=Fib_Prior_SD_TEST
#SBATCH --partition=gpunodes
#SBATCH --constraint=RTX_A4500
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --mail-user=alizat@cs.toronto.edu
#SBATCH --mail-type=ALL

# =========================================================================
# Prior-Based Fibrosis Diffusion Training (TEST)
# =========================================================================
set -e

USER_HOME_BASE="/u/alizat"
PROJECT_DIR="$USER_HOME_BASE/cv_project/GenMed-Rare"
VENV_LOCATION_FILE="$PROJECT_DIR/.scratch_venv_location"
W_STORAGE_BASE="/w/20251/alizat"
JOB_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="$W_STORAGE_BASE/diffusion_outputs/fibrosis_prior_test_job_${JOB_ID}"
CONFIG_FILE="$PROJECT_DIR/configs/config_diffusion_fibrosis_test.yaml"
TRAINING_SCRIPT="$PROJECT_DIR/scripts/train_diffusion_prior.py"

# PRE-FLIGHT CHECKS

echo "============================================================================"
echo "Prior-Based Fibrosis Diffusion Training (TEST)"
echo "============================================================================"
echo "Job ID: $JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# FIND VENV ON SCRATCH

echo "==> Looking for venv on scratch..."
if [ ! -f "$VENV_LOCATION_FILE" ]; then
    echo "ERROR: Venv location file not found: $VENV_LOCATION_FILE"
    exit 1
fi
VENV_DIR=$(cat "$VENV_LOCATION_FILE")
echo "Found venv location: $VENV_DIR"
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Venv not found at $VENV_DIR (scratch may have expired)"
    exit 1
fi
echo "✓ Venv verified at: $VENV_DIR"
echo ""

# Check config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
# Check training script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi
# Check data directories exist
FIBROSIS_DIR="$PROJECT_DIR/data/pure_class_folders/fibrosis"
HEALTHY_DIR="$PROJECT_DIR/data/pure_class_folders/healthy"
if [ ! -d "$FIBROSIS_DIR" ]; then
    echo "ERROR: Fibrosis data directory not found: $FIBROSIS_DIR"
    exit 1
fi
if [ ! -d "$HEALTHY_DIR" ]; then
    echo "ERROR: Healthy data directory not found: $HEALTHY_DIR"
    exit 1
fi
echo "✓ All files and directories verified"
echo ""

# DISPLAY CONFIGURATION

echo "Configuration:"
echo "  - Config: $CONFIG_FILE"
echo "  - Training Script: $TRAINING_SCRIPT"
echo "  - Venv: $VENV_DIR"
echo "  - Output Dir: $OUTPUT_DIR"
echo "  - Fibrosis Images: $FIBROSIS_DIR"
echo "  - Healthy Images: $HEALTHY_DIR"
echo ""

# Count images
FIBROSIS_COUNT=$(find "$FIBROSIS_DIR" -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l)
HEALTHY_COUNT=$(find "$HEALTHY_DIR" -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l)
echo "Dataset Statistics:"
echo "  - Fibrosis images: $FIBROSIS_COUNT"
echo "  - Healthy images: $HEALTHY_COUNT"
echo ""

# GPU INFORMATION

echo "GPU Information:"
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv,noheader
echo ""

# CREATE OUTPUT DIRECTORY

echo "==> Creating output directory..."
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory created: $OUTPUT_DIR"
echo ""

# ACTIVATE VENV

echo "==> Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Verify key packages
echo "==> Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
echo ""

# CREATE TEMPORARY CONFIG WITH UPDATED PATHS

echo "==> Creating temporary config with updated paths..."
TEMP_CONFIG="$OUTPUT_DIR/config_runtime.yaml"
python3 << EOF
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
if 'paths' not in config:
    config['paths'] = {}
config['paths']['output_dir'] = '$OUTPUT_DIR'
config['paths']['logging_dir'] = '$OUTPUT_DIR/logs'
config['training']['target_images_dir'] = '$FIBROSIS_DIR'
config['training']['prior_images_dir'] = '$HEALTHY_DIR'
config['training']['target_images_csv'] = '$PROJECT_DIR/data/pure_class_folders/fibrosis_images.csv'
config['training']['prior_images_csv'] = '$PROJECT_DIR/data/pure_class_folders/healthy_images.csv'
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('✓ Temporary config created')
EOF
echo ""

# COPY CONFIG TO OUTPUT DIRECTORY (for reproducibility)
echo "==> Saving configuration..."
cp "$CONFIG_FILE" "$OUTPUT_DIR/config_original.yaml"
echo "✓ Configurations saved"
echo ""

# SAVE ENVIRONMENT INFO
echo "==> Saving environment information..."
cat > "$OUTPUT_DIR/environment_info.txt" << EOF
Job Information:
  Job ID: $JOB_ID
  Node: $(hostname)
  Start Time: $(date)
GPU:
$(nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv)
Python Environment:
$(python --version)
$(pip freeze)
SLURM Configuration:
  Partition: $SLURM_JOB_PARTITION
  GPUs: $SLURM_GPUS
  CPUs: $SLURM_CPUS_PER_TASK
  Memory: $SLURM_MEM_PER_NODE
  Time Limit: $SLURM_TIMELIMIT
EOF
echo "✓ Environment info saved"
echo ""

# RUN TRAINING
echo "============================================================================"
echo "Starting Training (TEST)"
echo "============================================================================"
echo ""
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
cd "$PROJECT_DIR"
python "$TRAINING_SCRIPT" \
    --config "$TEMP_CONFIG" \
    2>&1 | tee "$OUTPUT_DIR/training.log"
TRAINING_EXIT_CODE=$?

echo ""
echo "============================================================================"
echo "Training Complete (TEST)"
echo "============================================================================"
echo ""
echo "Exit Code: $TRAINING_EXIT_CODE"
echo "End Time: $(date)"
echo ""
echo "Output Location: $OUTPUT_DIR"
echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""
    echo "Outputs:"
    echo "  - Checkpoints: $OUTPUT_DIR/checkpoint-*"
    echo "  - Logs: $OUTPUT_DIR/logs/"
    echo "  - Config: $OUTPUT_DIR/config_*.yaml"
    echo "  - Training log: $OUTPUT_DIR/training.log"
    echo ""
    CHECKPOINT_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | wc -l)
    echo "  Total checkpoints saved: $CHECKPOINT_COUNT"
    echo ""
    echo "Disk Usage:"
    du -sh "$OUTPUT_DIR"
else
    echo "✗ Training failed with exit code $TRAINING_EXIT_CODE"
    echo ""
    echo "Check the log file for details:"
    echo "  $OUTPUT_DIR/training.log"
    echo ""
    echo "Last 50 lines of log:"
    tail -50 "$OUTPUT_DIR/training.log"
fi
echo ""
echo "============================================================================"

exit $TRAINING_EXIT_CODE
