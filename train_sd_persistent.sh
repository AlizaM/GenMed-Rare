#!/bin/bash
#SBATCH --job-name=SD_CXR_LoRA             
#SBATCH --partition=gpunodes               
#SBATCH --constraint=RTX_A4500             
#SBATCH --gpus=1                         
#SBATCH --cpus-per-task=4                  
#SBATCH --mem=30G                          
#SBATCH --time=12:00:00                    
#SBATCH --mail-user=alizat@cs.toronto.edu
#SBATCH --mail-type=ALL

set -e  # Exit on any error

# ============================================================================
# CONFIGURATION
# ============================================================================
USER_HOME_BASE="/u/alizat"
PROJECT_DIR="$USER_HOME_BASE/cv_project/GenMed-Rare"

# Persistent locations on scratch (reused across jobs)
SCRATCH_BASE="/scratch/scratch-space"
PERSISTENT_VENV_NAME="sd_venv_persistent"
PERSISTENT_DATA_NAME="sd_data_persistent"

# Track locations in permanent storage
VENV_LOCATION_FILE="$PROJECT_DIR/.scratch_venv_location"
DATA_LOCATION_FILE="$PROJECT_DIR/.scratch_data_location"

# Source data
SOURCE_DATA_DIR="/w/20251/alizat/diffusion_data"

# Final outputs
OUTPUT_BASE_DIR="$PROJECT_DIR/outputs/diffusion_models/sd15_lora_fibrosis"

# Files
REQUIREMENTS_FILE="$PROJECT_DIR/requirements_server.txt"
CONFIG_FILE="$PROJECT_DIR/configs/config_diffusion.yaml"
TRAINING_SCRIPT="$PROJECT_DIR/scripts/train_diffusion.py"


# ============================================================================
# FIND OR CREATE PERSISTENT VENV ON SCRATCH
# ============================================================================
echo "==> Checking for existing venv on scratch..."

VENV_DIR=""
if [ -f "$VENV_LOCATION_FILE" ]; then
    SAVED_VENV_PATH=$(cat "$VENV_LOCATION_FILE")
    if [ -d "$SAVED_VENV_PATH" ] && [ -f "$SAVED_VENV_PATH/bin/activate" ]; then
        echo "==> Found existing venv at: $SAVED_VENV_PATH"
        VENV_DIR="$SAVED_VENV_PATH"
    else
        echo "==> Saved venv path no longer valid (scratch expired)"
    fi
fi

if [ -z "$VENV_DIR" ]; then
    echo "==> Creating new venv on scratch..."
    
    # Find the latest scratch directory
    LATEST_SCRATCH_DIR=$(ls -td $SCRATCH_BASE/expires-* 2>/dev/null | head -1)
    if [ -z "$LATEST_SCRATCH_DIR" ]; then
        echo "ERROR: No scratch directory found."
        exit 1
    fi
    
    VENV_DIR="$LATEST_SCRATCH_DIR/$PERSISTENT_VENV_NAME"
    
    if [ ! -d "$VENV_DIR" ]; then
        echo "==> Building venv at $VENV_DIR (this may take 10-15 minutes)..."
        python3 -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip
        pip install --no-cache-dir -r "$REQUIREMENTS_FILE"
        echo "==> Venv created successfully!"
        
        # Save location for next time
        echo "$VENV_DIR" > "$VENV_LOCATION_FILE"
        echo "==> Saved venv location to $VENV_LOCATION_FILE"
    else
        echo "==> Using existing venv in current scratch dir"
    fi
else
    source "$VENV_DIR/bin/activate"
fi

# Verify environment
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"


# ============================================================================
# FIND OR COPY DATA TO SCRATCH
# ============================================================================
echo "==> Checking for existing data on scratch..."

DATA_DIR=""
if [ -f "$DATA_LOCATION_FILE" ]; then
    SAVED_DATA_PATH=$(cat "$DATA_LOCATION_FILE")
    if [ -d "$SAVED_DATA_PATH" ] && [ -f "$SAVED_DATA_PATH/diffusion_dataset_balanced.csv" ]; then
        echo "==> Found existing data at: $SAVED_DATA_PATH"
        DATA_DIR="$SAVED_DATA_PATH"
    else
        echo "==> Saved data path no longer valid (scratch expired)"
    fi
fi

if [ -z "$DATA_DIR" ]; then
    echo "==> Copying data to scratch..."
    
    # Find the latest scratch directory (might be same as venv or different)
    LATEST_SCRATCH_DIR=$(ls -td $SCRATCH_BASE/expires-* 2>/dev/null | head -1)
    DATA_DIR="$LATEST_SCRATCH_DIR/$PERSISTENT_DATA_NAME"
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "==> Copying ~10k images (this may take 5-10 minutes)..."
        mkdir -p "$DATA_DIR"
        rsync -av --info=progress2 "$SOURCE_DATA_DIR/" "$DATA_DIR/"
        echo "==> Data copy complete!"
        
        # Save location for next time
        echo "$DATA_DIR" > "$DATA_LOCATION_FILE"
        echo "==> Saved data location to $DATA_LOCATION_FILE"
    else
        echo "==> Using existing data in current scratch dir"
    fi
fi

# Verify data
NUM_IMAGES=$(find "$DATA_DIR" -name "*.png" | wc -l)
echo "==> Data ready: $NUM_IMAGES images in $DATA_DIR"


# ============================================================================
# SETUP JOB-SPECIFIC SCRATCH FOR OUTPUTS
# ============================================================================
LATEST_SCRATCH_DIR=$(ls -td $SCRATCH_BASE/expires-* 2>/dev/null | head -1)
JOB_SCRATCH_DIR="$LATEST_SCRATCH_DIR/sd_job_$SLURM_JOB_ID"
mkdir -p "$JOB_SCRATCH_DIR"

SCRATCH_OUTPUT_DIR="$JOB_SCRATCH_DIR/outputs"
mkdir -p "$SCRATCH_OUTPUT_DIR"


# ============================================================================
# CREATE TEMPORARY CONFIG WITH SCRATCH PATHS
# ============================================================================
SCRATCH_CONFIG="$JOB_SCRATCH_DIR/config.yaml"
echo "==> Creating config with scratch paths..."

python3 << EOF
import yaml

with open("$CONFIG_FILE", 'r') as f:
    config = yaml.safe_load(f)

# Update paths to scratch locations
config['data_dir'] = "$DATA_DIR"
config['output_dir'] = "$SCRATCH_OUTPUT_DIR"

with open("$SCRATCH_CONFIG", 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"✓ Config ready")
print(f"  Data: {config['data_dir']}")
print(f"  Output: {config['output_dir']}")
EOF


# ============================================================================
# LAUNCH TRAINING
# ============================================================================
echo "==> Launching training..."
echo "==> Venv: $VENV_DIR"
echo "==> Data: $DATA_DIR"
echo "==> Output: $SCRATCH_OUTPUT_DIR"

cd "$PROJECT_DIR"

accelerate launch "$TRAINING_SCRIPT" --config "$SCRATCH_CONFIG" || TRAINING_FAILED=1


# ============================================================================
# COPY RESULTS BACK (EVEN IF FAILED)
# ============================================================================
echo "==> Copying outputs to permanent storage..."
mkdir -p "$OUTPUT_BASE_DIR"
rsync -av --info=progress2 "$SCRATCH_OUTPUT_DIR/" "$OUTPUT_BASE_DIR/"

echo "==> Outputs saved to: $OUTPUT_BASE_DIR"


# ============================================================================
# CLEANUP JOB-SPECIFIC SCRATCH (KEEP VENV AND DATA)
# ============================================================================
echo "==> Cleaning up job-specific directory (keeping venv and data for reuse)..."
rm -rf "$JOB_SCRATCH_DIR"

if [ -n "$TRAINING_FAILED" ]; then
    echo "ERROR: Training failed, but outputs were saved."
    echo "==> Venv and data preserved for next run"
    exit 1
fi

echo "==> Job completed successfully!"
echo "==> Venv location saved: $VENV_DIR"
echo "==> Data location saved: $DATA_DIR"
echo "==> These will be reused on next job submission"
