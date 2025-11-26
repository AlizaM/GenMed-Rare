#!/bin/bash
#SBATCH --job-name=generate_pneumonia_fibrosis_xrays
#SBATCH --partition=gpunodes
#SBATCH --constraint=RTX_A4500
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --mail-user=alizat@cs.toronto.edu
#SBATCH --mail-type=ALL

# Synthetic X-ray Generation Script
# This script generates synthetic chest X-rays for both Pneumonia and Fibrosis

echo "=========================================="
echo "Synthetic X-ray Generation Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Configuration
PROJECT_DIR="/u/$(whoami)/cv_project/GenMed-Rare"
VENV_MARKER="${PROJECT_DIR}/.scratch_venv_location"
OUTPUT_BASE="/w/20251/$(whoami)/generated_xrays"

# Determine which pathology to generate based on job parameter
PATHOLOGY="${1:-Pneumonia}"  # Default to Pneumonia if not specified
NUM_IMAGES="${2:-2000}"      # Default to 2000 images

echo "Configuration:"
echo "  Pathology: ${PATHOLOGY}"
echo "  Number of images: ${NUM_IMAGES}"
echo "  Output base: ${OUTPUT_BASE}"
echo ""

# Navigate to project directory
cd "${PROJECT_DIR}" || exit 1
echo "Working directory: $(pwd)"

# Load virtual environment
if [ -f "${VENV_MARKER}" ]; then
    VENV_PATH=$(cat "${VENV_MARKER}")
    echo "Loading virtual environment from: ${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
else
    echo "ERROR: Virtual environment marker not found at ${VENV_MARKER}"
    echo "Please run the training script first to set up the venv."
    exit 1
fi

# Verify Python environment
echo ""
echo "Python environment:"
python --version
which python
echo ""

# GPU information
echo "GPU information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set checkpoint paths based on pathology
if [ "${PATHOLOGY}" == "Pneumonia" ]; then
    # Pneumonia: Use LoRA checkpoint (best checkpoint from evaluation)
    CHECKPOINT="/w/20251/alizat/diffusion_outputs/pneumonia_prior_job_17006/checkpoint-3500/lora_weights"
    CONFIG="/u/alizat/cv_project/GenMed-Rare/configs/config_diffusion_pneumonia.yaml"
    LORA_SCALE=1.2
    echo "Using Pneumonia LoRA checkpoint"
elif [ "${PATHOLOGY}" == "Fibrosis" ]; then
    # Fibrosis: Use full model checkpoint (old format with model.safetensors)
    CHECKPOINT="/w/20251/alizat/diffusion_outputs/fibrosis_prior_job_16022/checkpoint-6500/"
    CONFIG="/u/alizat/cv_project/GenMed-Rare/configs/config_diffusion_fibrosis.yaml"
    LORA_SCALE=1.2
    echo "Using Fibrosis full model checkpoint"
else
    echo "ERROR: Unknown pathology: ${PATHOLOGY}"
    echo "Supported: Pneumonia, Fibrosis"
    exit 1
fi

# Verify checkpoint exists
if [ ! -d "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint directory not found: ${CHECKPOINT}"
    exit 1
fi

echo "Checkpoint: ${CHECKPOINT}"
echo "Config: ${CONFIG}"
echo ""

# Create output directory
OUTPUT_DIR="${OUTPUT_BASE}/${PATHOLOGY}_${NUM_IMAGES}_images_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Save generation parameters
cat > "${OUTPUT_DIR}/generation_params.txt" <<EOF
Generation Parameters
=====================
Date: $(date)
Job ID: ${SLURM_JOB_ID}
Node: ${SLURM_NODELIST}

Pathology: ${PATHOLOGY}
Checkpoint: ${CHECKPOINT}
Config: ${CONFIG}
Number of images: ${NUM_IMAGES}
LoRA scale: ${LORA_SCALE}
Inference steps: 50
Guidance scale: 7.5
Seed: 42

Command:
python scripts/generate_xrays.py \\
    --checkpoint ${CHECKPOINT} \\
    --config ${CONFIG} \\
    --pathology ${PATHOLOGY} \\
    --num-images ${NUM_IMAGES} \\
    --lora-scale ${LORA_SCALE} \\
    --num-inference-steps 50 \\
    --guidance-scale 7.5 \\
    --seed 42 \\
    --output-dir ${OUTPUT_DIR}
EOF

echo "=========================================="
echo "Starting Image Generation"
echo "=========================================="
echo ""

# Run generation
python scripts/generate_xrays.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --pathology "${PATHOLOGY}" \
    --num-images "${NUM_IMAGES}" \
    --lora-scale "${LORA_SCALE}" \
    --num-inference-steps 50 \
    --guidance-scale 7.5 \
    --seed 42 \
    --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job Complete"
echo "=========================================="
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "Output directory: ${OUTPUT_DIR}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Generation successful!"
    echo "Generated images are in: ${OUTPUT_DIR}"

    # Count generated images
    IMAGE_COUNT=$(find "${OUTPUT_DIR}" -name "*.png" | wc -l)
    echo "Total images generated: ${IMAGE_COUNT}"

    # Show CSV if it exists
    if [ -f "${OUTPUT_DIR}/generation_results.csv" ]; then
        echo ""
        echo "Results CSV: ${OUTPUT_DIR}/generation_results.csv"
        echo "First few entries:"
        head -n 5 "${OUTPUT_DIR}/generation_results.csv"
    fi
else
    echo ""
    echo "ERROR: Generation failed with exit code ${EXIT_CODE}"
    echo "Check the log file for details."
fi

echo "=========================================="

exit ${EXIT_CODE}
