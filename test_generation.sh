#!/bin/bash
#SBATCH --job-name=test_generation
#SBATCH --partition=gpunodes
#SBATCH --constraint=RTX_A4500
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --mail-user=alizat@cs.toronto.edu
#SBATCH --mail-type=ALL

# Test image generation with a small number of images

echo "=========================================="
echo "Testing X-ray Generation Script"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Configuration
PROJECT_DIR="/u/$(whoami)/cv_project/GenMed-Rare"
VENV_MARKER="${PROJECT_DIR}/.scratch_venv_location"

PATHOLOGY="${1:-Pneumonia}"
NUM_IMAGES="${2:-2}"

echo "Testing with:"
echo "  Pathology: ${PATHOLOGY}"
echo "  Number of images: ${NUM_IMAGES}"
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
    echo "Trying local .venv..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "Using local .venv"
    else
        echo "ERROR: No virtual environment found"
        exit 1
    fi
fi

# Verify Python
echo "Python: $(which python)"
python --version
echo ""

# Set checkpoint paths based on pathology
if [ "${PATHOLOGY}" == "Pneumonia" ]; then
    CHECKPOINT="/w/20251/alizat/diffusion_outputs/pneumonia_prior_job_17006/checkpoint-3500/lora_weights"
    CONFIG="/u/alizat/cv_project/GenMed-Rare/configs/config_diffusion_pneumonia.yaml"
elif [ "${PATHOLOGY}" == "Fibrosis" ]; then
    CHECKPOINT="/w/20251/alizat/diffusion_outputs/fibrosis_prior_job_16022/checkpoint-6500/"
    CONFIG="/u/alizat/cv_project/GenMed-Rare/configs/config_diffusion_fibrosis.yaml"
else
    echo "ERROR: Unknown pathology: ${PATHOLOGY}"
    exit 1
fi

# Create test output directory
OUTPUT_DIR="/w/20251/alizat/outputs/test_generation_${PATHOLOGY}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Run generation
python scripts/generate_xrays.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --pathology "${PATHOLOGY}" \
    --num-images "${NUM_IMAGES}" \
    --lora-scale 1.0 \
    --num-inference-steps 50 \
    --guidance-scale 7.5 \
    --seed 42 \
    --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ Test successful!"
    echo "Generated images: ${OUTPUT_DIR}"
    ls -lh "${OUTPUT_DIR}"/*.png 2>/dev/null || echo "No images found"
else
    echo ""
    echo "✗ Test failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
