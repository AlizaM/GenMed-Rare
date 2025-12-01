#!/bin/bash
#SBATCH --job-name=eval_gen_images
#SBATCH --partition=gpunodes
#SBATCH --constraint=RTX_A4500
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --mail-user=alizat@cs.toronto.edu
#SBATCH --mail-type=ALL

# Evaluate both Fibrosis and Pneumonia generated images
# This script runs comprehensive evaluation on pre-generated image folders

echo "=========================================="
echo "Evaluating Generated Images"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Configuration
PROJECT_DIR="/u/$(whoami)/cv_project/GenMed-Rare"
VENV_MARKER="${PROJECT_DIR}/.scratch_venv_location"

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

# GPU information
echo "GPU information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Define paths
FIBROSIS_GENERATED="/w/20251/alizat/generated_xrays/Fibrosis_2000_images_20251126_032343"
FIBROSIS_REAL="${PROJECT_DIR}/data/pure_class_folders/fibrosis"
PNEUMONIA_GENERATED="/w/20251/alizat/generated_xrays/Pneumonia_2000_images_20251126_004836"
PNEUMONIA_REAL="${PROJECT_DIR}/data/pure_class_folders/pneumonia"
HEALTHY="${PROJECT_DIR}/data/pure_class_folders/healthy"

OUTPUT_BASE="/w/20251/alizat/outputs/evaluation_$(date +%Y%m%d_%H%M%S)"

# Evaluation preset (checkpoint, diversity, or full)
PRESET="${1:-full}"  # Default to full evaluation (all 9 metrics)

echo "Evaluation configuration:"
echo "  Preset: ${PRESET}"
echo "  Output directory: ${OUTPUT_BASE}"
echo ""

# Verify generated image directories exist
echo "Verifying paths..."
if [ ! -d "${FIBROSIS_GENERATED}" ]; then
    echo "ERROR: Fibrosis generated images not found: ${FIBROSIS_GENERATED}"
    exit 1
fi
if [ ! -d "${PNEUMONIA_GENERATED}" ]; then
    echo "ERROR: Pneumonia generated images not found: ${PNEUMONIA_GENERATED}"
    exit 1
fi
echo "✓ All paths verified"
echo ""

# Count images
FIBROSIS_COUNT=$(find "${FIBROSIS_GENERATED}" -name "*.png" | wc -l)
PNEUMONIA_COUNT=$(find "${PNEUMONIA_GENERATED}" -name "*.png" | wc -l)
echo "Image counts:"
echo "  Fibrosis: ${FIBROSIS_COUNT} images"
echo "  Pneumonia: ${PNEUMONIA_COUNT} images"
echo ""

# ========================================
# Evaluate Fibrosis
# ========================================
echo "=========================================="
echo "Evaluating Fibrosis Images"
echo "=========================================="
echo ""

FIBROSIS_OUTPUT="${OUTPUT_BASE}/fibrosis_evaluation"

python scripts/evaluate_diffusion_generation.py \
    --generated-dir "${FIBROSIS_GENERATED}" \
    --real-dir "${FIBROSIS_REAL}" \
    --label Fibrosis \
    --output-dir "${FIBROSIS_OUTPUT}" \
    --metrics novelty \
    --crop-border-pixels 10 \
    --healthy-images-dir "${HEALTHY}"

FIBROSIS_EXIT=$?

if [ ${FIBROSIS_EXIT} -eq 0 ]; then
    echo ""
    echo "✓ Fibrosis evaluation completed successfully"
    echo "  Results: ${FIBROSIS_OUTPUT}"
else
    echo ""
    echo "✗ Fibrosis evaluation failed with exit code ${FIBROSIS_EXIT}"
fi

echo ""

# ========================================
# Evaluate Pneumonia
# ========================================
echo "=========================================="
echo "Evaluating Pneumonia Images"
echo "=========================================="
echo ""

PNEUMONIA_OUTPUT="${OUTPUT_BASE}/pneumonia_evaluation"

python scripts/evaluate_diffusion_generation.py \
    --generated-dir "${PNEUMONIA_GENERATED}" \
    --real-dir "${PNEUMONIA_REAL}" \
    --label Pneumonia \
    --output-dir "${PNEUMONIA_OUTPUT}" \
    --metrics novelty \
    --crop-border-pixels 10 \
    --healthy-images-dir "${HEALTHY}"

PNEUMONIA_EXIT=$?

if [ ${PNEUMONIA_EXIT} -eq 0 ]; then
    echo ""
    echo "✓ Pneumonia evaluation completed successfully"
    echo "  Results: ${PNEUMONIA_OUTPUT}"
else
    echo ""
    echo "✗ Pneumonia evaluation failed with exit code ${PNEUMONIA_EXIT}"
fi

echo ""

# ========================================
# Summary
# ========================================
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results directory: ${OUTPUT_BASE}"
echo ""

if [ ${FIBROSIS_EXIT} -eq 0 ] && [ ${PNEUMONIA_EXIT} -eq 0 ]; then
    echo "✓ Both evaluations completed successfully!"
    echo ""
    echo "Key results:"
    echo ""
    echo "Fibrosis:"
    if [ -f "${FIBROSIS_OUTPUT}/results.json" ]; then
        echo "  Results: ${FIBROSIS_OUTPUT}/results.json"
        echo "  Visualizations: ${FIBROSIS_OUTPUT}/"
    fi
    echo ""
    echo "Pneumonia:"
    if [ -f "${PNEUMONIA_OUTPUT}/results.json" ]; then
        echo "  Results: ${PNEUMONIA_OUTPUT}/results.json"
        echo "  Visualizations: ${PNEUMONIA_OUTPUT}/"
    fi
    echo ""
    echo "Summary files:"
    find "${OUTPUT_BASE}" -name "results.json" -o -name "*.png" | head -20

    EXIT_CODE=0
elif [ ${FIBROSIS_EXIT} -eq 0 ]; then
    echo "⚠ Only Fibrosis evaluation succeeded"
    EXIT_CODE=1
elif [ ${PNEUMONIA_EXIT} -eq 0 ]; then
    echo "⚠ Only Pneumonia evaluation succeeded"
    EXIT_CODE=1
else
    echo "✗ Both evaluations failed"
    EXIT_CODE=1
fi

echo "=========================================="

exit ${EXIT_CODE}
