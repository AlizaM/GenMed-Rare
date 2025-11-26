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

echo "=========================================="
echo "Submitting X-ray Generation Jobs"
echo "=========================================="

# Make the generation script executable
chmod +x generate_synthetic_xrays.sh

# Submit Pneumonia generation job (2000 images)
echo ""
echo "Submitting Pneumonia generation job..."
PNEUMONIA_JOB=$(sbatch --parsable generate_synthetic_xrays.sh Pneumonia 2000)
echo "Pneumonia job ID: ${PNEUMONIA_JOB}"

# Submit Fibrosis generation job (2000 images) - will start after Pneumonia completes
echo ""
echo "Submitting Fibrosis generation job (dependent on Pneumonia)..."
FIBROSIS_JOB=$(sbatch --parsable --dependency=afterok:${PNEUMONIA_JOB} generate_synthetic_xrays.sh Fibrosis 2000)
echo "Fibrosis job ID: ${FIBROSIS_JOB}"

echo ""
echo "=========================================="
echo "Jobs Submitted Successfully"
echo "=========================================="
echo "Pneumonia: Job ${PNEUMONIA_JOB}"
echo "Fibrosis:  Job ${FIBROSIS_JOB} (starts after Pneumonia completes)"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  /w/\$USER/xray_generation_${PNEUMONIA_JOB}.log"
echo "  /w/\$USER/xray_generation_${FIBROSIS_JOB}.log"
echo ""
echo "Cancel jobs with:"
echo "  scancel ${PNEUMONIA_JOB}"
echo "  scancel ${FIBROSIS_JOB}"
echo "=========================================="
