#!/bin/bash -l
#SBATCH -J nnunet_predict_real
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpu-bind=closest
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output="logs/%x_%j.out"
#SBATCH --error="logs/%x_%j.err"
#SBATCH --gres=gpu:1

set -euo pipefail

export CUDA_LAUNCH_BLOCKING=1
export APPTAINER_CACHEDIR=/net/storage/pr3/plgrid/plggsmartudlteam/apptainer_cache
export PYTHONPATH="${PYTHONPATH:-}:/"

IMG="/net/storage/pr3/plgrid/plggsmartudlteam/sif_images/sif_image.sif"

DATASET_ID=701
CFG="2d"
TRAINER="nnUNetTrainer"
FOLDS="0 1 2 3 4"

# Inputs/outputs (paths are inside the apptainer container; bind mounts below
# point them at the real network storage.)
IN_DIR="/app/output/real_raw_nifti"
OUT_DIR="/app/output/real_raw_seg"

mkdir -p logs
mkdir -p /net/storage/pr3/plgrid/plggsmartudlteam/output/real_raw_seg

apptainer exec --nv \
  --bind /net/storage/pr3/plgrid/plggsmartudlteam/data:/app/data,/net/storage/pr3/plgrid/plggsmartudlteam/output:/app/output,/net/storage/pr3/plgrid/plggsmartudlteam/nnUNet_output:/app/nnUNet_output \
  "$IMG" \
  bash -lc "
    set -euo pipefail

    export nnUNet_raw=/app/data/nnunet/nnUNet_raw
    export nnUNet_preprocessed=/app/nnUNet_output/nnunet/nnUNet_preprocessed
    export nnUNet_results=/app/nnUNet_output/nnunet/nnUNet_results

    mkdir -p '${OUT_DIR}'

    nnUNetv2_predict \
      -d ${DATASET_ID} \
      -c ${CFG} \
      -i '${IN_DIR}' \
      -o '${OUT_DIR}' \
      -f ${FOLDS} \
      -tr ${TRAINER}
  "
