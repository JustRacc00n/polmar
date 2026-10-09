#!/bin/bash -l
#SBATCH -J raw2nii_real
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output="logs/%x_%j.out"
#SBATCH --error="logs/%x_%j.err"

set -euo pipefail

export APPTAINER_CACHEDIR=/net/storage/pr3/plgrid/plggsmartudlteam/apptainer_cache
export PYTHONPATH="${PYTHONPATH:-}:/"

IMG="/net/storage/pr3/plgrid/plggsmartudlteam/sif_images/sif_image.sif"

RAW_ROOT_HOST="/net/storage/pr3/plgrid/plggsmartudlteam/data/real_raw"
NIFTI_OUT_HOST="/net/storage/pr3/plgrid/plggsmartudlteam/output/real_raw_nifti"
SCRIPT_HOST="/net/storage/pr3/plgrid/plggsmartudlteam/nnUNet_jobs/tools/raw_to_nifti_batch.py"

# Override via environment if you need to smoke-test on a few files first:
#   sbatch --export=ALL,RAW_LIMIT=5 convert_real_raw_to_nifti.sbatch
RAW_LIMIT="${RAW_LIMIT:-0}"
RAW_DTYPE="${RAW_DTYPE:-float32}"
RAW_ORDER="${RAW_ORDER:-C}"

mkdir -p logs
mkdir -p "${NIFTI_OUT_HOST}"

apptainer exec \
  --bind /net/storage/pr3/plgrid/plggsmartudlteam/data:/app/data,/net/storage/pr3/plgrid/plggsmartudlteam/output:/app/output,/net/storage/pr3/plgrid/plggsmartudlteam/nnUNet_jobs:/app/nnUNet_jobs \
  "$IMG" \
  bash -lc "
    set -euo pipefail

    python /app/nnUNet_jobs/tools/raw_to_nifti_batch.py \
      --input-root /app/data/real_raw \
      --output-root /app/output/real_raw_nifti \
      --dtype ${RAW_DTYPE} \
      --order ${RAW_ORDER} \
      --default-h 512 \
      --default-w 512 \
      --spacing-x 1.0 \
      --spacing-y 1.0 \
      --spacing-z 1.0 \
      --limit ${RAW_LIMIT}
  "
