#!/bin/bash -l

#SBATCH -J circnet_infer_jitter
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

export APPTAINER_CACHEDIR=/net/storage/pr3/plgrid/plggsmartudlteam/apptainer_cache
export PYTHONPATH="$PYTHONPATH:/"

TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam

# =========================================================
# KONFIGURACJA
# =========================================================
CIRCNET_MODEL=${CIRCNET_MODEL:-"circnet_body_64f_16838558"}
RADIUS_FEATURES=${RADIUS_FEATURES:-64}
# =========================================================

RUN_DIR=${TEAM_DIR}/CircNet/${CIRCNET_MODEL}
OUTPUT_DIR=${RUN_DIR}/corrected_jitter
LOG_DIR=${RUN_DIR}/slurm_logs

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

exec > >(tee ${LOG_DIR}/slurm.out) 2>&1

echo "[INFO] Job started: $(date)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] Model: ${CIRCNET_MODEL}  radius_features: ${RADIUS_FEATURES}"

apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet,${TEAM_DIR}/output:/app/output \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/circnet_jitter_inference.py \
    --model-path      /app/CircNet/${CIRCNET_MODEL}/best_model.pth \
    --input-dir       /app/output/3frames_jitter_raw_clean_real \
    --output-dir      /app/CircNet/${CIRCNET_MODEL}/corrected_jitter \
    --polar-size      512x1024 \
    --radius-features ${RADIUS_FEATURES} \
  2>&1 | tee ${LOG_DIR}/circnet_infer_jitter.log

echo "[INFO] Job finished: $(date)"
echo "[INFO] Wyniki: ${OUTPUT_DIR}"
