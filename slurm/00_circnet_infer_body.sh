#!/bin/bash -l

#SBATCH -J circnet_infer_body
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
FOLDERS="body1 body2 body3 body4 body5 body6 body7 body8 body9 body10 body11 body12 body13"
# =========================================================

LOG_DIR=${TEAM_DIR}/CircNet/${CIRCNET_MODEL}/slurm_logs
mkdir -p ${LOG_DIR}

exec > >(tee ${LOG_DIR}/infer_body_${SLURM_JOB_ID}.out) 2>&1

echo "[INFO] Job started: $(date)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] Model: ${CIRCNET_MODEL}  radius_features: ${RADIUS_FEATURES}"
echo "[INFO] Foldery: ${FOLDERS}"

apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/inference.py \
    --model-path      /app/CircNet/${CIRCNET_MODEL}/best_model.pth \
    --data-root       /app/data/RPI/body \
    --test-folders    ${FOLDERS} \
    --output-dir      /app/CircNet/${CIRCNET_MODEL}/corrected \
    --polar-size      512x1024 \
    --radius-features ${RADIUS_FEATURES} \
  2>&1 | tee ${LOG_DIR}/infer_body_${SLURM_JOB_ID}.log

echo "[INFO] Job finished: $(date)"
echo "[INFO] Wyniki: ${TEAM_DIR}/CircNet/${CIRCNET_MODEL}/corrected"
