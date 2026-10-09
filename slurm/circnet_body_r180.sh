#!/bin/bash -l
# ============================================================
# circnet_body_r180.sh — Trening CircNet 128f z radialnym cutoffem r_train=180px
#
# Wariant testowy: identyczny z circnet_body.sh ale używa train_radial.py
# (ScalarCorrectionLossRadial zamiast ScalarCorrectionLoss).
# Gradientu L1/MSE nie wnoszą piksele dalej niż 180px od centrum metalu.
#
# Uruchomienie:
#   sbatch circnet_body_r180.sh
#   sbatch --export=ALL,RADIUS_FEATURES=128,TRAIN_EPOCHS=200 circnet_body_r180.sh
# ============================================================

#SBATCH -J circnet_r180
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

export CUDA_LAUNCH_BLOCKING=1
export APPTAINER_CACHEDIR=/net/storage/pr3/plgrid/plggsmartudlteam/apptainer_cache
export PYTHONPATH="$PYTHONPATH:/"

TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam

# =========================================================
# KONFIGURACJA
# =========================================================
RADIUS_FEATURES=${RADIUS_FEATURES:-128}    # najlepszy wariant z sweepU
R_TRAIN=${R_TRAIN:-180.0}                  # cutoff radialny [px]
TRAIN_EPOCHS=${TRAIN_EPOCHS:-200}
TRAIN_BATCH=${TRAIN_BATCH:-2}
TRAIN_FOLDERS="body1 body2 body3"
VAL_FOLDER="body4"
# =========================================================

RUN_NAME=circnet_body_r180_${RADIUS_FEATURES}f_${SLURM_JOB_ID}
RUN_DIR=${TEAM_DIR}/CircNet/${RUN_NAME}
LOG_DIR=${RUN_DIR}/slurm_logs

mkdir -p ${RUN_DIR}
mkdir -p ${LOG_DIR}

exec > >(tee ${LOG_DIR}/slurm.out) 2>&1

echo "[INFO] Job started: $(date)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] Run name: ${RUN_NAME}"
echo "[INFO] radius_features=${RADIUS_FEATURES}  r_train=${R_TRAIN}px  epochs=${TRAIN_EPOCHS}  batch=${TRAIN_BATCH}"

apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/train_radial.py \
    --data-root       /app/data/RPI/body \
    --train-folders   ${TRAIN_FOLDERS} \
    --val-folders     ${VAL_FOLDER} \
    --batch-size      ${TRAIN_BATCH} \
    --num-workers     8 \
    --epochs          ${TRAIN_EPOCHS} \
    --output-dir      /app/CircNet \
    --run-name        ${RUN_NAME} \
    --polar-size      512x1024 \
    --radius-features ${RADIUS_FEATURES} \
    --r-train         ${R_TRAIN} \
    --l1-w            0.60 \
    --mse-w           0.15 \
    --zero-w          0.20 \
    --range-w         0.05 \
  2>&1 | tee ${LOG_DIR}/01_train.log

if [ $? -ne 0 ]; then
    echo "[ERROR] Trening nie powiódł się. Przerywam."
    exit 1
fi

echo "[INFO] Job finished: $(date)"
echo "[INFO] Model: ${RUN_DIR}"
