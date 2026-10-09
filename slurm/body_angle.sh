#!/bin/bash -l

#SBATCH -J body_angle
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
CIRCNET_MODEL=${CIRCNET_MODEL:-"circnet_body_64f_16838558"}
TRAIN_FOLDERS="body1 body2 body3 body4 body6 body7 body8 body10 body11 body12 body13"
VAL_FOLDER="body4"
TEST_FOLDER="body5"
TEST_FOLDER2="body9"
REFINER_EPOCHS=${REFINER_EPOCHS:-200}
REFINER_BATCH=${REFINER_BATCH:-4}
REFINER_BASE=${REFINER_BASE:-128}
# =========================================================

_RUNS_ROOT=${BATCH_DIR:-${TEAM_DIR}/CircNet/runs}
RUN_DIR=${_RUNS_ROOT}/refiner_body_angle_${REFINER_BASE}f_${SLURM_JOB_ID}
CONTAINER_RUN_DIR=/app/CircNet${RUN_DIR#${TEAM_DIR}/CircNet}
LOG_DIR=${RUN_DIR}/slurm_logs

mkdir -p ${RUN_DIR}
mkdir -p ${LOG_DIR}

exec > >(tee ${LOG_DIR}/slurm.out) 2>&1

echo "[INFO] Job started: $(date)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] Wariant: body / prior=Angle / base=${REFINER_BASE}"
echo "[INFO] CircNet model: ${CIRCNET_MODEL}"
echo "[INFO] Run dir: ${RUN_DIR}"

# =========================================================
# Krok 0: inference CircNet na folderach extra (poza body1–5)
# =========================================================
EXTRA_FOLDERS="body6 body7 body8 body10 body11 body12 body13"
echo "[INFO] Generuję korekcje CircNet dla: ${EXTRA_FOLDERS}..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/inference.py \
    --model-path   /app/CircNet/${CIRCNET_MODEL}/best_model.pth \
    --data-root    /app/data/RPI/body \
    --test-folders ${EXTRA_FOLDERS} \
    --output-dir   /app/CircNet/${CIRCNET_MODEL}/corrected \
  2>&1 | tee ${LOG_DIR}/00_inference.log

if [ $? -ne 0 ]; then
    echo "[ERROR] Inference nie powiodła się. Przerywam."
    exit 1
fi
echo "[INFO] Inference zakończona."

# =========================================================
# Trening refinera [art, Angle]
# =========================================================
echo "[INFO] Trening refinera..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/circnet_refiner_train.py \
    --data-root        /app/data/RPI/body \
    --prior-dir        /app/CircNet/${CIRCNET_MODEL}/corrected \
    --train-folders    ${TRAIN_FOLDERS} \
    --val-folders      ${VAL_FOLDER} \
    --output-dir       ${CONTAINER_RUN_DIR} \
    --epochs           ${REFINER_EPOCHS} \
    --batch-size       ${REFINER_BATCH} \
    --num-workers      8 \
    --base-features    ${REFINER_BASE} \
    --lr               2e-4 \
    --lam-out          1.0 \
    --lam-in           0.25 \
  2>&1 | tee ${LOG_DIR}/01_train.log

if [ $? -ne 0 ]; then
    echo "[ERROR] Trening nie powiódł się. Przerywam."
    exit 1
fi
echo "[INFO] Trening zakończony."

# =========================================================
# Metryki na zbiorze testowym
# =========================================================
echo "[INFO] Liczę metryki na ${TEST_FOLDER}..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/circnet_refiner_metrics.py \
    --model-path       ${CONTAINER_RUN_DIR}/best_refiner.pth \
    --data-root        /app/data/RPI/body \
    --prior-dir        /app/CircNet/${CIRCNET_MODEL}/corrected \
    --test-folder      ${TEST_FOLDER} \
    --output-dir       ${CONTAINER_RUN_DIR}/metrics \
    --base-features    ${REFINER_BASE} \
  2>&1 | tee ${LOG_DIR}/02_metrics_${TEST_FOLDER}.log

echo "[INFO] Liczę metryki na ${TEST_FOLDER2}..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/circnet_refiner_metrics.py \
    --model-path       ${CONTAINER_RUN_DIR}/best_refiner.pth \
    --data-root        /app/data/RPI/body \
    --prior-dir        /app/CircNet/${CIRCNET_MODEL}/corrected \
    --test-folder      ${TEST_FOLDER2} \
    --output-dir       ${CONTAINER_RUN_DIR}/metrics \
    --base-features    ${REFINER_BASE} \
  2>&1 | tee ${LOG_DIR}/02_metrics_${TEST_FOLDER2}.log

echo "[INFO] Job finished: $(date)"
echo "[INFO] Run dir: ${RUN_DIR}"
