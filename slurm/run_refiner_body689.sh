#!/bin/bash -l

#SBATCH -J refiner_body689
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=12:00:00
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
CIRCNET_RUN="mar_run_16838558"
RADIUS_FEATURES=32
TRAIN_FOLDERS="body6 body8"
VAL_FOLDER="body9"
TEST_FOLDER="body9"
REFINER_EPOCHS=50
REFINER_BATCH=4
REFINER_BASE=32
# =========================================================

REFINER_DIR=${TEAM_DIR}/CircNet/refiner_body689_run_${SLURM_JOB_ID}
LOG_DIR=${REFINER_DIR}/slurm_logs

mkdir -p ${REFINER_DIR}
mkdir -p ${LOG_DIR}

exec > >(tee ${LOG_DIR}/slurm.out) 2>&1

echo "[INFO] Job started: $(date)"
echo "[INFO] Running on: $(hostname)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] CircNet run: ${CIRCNET_RUN}"
echo "[INFO] Train: ${TRAIN_FOLDERS}  Val/Test: ${VAL_FOLDER}"
echo "[INFO] Run dir: ${REFINER_DIR}"

# =========================================================
# Krok 1: Inferencja CircNetu na body6, body8, body9
# =========================================================
echo "[INFO] Uruchamiam inferencję CircNetu dla body6 body8 body9..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/inference.py \
    --model-path      /app/CircNet/${CIRCNET_RUN}/best_model.pth \
    --data-root       /app/data/RPI/body \
    --test-folders    ${TRAIN_FOLDERS} ${VAL_FOLDER} \
    --output-dir      /app/CircNet/${CIRCNET_RUN}/corrected \
    --polar-size      512x1024 \
    --radius-features ${RADIUS_FEATURES} \
  2>&1 | tee ${LOG_DIR}/01_inference.log

if [ $? -ne 0 ]; then
    echo "[ERROR] Inferencja CircNetu nie powiodła się. Przerywam."
    exit 1
fi
echo "[INFO] Inferencja CircNetu zakończona."

# =========================================================
# Krok 2: Trening refinera [art, CircNet, seg_channel]
# =========================================================
echo "[INFO] Uruchamiam trening refinera (3 kanały: art, CircNet, graf)..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/circnet_refiner_train.py \
    --data-root        /app/data/RPI/body \
    --prior-dir        /app/CircNet/${CIRCNET_RUN}/corrected \
    --train-folders    ${TRAIN_FOLDERS} \
    --val-folders      ${VAL_FOLDER} \
    --seg-channel-dir  /app/data/seg_channels_body689 \
    --output-dir       /app/CircNet/refiner_body689_run_${SLURM_JOB_ID} \
    --epochs           ${REFINER_EPOCHS} \
    --batch-size       ${REFINER_BATCH} \
    --num-workers      8 \
    --base-features    ${REFINER_BASE} \
    --lr               2e-4 \
    --lam-out          1.0 \
    --lam-in           0.25 \
  2>&1 | tee ${LOG_DIR}/02_train_circnet_graph_3ch.log

if [ $? -ne 0 ]; then
    echo "[ERROR] Trening refinera nie powiódł się. Przerywam."
    exit 1
fi
echo "[INFO] Trening refinera zakończony."

# =========================================================
# Krok 3: Metryki na zbiorze testowym (body9)
# =========================================================
echo "[INFO] Liczę metryki na ${TEST_FOLDER}..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/circnet_refiner_metrics.py \
    --model-path       /app/CircNet/refiner_body689_run_${SLURM_JOB_ID}/best_refiner.pth \
    --data-root        /app/data/RPI/body \
    --prior-dir        /app/CircNet/${CIRCNET_RUN}/corrected \
    --seg-channel-dir  /app/data/seg_channels_body689 \
    --test-folder      ${TEST_FOLDER} \
    --output-dir       /app/CircNet/refiner_body689_run_${SLURM_JOB_ID}/metrics \
    --base-features    ${REFINER_BASE} \
  2>&1 | tee ${LOG_DIR}/03_metrics_circnet_graph_3ch_body9.log

echo "[INFO] Metryki zakończone."
echo "[INFO] Job finished: $(date)"
echo "[INFO] Run dir: ${REFINER_DIR}"
