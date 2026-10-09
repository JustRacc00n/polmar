#!/bin/bash -l

#SBATCH -J refiner_jitter_circnet
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
TRAIN_SCAN_IDS="1 2"
VAL_SCAN_IDS="3"
REFINER_EPOCHS=50
REFINER_BATCH=4
REFINER_BASE=32
# =========================================================

REFINER_DIR=${TEAM_DIR}/CircNet/refiner_jitter_circnet_run_${SLURM_JOB_ID}
LOG_DIR=${REFINER_DIR}/slurm_logs

mkdir -p ${REFINER_DIR}
mkdir -p ${LOG_DIR}

exec > >(tee ${LOG_DIR}/slurm.out) 2>&1

echo "[INFO] Job started: $(date)"
echo "[INFO] Running on: $(hostname)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] Train scans: ${TRAIN_SCAN_IDS}  Val scans: ${VAL_SCAN_IDS}"
echo "[INFO] Run dir: ${REFINER_DIR}"

# =========================================================
# Trening refinera [art, CircNet, seg_channel]
# =========================================================
echo "[INFO] Uruchamiam trening refinera jitter (3 kanały: art, CircNet, graf)..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet,${TEAM_DIR}/output:/app/output \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/jitter_refiner_train.py \
    --art-dir         /app/output/3frames_jitter_raw_clean_real \
    --clean-dir       /app/data/Clean_raw \
    --prior-dir       /app/output/circnet_jitter_results \
    --prior-suffix    _circnet \
    --seg-chan-dir    /app/data/seg_channels_jitter \
    --train-scan-ids  ${TRAIN_SCAN_IDS} \
    --val-scan-ids    ${VAL_SCAN_IDS} \
    --output-dir      /app/CircNet/refiner_jitter_circnet_run_${SLURM_JOB_ID} \
    --epochs          ${REFINER_EPOCHS} \
    --batch-size      ${REFINER_BATCH} \
    --num-workers     8 \
    --base-features   ${REFINER_BASE} \
    --lr              2e-4 \
    --lam-out         1.0 \
    --lam-in          0.25 \
  2>&1 | tee ${LOG_DIR}/01_train_circnet_graph_3ch.log

if [ $? -ne 0 ]; then
    echo "[ERROR] Trening refinera nie powiódł się. Przerywam."
    exit 1
fi

echo "[INFO] Trening zakończony."
echo "[INFO] Job finished: $(date)"
echo "[INFO] Run dir: ${REFINER_DIR}"
