#!/bin/bash -l

#SBATCH -J metrics_jitter
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam

# =========================================================
# KONFIGURACJA
# =========================================================
REFINER_JOB_ID="17272148"
TEST_SCAN_ID=4
REFINER_BASE=32
# =========================================================

RUN_DIR=${TEAM_DIR}/CircNet/refiner_jitter_run_${REFINER_JOB_ID}
LOG_DIR=${RUN_DIR}/slurm_logs

mkdir -p ${RUN_DIR}/metrics
mkdir -p ${LOG_DIR}

exec > >(tee ${LOG_DIR}/metrics_nmar_graph_3ch_scan${TEST_SCAN_ID}_${SLURM_JOB_ID}.out) 2>&1

echo "[INFO] Job started: $(date)"
echo "[INFO] Running on: $(hostname)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] Ewaluacja: refiner_jitter_run_${REFINER_JOB_ID} (NMAR+graf 3ch, scan${TEST_SCAN_ID})"
echo "[INFO] Run dir: ${RUN_DIR}"

apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet,${TEAM_DIR}/output:/app/output \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/jitter_refiner_metrics.py \
    --model-path    /app/CircNet/refiner_jitter_run_${REFINER_JOB_ID}/best_refiner.pth \
    --art-dir       /app/output/3frames_jitter_raw_clean_real \
    --clean-dir     /app/data/Clean_raw \
    --prior-dir     /app/output/nmar_results \
    --prior-suffix  _nmar2 \
    --seg-chan-dir   /app/data/seg_channels_jitter \
    --test-scan-id  ${TEST_SCAN_ID} \
    --output-dir    /app/CircNet/refiner_jitter_run_${REFINER_JOB_ID}/metrics \
    --base-features ${REFINER_BASE} \
  2>&1 | tee ${LOG_DIR}/metrics_nmar_graph_3ch_scan${TEST_SCAN_ID}.log

echo "[INFO] Metryki zakończone."
echo "[INFO] Job finished: $(date)"
echo "[INFO] Run dir: ${RUN_DIR}"
