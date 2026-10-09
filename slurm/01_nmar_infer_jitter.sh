#!/bin/bash -l
# NMAR inference na datasecie jitter (CPU-only)
# Wejście:  output/3frames_jitter_raw_clean_real/
# Wyjście:  output/nmar_results_2/

#SBATCH -J nmar_infer_jitter
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

export APPTAINER_CACHEDIR=/net/storage/pr3/plgrid/plggsmartudlteam/apptainer_cache
export PYTHONPATH="$PYTHONPATH:/"

TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam

# =========================================================
# KONFIGURACJA
# =========================================================
NUM_WORKERS=${NMAR_WORKERS:-16}
# =========================================================

LOG_DIR=${TEAM_DIR}/CircNet/slurm_logs
mkdir -p ${LOG_DIR}
mkdir -p ${TEAM_DIR}/output/nmar_results_2

exec > >(tee ${LOG_DIR}/nmar_infer_jitter_${SLURM_JOB_ID}.out) 2>&1

echo "[INFO] Job started: $(date)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] Workers: ${NUM_WORKERS}"

apptainer exec \
  --bind ${TEAM_DIR}/output:/app/output,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/nmar_inference.py \
    --input-dir   /app/output/3frames_jitter_raw_clean_real \
    --output-dir  /app/output/nmar_results_2 \
    --num-workers ${NUM_WORKERS} \
  2>&1 | tee ${LOG_DIR}/nmar_infer_jitter_${SLURM_JOB_ID}.log

echo "[INFO] Job finished: $(date)"
echo "[INFO] Wyniki: ${TEAM_DIR}/output/nmar_results_2"
