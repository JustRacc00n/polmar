#!/bin/bash -l

#SBATCH -J nmar_inference
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:0
#SBATCH --output="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet/slurm_logs/nmar_inference_%j.out"
#SBATCH --error="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet/slurm_logs/nmar_inference_%j.err"

TEAM_DIR=$PLG_GROUPS_STORAGE/plggsmartudlteam

# =========================================================
# KONFIGURACJA
# =========================================================
INPUT_DIR=${TEAM_DIR}/output/3frames_jitter_raw_clean_real
OUTPUT_DIR=${TEAM_DIR}/output/nmar_results
NUM_WORKERS=16   # NMAR jest CPU-only — można użyć więcej wątków
# =========================================================

mkdir -p ${TEAM_DIR}/CircNet/slurm_logs
mkdir -p ${OUTPUT_DIR}

echo "[INFO] Job started: $(date)"
echo "[INFO] Running on: $(hostname)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] Input dir:  ${INPUT_DIR}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Workers:    ${NUM_WORKERS}"

apptainer exec \
  --bind ${TEAM_DIR}:/app \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/nmar_inference.py \
    --input-dir   /app/output/3frames_jitter_raw_clean_real \
    --output-dir  /app/output/nmar_results \
    --num-workers ${NUM_WORKERS} \
  2>&1 | tee ${TEAM_DIR}/CircNet/slurm_logs/nmar_inference_${SLURM_JOB_ID}.log

echo "[INFO] Job finished: $(date)"
echo "[INFO] Wyniki w: ${OUTPUT_DIR}"
