#!/bin/bash -l

#SBATCH -J mar_metrics
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --output="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet/slurm_logs/metrics_%j.out"
#SBATCH --error="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet/slurm_logs/metrics_%j.err"

export APPTAINER_CACHEDIR=$PLG_GROUPS_STORAGE/plggsmartudlteam/apptainer_cache

TEAM_DIR=$PLG_GROUPS_STORAGE/plggsmartudlteam

# Ustaw na folder z modelem do ewaluacji
RUN_NAME="mar_run_16838558"

mkdir -p ${TEAM_DIR}/CircNet/slurm_logs

echo "[INFO] Job started: $(date)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] Evaluating: ${RUN_NAME}"

apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,\
${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/compute_metrics.py \
    --model-path      /app/CircNet/${RUN_NAME}/best_model.pth \
    --data-root       /app/data/RPI/body \
    --test-folder     body5 \
    --output-dir      /app/CircNet/${RUN_NAME}/metrics \
    --polar-size      512x1024 \
    --radius-features 64 \
    --device          cuda \
  2>&1 | tee ${TEAM_DIR}/CircNet/slurm_logs/metrics_${SLURM_JOB_ID}.log

echo "[INFO] Job finished: $(date)"
