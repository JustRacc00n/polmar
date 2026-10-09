#!/bin/bash -l

#SBATCH -J mar_training
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --output="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet/slurm_logs/training_%j.out"
#SBATCH --error="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet/slurm_logs/training_%j.err"

export CUDA_LAUNCH_BLOCKING=1
export APPTAINER_CACHEDIR=$PLG_GROUPS_STORAGE/plggsmartudlteam/apptainer_cache
export PYTHONPATH="$PYTHONPATH:/"

TEAM_DIR=$PLG_GROUPS_STORAGE/plggsmartudlteam

mkdir -p ${TEAM_DIR}/CircNet/slurm_logs

echo "[INFO] Job started: $(date)"
echo "[INFO] Running on node: $(hostname)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"

# ---------- eksperyment 1 ----------
echo "[INFO] Starting mar_run"
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,\
${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/train.py \
    --data-root     /app/data/RPI/body \
    --train-folders body1 body2 body3 \
    --val-folders   body4 \
    --batch-size    2 \
    --num-workers   0 \
    --epochs        100 \
    --output-dir    /app/CircNet \
    --run-name      mar_run_$SLURM_JOB_ID \
    --polar-size    512x1024 \
    --radius-features 64 \
    --l1-w          0.60 \
    --mse-w         0.15 \
    --zero-w        0.20 \
    --range-w       0.05 \
  2>&1 | tee ${TEAM_DIR}/CircNet/slurm_logs/mar_run_${SLURM_JOB_ID}.log
echo "[INFO] Finished mar_run"

echo "[INFO] Job finished: $(date)"
