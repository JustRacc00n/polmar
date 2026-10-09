#!/bin/bash -l

#SBATCH -J graph_jitter
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam
RUN_DIR=${TEAM_DIR}/CircNet/graph_jitter_run_${SLURM_JOB_ID}
LOG_DIR=${RUN_DIR}/slurm_logs

mkdir -p ${TEAM_DIR}/data/seg_channels_jitter
mkdir -p ${LOG_DIR}

exec > >(tee ${LOG_DIR}/slurm.out) 2>&1

echo "[INFO] Job started: $(date)"
echo "[INFO] Running on: $(hostname)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] Run dir: ${RUN_DIR}"

apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet,${TEAM_DIR}/output:/app/output \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/graph_channel_generator.py \
    --mode          jitter_nii \
    --ct-dir        /app/output/3framesjitterRawRealNifti \
    --seg-dir       /app/output/3framesjitterRawRealSeg \
    --out-dir       /app/data/seg_channels_jitter \
    --num-workers   12 \
    --z-neighbors   2 \
    --z-decay       0.7 \
  2>&1 | tee ${LOG_DIR}/graph_jitter.log

echo "[INFO] Job finished: $(date)"
echo "[INFO] Run dir: ${RUN_DIR}"
