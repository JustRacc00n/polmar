#!/bin/bash -l
# Generowanie kanałów grafowych dla datasetu jitter
# Wejście:  output/3framesjitterRawRealNifti/  (CT w NIfTI)
#           output/3framesjitterRawRealSeg/     (segmentacje nnUNet)
# Wyjście:  data/seg_channels_jitter/

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

export APPTAINER_CACHEDIR=/net/storage/pr3/plgrid/plggsmartudlteam/apptainer_cache
export PYTHONPATH="$PYTHONPATH:/"

TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam

# =========================================================
# KONFIGURACJA
# =========================================================
NUM_WORKERS=${GRAPH_WORKERS:-12}
Z_NEIGHBORS=${GRAPH_Z_NEIGHBORS:-2}
Z_DECAY=${GRAPH_Z_DECAY:-0.7}
# =========================================================

RUN_DIR=${TEAM_DIR}/CircNet/graph_jitter_run_${SLURM_JOB_ID}
LOG_DIR=${RUN_DIR}/slurm_logs

mkdir -p ${TEAM_DIR}/data/seg_channels_jitter
mkdir -p ${LOG_DIR}

exec > >(tee ${LOG_DIR}/slurm.out) 2>&1

echo "[INFO] Job started: $(date)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] Workers: ${NUM_WORKERS}  z_neighbors: ${Z_NEIGHBORS}  z_decay: ${Z_DECAY}"

# Sprawdź czy wymagane wejścia istnieją
CT_DIR="${TEAM_DIR}/output/3framesjitterRawRealNifti"
SEG_DIR="${TEAM_DIR}/output/3framesjitterRawRealSeg"

if [[ ! -d "${CT_DIR}" ]]; then
  echo "[ERROR] Brak katalogu NIfTI CT: ${CT_DIR}"
  exit 1
fi
if [[ ! -d "${SEG_DIR}" ]]; then
  echo "[ERROR] Brak katalogu segmentacji: ${SEG_DIR}"
  exit 1
fi

apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet,${TEAM_DIR}/output:/app/output \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/graph_channel_generator.py \
    --mode          jitter_nii \
    --ct-dir        /app/output/3framesjitterRawRealNifti \
    --seg-dir       /app/output/3framesjitterRawRealSeg \
    --out-dir       /app/data/seg_channels_jitter \
    --num-workers   ${NUM_WORKERS} \
    --z-neighbors   ${Z_NEIGHBORS} \
    --z-decay       ${Z_DECAY} \
  2>&1 | tee ${LOG_DIR}/graph_jitter.log

echo "[INFO] Job finished: $(date)"
echo "[INFO] Wyniki: ${TEAM_DIR}/data/seg_channels_jitter"
