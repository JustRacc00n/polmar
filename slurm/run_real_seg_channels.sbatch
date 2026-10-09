#!/bin/bash -l
#
# run_real_seg_channels.sbatch — Generuje kanały SEG dla rzeczywistych danych szpitalnych
#
# Wejście:
#   CT NIfTI:  /app/output/real_raw_nifti/   (metal{NN}_slice{NNNN}_H512_W512_0000.nii.gz)
#   SEG NIfTI: /app/output/real_raw_seg/      (metal{NN}_slice{NNNN}_H512_W512.nii.gz)
# Wyjście:
#   /app/data/real_seg_channels/               (metal{NN}_slice{NNNN}_H512_W512_seg_channel.raw)
#
# Uruchomienie:
#   sbatch run_real_seg_channels.sbatch
#   # Smoke-test (5 plików):
#   sbatch --export=ALL,SEG_LIMIT=5 run_real_seg_channels.sbatch
#

#SBATCH -J seg_channels_real
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --output="logs/%x_%j.out"
#SBATCH --error="logs/%x_%j.err"

set -euo pipefail

export APPTAINER_CACHEDIR=/net/storage/pr3/plgrid/plggsmartudlteam/apptainer_cache
export PYTHONPATH="${PYTHONPATH:-}:/"

IMG="/net/storage/pr3/plgrid/plggsmartudlteam/sif_images/sif_image.sif"
CIRCNET_DIR="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet"

# Ogranicz liczbę przetwarzanych plików (0 = bez limitu)
SEG_LIMIT="${SEG_LIMIT:-0}"

mkdir -p logs
mkdir -p /net/storage/pr3/plgrid/plggsmartudlteam/data/real_seg_channels

apptainer exec \
  --bind /net/storage/pr3/plgrid/plggsmartudlteam/data:/app/data,/net/storage/pr3/plgrid/plggsmartudlteam/output:/app/output,/net/storage/pr3/plgrid/plggsmartudlteam/CircNet:/app/CircNet \
  "$IMG" \
  bash -lc "
    set -euo pipefail

    LIMIT_ARG=''
    if [ '${SEG_LIMIT}' -gt 0 ]; then
      LIMIT_ARG=\"--limit ${SEG_LIMIT}\"
    fi

    python /app/CircNet/runs/graph_channel_generator.py \
      --mode        real_nii \
      --ct-dir      /app/output/real_raw_nifti \
      --seg-dir     /app/output/real_raw_seg \
      --out-dir     /app/data/real_seg_channels \
      --num-workers 14 \
      --z-neighbors 2 \
      --z-decay     0.7 \
      \${LIMIT_ARG}
  "

echo "Kanały SEG zapisane w: /net/storage/pr3/plgrid/plggsmartudlteam/data/real_seg_channels/"
