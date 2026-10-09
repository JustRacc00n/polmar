#!/bin/bash -l
#
# run_real_visualization.sbatch — Wizualizacja wyników inferencji na danych rzeczywistych
#
# Generuje PNG z porównaniem:
#   Artefakt | CircNet | Refiner Base+CircNet | Refiner Base+CircNet+SEG
#
# Uruchomienie:
#   sbatch run_real_visualization.sbatch
#   # Z węższym oknem (np. tkanka miękka):
#   sbatch --export=ALL,VIS_WW=400,VIS_WL=40 run_real_visualization.sbatch
#

#SBATCH -J vis_real
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output="logs/%x_%j.out"
#SBATCH --error="logs/%x_%j.err"

set -euo pipefail

export APPTAINER_CACHEDIR=/net/storage/pr3/plgrid/plggsmartudlteam/apptainer_cache
export PYTHONPATH="${PYTHONPATH:-}:/"

IMG="/net/storage/pr3/plgrid/plggsmartudlteam/sif_images/sif_image.sif"

# Window width / level — nadpisz przez --export
VIS_WW="${VIS_WW:-3000}"
VIS_WL="${VIS_WL:-0}"

mkdir -p logs
mkdir -p /net/storage/pr3/plgrid/plggsmartudlteam/output/real_vis

apptainer exec \
  --bind /net/storage/pr3/plgrid/plggsmartudlteam/data:/app/data,/net/storage/pr3/plgrid/plggsmartudlteam/output:/app/output,/net/storage/pr3/plgrid/plggsmartudlteam/CircNet:/app/CircNet \
  "$IMG" \
  bash -lc "
    set -euo pipefail

    # Sprawdź czy katalog SEG istnieje
    SEG_ARG=''
    if [ -d '/app/output/real_refiner_angle_graph' ] && \
       [ \"\$(ls /app/output/real_refiner_angle_graph/*.raw 2>/dev/null | wc -l)\" -gt 0 ]; then
      SEG_ARG='--refiner-seg-dir /app/output/real_refiner_angle_graph'
    fi

    python /app/CircNet/runs/visualize_real_results.py \
      --art-dir       /app/data/real_raw \
      --circnet-dir   /app/output/real_circnet \
      --refiner-dir   /app/output/real_refiner_angle \
      \${SEG_ARG} \
      --out-dir       /app/output/real_vis \
      --ww ${VIS_WW} \
      --wl ${VIS_WL}
  "

echo "PNG zapisane w: /net/storage/pr3/plgrid/plggsmartudlteam/output/real_vis/"
