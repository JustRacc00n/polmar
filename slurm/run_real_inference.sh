#!/bin/bash -l
#
# run_real_inference.sh — Pipeline inferencji PolMAR na rzeczywistych danych szpitalnych
#
# Kroki:
#   1. CircNet inference         → real_circnet/
#   2. Refiner Base+CircNet      → real_refiner_angle/
#   3. Refiner Base+CircNet+SEG  → real_refiner_angle_graph/  (wymaga real_seg_channels/)
#
# Warunek wstępny dla kroku 3:
#   Katalog /app/data/real_seg_channels/ musi zawierać pliki *_seg_channel.raw
#   wygenerowane przez nnU-Net + graph_channel_generator.py na wolumenach real_raw.
#   Jeśli niedostępne, ustaw RUN_SEG=0 aby pominąć krok 3.
#

#SBATCH -J real_inference
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

export APPTAINER_CACHEDIR=/net/storage/pr3/plgrid/plggsmartudlteam/apptainer_cache
export PYTHONPATH="$PYTHONPATH:/"

TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURACJA
# ══════════════════════════════════════════════════════════════════════════════

# CircNet — model trenowany na danych jitter (ten sam używany do ewaluacji jitter)
CIRCNET_MODEL="circnet_body_64f_16838558"
CIRCNET_RADIUS_FEATURES=64
CIRCNET_POLAR="512x1024"

# Refiner Base+CircNet
REFINER_ANGLE_MODEL="runs/jitter_circ128f_ref128f_20260526_1829/refiner_jitter_angle_128f_17698393"
REFINER_BASE=128

# Refiner Base+CircNet+SEG
REFINER_ANGLE_GRAPH_MODEL="runs/jitter_circ128f_ref128f_20260526_1829/refiner_jitter_angle-graph_128f_17698395"

# Czy uruchomić wariant z SEG? (wymaga real_seg_channels/)
RUN_SEG=1

# ══════════════════════════════════════════════════════════════════════════════

LOG_DIR=${TEAM_DIR}/CircNet/runs/real_inference_${SLURM_JOB_ID}/slurm_logs
mkdir -p ${LOG_DIR}

exec > >(tee ${LOG_DIR}/slurm.out) 2>&1

echo "═══════════════════════════════════════════════════════"
echo " PolMAR — inferencja na danych rzeczywistych"
echo " Job ID:    ${SLURM_JOB_ID}"
echo " Data:      $(date)"
echo " Host:      $(hostname)"
echo " CircNet:   ${CIRCNET_MODEL}"
echo " Refiner:   ${REFINER_ANGLE_MODEL}  (angle)"
if [ "$RUN_SEG" -eq 1 ]; then
echo " Refiner:   ${REFINER_ANGLE_GRAPH_MODEL}  (angle+graph)"
fi
echo "═══════════════════════════════════════════════════════"

# ── Krok 1: CircNet inference ─────────────────────────────────────────────────
echo ""
echo "[KROK 1] CircNet inference na real_raw..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet,${TEAM_DIR}/output:/app/output \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/runs/real_circnet_inference.py \
    --model-path      /app/CircNet/${CIRCNET_MODEL}/best_model.pth \
    --input-dir       /app/data/real_raw \
    --output-dir      /app/output/real_circnet \
    --polar-size      ${CIRCNET_POLAR} \
    --radius-features ${CIRCNET_RADIUS_FEATURES} \
    --metal-hu        2500 \
  || { echo "[ERROR] CircNet inference nie powiódł się. Przerywam."; exit 1; }

echo "[KROK 1] CircNet gotowy → /app/output/real_circnet"

# ── Krok 2: Refiner Base+CircNet ─────────────────────────────────────────────
echo ""
echo "[KROK 2] Refiner Base+CircNet..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet,${TEAM_DIR}/output:/app/output \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/runs/real_refiner_inference.py \
    --model-path    /app/CircNet/${REFINER_ANGLE_MODEL}/best_refiner.pth \
    --art-dir       /app/data/real_raw \
    --circnet-dir   /app/output/real_circnet \
    --output-dir    /app/output/real_refiner_angle \
    --base-features ${REFINER_BASE} \
    --metal-hu      2500 \
  || { echo "[ERROR] Refiner (angle) nie powiódł się. Przerywam."; exit 1; }

echo "[KROK 2] Refiner (Base+CircNet) gotowy → /app/output/real_refiner_angle"

# ── Krok 3: Refiner Base+CircNet+SEG (opcjonalny) ────────────────────────────
if [ "$RUN_SEG" -eq 1 ]; then
    echo ""
    echo "[KROK 3] Refiner Base+CircNet+SEG..."

    if [ ! -d "${TEAM_DIR}/data/real_seg_channels" ]; then
        echo "[WARN] Katalog real_seg_channels/ nie istnieje."
        echo "       Uruchom najpierw nnU-Net na wolumenach real_raw,"
        echo "       a następnie graph_channel_generator.py z trybem real."
        echo "       Pomijam krok 3."
    else
        apptainer exec --nv \
          --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet,${TEAM_DIR}/output:/app/output \
          ${TEAM_DIR}/sif_images/sif_image.sif \
          python3 /app/CircNet/runs/real_refiner_inference.py \
            --model-path    /app/CircNet/${REFINER_ANGLE_GRAPH_MODEL}/best_refiner.pth \
            --art-dir       /app/data/real_raw \
            --circnet-dir   /app/output/real_circnet \
            --seg-chan-dir  /app/data/real_seg_channels \
            --output-dir    /app/output/real_refiner_angle_graph \
            --base-features ${REFINER_BASE} \
            --metal-hu      2500 \
          || { echo "[ERROR] Refiner (angle+graph) nie powiódł się. Przerywam."; exit 1; }

        echo "[KROK 3] Refiner (Base+CircNet+SEG) gotowy → /app/output/real_refiner_angle_graph"
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Pipeline zakończony: $(date)"
echo " Wyniki:"
echo "   CircNet:          /app/output/real_circnet/"
echo "   Base+CircNet:     /app/output/real_refiner_angle/"
[ "$RUN_SEG" -eq 1 ] && \
echo "   Base+CircNet+SEG: /app/output/real_refiner_angle_graph/"
echo "═══════════════════════════════════════════════════════"
