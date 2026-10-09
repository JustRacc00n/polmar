#!/bin/bash -l
# ============================================================
# run_sweep_eval_redo.sh — ponowna ewaluacja dla wszystkich
# wariantów sweepów CircNet (inferencja + metryki + sanity check)
#
# Użyj gdy eval joby z run_circnet_sweep.sh nie uruchomiły się
# z powodu dependency "never satisfied" (afterok po TIMEOUT treningu).
#
# Zakłada że best_model.pth istnieje w każdym run_dir.
# Uruchamia 5 niezależnych jobów eval (po jednym na wariant).
#
# Użycie:
#   bash run_sweep_eval_redo.sh
# ============================================================

TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam

# ============================================================
# KONFIGURACJA — uzupełnij ID jobów z sweepowego uruchomienia
# ============================================================
declare -A MODELS=(
    [32]="circnet_body_32f_18562699"
    [64]="circnet_body_64f_18562701"
    [128]="circnet_body_128f_18562703"
    [256]="circnet_body_256f_18562705"
    [512]="circnet_body_512f_18562707"
)
TEST_FOLDER="body5"
N_SAMPLES=16
COLS=4
WINDOW="soft_tissue"
# ============================================================

echo "========================================================"
echo " run_sweep_eval_redo.sh"
echo " Data: $(date)"
echo " Uruchamia: inferencja + metryki + sanity check"
echo "========================================================"

ALL_JOBS=""

for RF in 32 64 128 256 512; do
    MODEL="${MODELS[$RF]}"
    RUN_DIR="${TEAM_DIR}/CircNet/${MODEL}"
    LOG_DIR="${RUN_DIR}/slurm_logs"

    echo ""
    echo "---- rf=${RF}  model=${MODEL} ----"

    # Sprawdź czy model w ogóle istnieje
    if [ ! -f "${RUN_DIR}/best_model.pth" ]; then
        echo "[WARN] Brak best_model.pth — trening nie skończony? Pomijam."
        continue
    fi

    JOB_ID=$(sbatch --parsable \
        --job-name="eval_${RF}f" \
        -N 1 -p plgrid-gpu-gh200 -A plgsmartudl-gpu-gh200 \
        --ntasks-per-node=1 --cpus-per-task=8 --mem=32GB --time=03:00:00 \
        --gres=gpu:1 --gpu-bind=closest \
        --output=/dev/null --error=/dev/null \
        --wrap="
export APPTAINER_CACHEDIR=${TEAM_DIR}/apptainer_cache
export PYTHONPATH=\"\$PYTHONPATH:/\"
mkdir -p ${LOG_DIR}
exec > >(tee ${LOG_DIR}/eval_redo_${RF}f.out) 2>&1
set -e

echo '[INFO] eval_redo started: \$(date)'
echo '[INFO] Model: ${MODEL}  rf=${RF}'
echo '[INFO] SLURM_JOB_ID: '\$SLURM_JOB_ID

# ── Krok 1: Inferencja → corrected/ ──────────────────────────
echo ''
echo '[STEP 1/3] Inferencja na ${TEST_FOLDER}...'
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/inference.py \
    --model-path   /app/CircNet/${MODEL}/best_model.pth \
    --data-root    /app/data/RPI/body \
    --test-folders ${TEST_FOLDER} \
    --output-dir   /app/CircNet/${MODEL}/corrected \
  2>&1 | tee ${LOG_DIR}/eval_redo_01_infer_${RF}f.log
echo '[INFO] Krok 1 OK'

# ── Krok 2: Metryki → metrics/ ───────────────────────────────
echo ''
echo '[STEP 2/3] Metryki na ${TEST_FOLDER}...'
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/compute_metrics.py \
    --model-path   /app/CircNet/${MODEL}/best_model.pth \
    --data-root    /app/data/RPI/body \
    --test-folder  ${TEST_FOLDER} \
    --output-dir   /app/CircNet/${MODEL}/metrics \
  2>&1 | tee ${LOG_DIR}/eval_redo_02_metrics_${RF}f.log
echo '[INFO] Krok 2 OK'

# ── Krok 3: Sanity check → sanitycheck/ ──────────────────────
echo ''
echo '[STEP 3/3] Sanity check...'
BODY5_BASELINE=\$(apptainer exec \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 -c \"
from pathlib import Path
import sys
sys.path.insert(0, '/app/CircNet')
from dataset import find_subfolders
p = find_subfolders(Path('/app/data/RPI/body/${TEST_FOLDER}'))
print(p / 'Baseline')
\")
echo '[INFO] Baseline dir: '\${BODY5_BASELINE}

apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/sanity_check.py \
    --corrected-dir /app/CircNet/${MODEL}/corrected \
    --baseline-dir  \${BODY5_BASELINE} \
    --output-dir    /app/CircNet/${MODEL}/sanitycheck \
    --n-samples     ${N_SAMPLES} \
    --cols          ${COLS} \
    --window        ${WINDOW} \
  2>&1 | tee ${LOG_DIR}/eval_redo_03_sanity_${RF}f.log
echo '[INFO] Krok 3 OK'

echo ''
echo '[INFO] eval_redo finished: \$(date)'
echo '[INFO] Wyniki w: ${RUN_DIR}/'
")

    echo "  -> job id: ${JOB_ID}"
    ALL_JOBS="${ALL_JOBS:+${ALL_JOBS}:}${JOB_ID}"
done

echo ""
echo "========================================================"
echo " Zgłoszone joby: ${ALL_JOBS}"
echo " Sprawdź status: squeue -u \$USER"
echo ""
echo " Wyniki (po zakończeniu):"
for RF in 32 64 128 256 512; do
    MODEL="${MODELS[$RF]}"
    echo "   rf=${RF}: \${TEAM_DIR}/CircNet/${MODEL}/{corrected,metrics,sanitycheck}/"
done
echo "========================================================"
