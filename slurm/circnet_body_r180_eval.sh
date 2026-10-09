#!/bin/bash -l
# ============================================================
# circnet_body_r180_eval.sh — Ewaluacja modelu r180 na body5
#
# Uruchamiaj po zakończeniu treningu circnet_body_r180.sh.
# Identyczny pipeline jak circnet_body_eval.sh, ale domyślnie
# wskazuje na model z radialnym cutoffem.
#
# Użycie:
#   CIRCNET_MODEL=circnet_body_r180_128f_<JOBID> sbatch --export=ALL circnet_body_r180_eval.sh
#   # lub jeśli trening jeszcze trwa:
#   sbatch --dependency=afterok:<JOBID_TRAIN> --export=ALL circnet_body_r180_eval.sh
# ============================================================

#SBATCH -J circnet_r180_eval
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
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
CIRCNET_MODEL=${CIRCNET_MODEL:-"circnet_body_r180_128f_UNKNOWN"}
TEST_FOLDER="body5"
# =========================================================

RUN_DIR=${TEAM_DIR}/CircNet/${CIRCNET_MODEL}
LOG_DIR=${RUN_DIR}/slurm_logs
CORRECTED_DIR=${RUN_DIR}/corrected
METRICS_DIR=${RUN_DIR}/metrics
SANITY_DIR=${RUN_DIR}/sanitycheck

mkdir -p ${LOG_DIR}

exec > >(tee ${LOG_DIR}/eval_${SLURM_JOB_ID}.out) 2>&1

echo "[INFO] Job started: $(date)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] Model: ${CIRCNET_MODEL}"
echo "[INFO] Test folder: ${TEST_FOLDER}"
echo "[INFO] Run dir: ${RUN_DIR}"

# =========================================================
# Krok 1: Inferencja na body5
# =========================================================
echo ""
echo "[STEP 1/3] Inferencja na ${TEST_FOLDER}..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/inference.py \
    --model-path      /app/CircNet/${CIRCNET_MODEL}/best_model.pth \
    --data-root       /app/data/RPI/body \
    --test-folders    ${TEST_FOLDER} \
    --output-dir      /app/CircNet/${CIRCNET_MODEL}/corrected \
  2>&1 | tee ${LOG_DIR}/eval_01_infer.log

if [ $? -ne 0 ]; then
    echo "[ERROR] Inferencja nie powiodła się. Przerywam."
    exit 1
fi

# =========================================================
# Krok 2: Metryki na body5
# =========================================================
echo ""
echo "[STEP 2/3] Metryki na ${TEST_FOLDER}..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/compute_metrics.py \
    --model-path      /app/CircNet/${CIRCNET_MODEL}/best_model.pth \
    --data-root       /app/data/RPI/body \
    --test-folder     ${TEST_FOLDER} \
    --output-dir      /app/CircNet/${CIRCNET_MODEL}/metrics \
  2>&1 | tee ${LOG_DIR}/eval_02_metrics.log

if [ $? -ne 0 ]; then
    echo "[ERROR] Metryki nie powiodły się. Kontynuuję do sanity check..."
fi

# =========================================================
# Krok 3: Sanity check — siatka PNG
# =========================================================
echo ""
echo "[STEP 3/3] Sanity check..."

BODY5_BASELINE=$(apptainer exec \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 -c "
from pathlib import Path
from dataset import find_subfolders
p = find_subfolders(Path('/app/data/RPI/body/${TEST_FOLDER}'))
print(p / 'Baseline')
")

echo "[INFO] body5 Baseline dir: ${BODY5_BASELINE}"

apptainer exec \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/sanity_check.py \
    --corrected-dir   /app/CircNet/${CIRCNET_MODEL}/corrected \
    --baseline-dir    ${BODY5_BASELINE} \
    --output-dir      /app/CircNet/${CIRCNET_MODEL}/sanitycheck \
    --n-samples       16 \
    --cols            4 \
    --window          soft_tissue \
  2>&1 | tee ${LOG_DIR}/eval_03_sanity.log

echo ""
echo "[INFO] Job finished: $(date)"
echo "[INFO] Wyniki:"
echo "  Corrected : ${CORRECTED_DIR}/"
echo "  Metryki   : ${METRICS_DIR}/metrics_${TEST_FOLDER}.csv"
echo "  Sanity    : ${SANITY_DIR}/"
echo ""
echo "  Aby wpiąć do collect_metrics:"
echo "  python3 collect_metrics.py --run-dirs /app/CircNet/${CIRCNET_MODEL} ..."
