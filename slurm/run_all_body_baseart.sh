#!/bin/bash -l
# ============================================================
# Masterscrypt: body baseart — warianty refinera z --base-art
# (baza residualna = zawsze art, prior tylko jako info)
#
# Kolejność:
#   0. Pre-flight check — sprawdza czy potrzebne priory istnieją:
#        CircNet body: auto-submit jeśli brakuje
#        NMAR body:    abort jeśli brakuje
#   1. Refiner: base, nmar_baseart, angle_baseart, nmar-angle_baseart — równolegle
#      (body_base nie zmienia się — no_prior => art jest bazą domyślnie)
#   2. collect_metrics — zbiorcza tabela CSV po zakończeniu runów
#
# Użycie:
#   bash run_all_body_baseart.sh                # auto-wykrywanie priorów
#   bash run_all_body_baseart.sh --force-infer  # wymuś ponowną inferencję CircNet
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCE_INFER=0

for arg in "$@"; do
  [[ "$arg" == "--force-infer" ]] && FORCE_INFER=1
done

# ============================================================
# KONFIGURACJA — zmień tutaj, nie w podskryptach
# ============================================================
export TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam
export CIRCNET_MODEL="circnet_body_128f_17658441"
export RADIUS_FEATURES=128
export REFINER_BASE=128
export REFINER_EPOCHS=200
export REFINER_BATCH=4
# ============================================================

echo "========================================================"
echo " run_all_body_baseart.sh"
echo " Data: $(date)"
echo " REFINER_BASE=${REFINER_BASE}  REFINER_EPOCHS=${REFINER_EPOCHS}"
echo " CIRCNET_MODEL=${CIRCNET_MODEL}"
echo " Force infer: ${FORCE_INFER}"
echo " Tryb: --base-art (art zawsze bazą residualną)"
echo "========================================================"

# ============================================================
# Funkcja pomocnicza: sprawdza czy katalog zawiera pliki .raw
# ============================================================
has_raw_files() {
  local dir="$1"
  [[ -d "$dir" ]] && compgen -G "${dir}/*.raw" > /dev/null 2>&1
}

# ============================================================
# PRE-FLIGHT CHECK
# ============================================================
echo ""
echo "--- Pre-flight check ---"
MISSING=0
INFER_DEPENDENCY=""
INFER_JOB=""

# ---- 1. CircNet body inference (body_angle_baseart, body_nmar-angle_baseart) ----
CIRCNET_CORRECTED="${TEAM_DIR}/CircNet/${CIRCNET_MODEL}/corrected"
if [[ $FORCE_INFER -eq 1 ]]; then
  echo "[FORCE]   CircNet body inference — wymuszony ponowny run..."
  INFER_JOB=$(sbatch --parsable --export=ALL ${SCRIPT_DIR}/prep/00_circnet_infer_body.sh)
  echo "  -> job id: ${INFER_JOB}"
  INFER_DEPENDENCY="--dependency=afterok:${INFER_JOB}"
elif has_raw_files "${CIRCNET_CORRECTED}"; then
  echo "[OK]      CircNet body inference: ${CIRCNET_CORRECTED}"
else
  echo "[MISSING] CircNet body inference — submituje 00_circnet_infer_body.sh automatycznie..."
  INFER_JOB=$(sbatch --parsable --export=ALL ${SCRIPT_DIR}/prep/00_circnet_infer_body.sh)
  echo "  -> job id: ${INFER_JOB}"
  INFER_DEPENDENCY="--dependency=afterok:${INFER_JOB}"
fi

# ---- 2. NMAR body data (body_nmar_baseart, body_nmar-angle_baseart) ----
NMAR_DIR="${TEAM_DIR}/output/aapm_inference_results"
if has_raw_files "${NMAR_DIR}/body1"; then
  echo "[OK]      NMAR body: ${NMAR_DIR}/body1/"
else
  echo "[ERROR]   NMAR body NIE ISTNIEJE: ${NMAR_DIR}/body1/*.raw"
  echo "          Uruchom nmar_inference.py na body1-5 i poczekaj na wyniki."
  MISSING=1
fi

if [[ $MISSING -eq 1 ]]; then
  echo ""
  echo "[ABORT] Brakujące priory uniemożliwiają uruchomienie. Przerwano."
  exit 1
fi

echo ""
echo "Pre-flight OK — zgłaszam joby refinera (base-art)."
echo ""

# ============================================================
# Folder zbiorczy dla tego runu
# ============================================================
export BATCH_DIR=${TEAM_DIR}/CircNet/runs/body_baseart_circ${RADIUS_FEATURES}f_ref${REFINER_BASE}f_$(date +%Y%m%d_%H%M)
mkdir -p ${BATCH_DIR}
echo "Folder zbiorczy: ${BATCH_DIR}"
echo ""

# ============================================================
# Refiner — wszystkie warianty równolegle
# body_base: no_prior => art jest bazą domyślnie (bez --base-art)
# pozostałe: --base-art wymuszony przez dedykowane skrypty
# ============================================================

echo "[SUBMIT] body_base (bez zmian — no_prior => art jest bazą)..."
JOB_BASE=$(sbatch --parsable --export=ALL \
  ${SCRIPT_DIR}/body/body_base.sh)
echo "  -> job id: ${JOB_BASE}"

echo "[SUBMIT] body_nmar_baseart..."
JOB_NMAR=$(sbatch --parsable --export=ALL \
  ${SCRIPT_DIR}/body/body_nmar_baseart.sh)
echo "  -> job id: ${JOB_NMAR}"

echo "[SUBMIT] body_angle_baseart..."
JOB_ANGLE=$(sbatch --parsable --export=ALL \
  ${INFER_DEPENDENCY} \
  ${SCRIPT_DIR}/body/body_angle_baseart.sh)
echo "  -> job id: ${JOB_ANGLE}"

echo "[SUBMIT] body_nmar-angle_baseart..."
JOB_NMAR_ANGLE=$(sbatch --parsable --export=ALL \
  ${INFER_DEPENDENCY} \
  ${SCRIPT_DIR}/body/body_nmar-angle_baseart.sh)
echo "  -> job id: ${JOB_NMAR_ANGLE}"

# ============================================================
# collect_metrics po zakończeniu wszystkich runów
# ============================================================
ALL_JOBS="${JOB_BASE}:${JOB_NMAR}:${JOB_ANGLE}:${JOB_NMAR_ANGLE}"

RUN_BASE=${BATCH_DIR}/refiner_body_base_${REFINER_BASE}f_${JOB_BASE}
RUN_NMAR=${BATCH_DIR}/refiner_body_nmar_baseart_${REFINER_BASE}f_${JOB_NMAR}
RUN_ANGLE=${BATCH_DIR}/refiner_body_angle_baseart_${REFINER_BASE}f_${JOB_ANGLE}
RUN_NMAR_ANGLE=${BATCH_DIR}/refiner_body_nmar-angle_baseart_${REFINER_BASE}f_${JOB_NMAR_ANGLE}

COLLECT_OUT=${BATCH_DIR}/body_baseart_metrics_summary.csv

echo "[SUBMIT] collect_metrics (po zakończeniu wszystkich runów)..."
JOB_COLLECT=$(sbatch --parsable \
  --dependency=afterany:${ALL_JOBS} \
  --job-name=body_baseart_collect \
  -N 1 -p plgrid-gpu-gh200 -A plgsmartudl-gpu-gh200 \
  --ntasks-per-node=1 --cpus-per-task=4 --mem=8GB --time=00:15:00 \
  --output=/dev/null --error=/dev/null \
  --wrap="
    export PYTHONPATH=\"\$PYTHONPATH:/\"
    exec > >(tee ${BATCH_DIR}/collect_metrics.log) 2>&1
    echo \"[INFO] collect_metrics started: \$(date)\"
    apptainer exec \
      --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
      ${TEAM_DIR}/sif_images/sif_image.sif \
      python3 /app/CircNet/collect_metrics.py \
        --run-dirs \
          /app/CircNet${RUN_BASE#${TEAM_DIR}/CircNet} \
          /app/CircNet${RUN_NMAR#${TEAM_DIR}/CircNet} \
          /app/CircNet${RUN_ANGLE#${TEAM_DIR}/CircNet} \
          /app/CircNet${RUN_NMAR_ANGLE#${TEAM_DIR}/CircNet} \
        --output /app/CircNet${COLLECT_OUT#${TEAM_DIR}/CircNet}
    echo \"[INFO] collect_metrics finished: \$(date)\"
    echo \"[INFO] Tabela: ${COLLECT_OUT}\"
  ")
echo "  -> job id: ${JOB_COLLECT}"

echo "========================================================"
echo " Zgłoszone joby (base-art):"
[[ -n $INFER_JOB ]] && echo "  CircNet infer          : ${INFER_JOB}"
echo "  body_base              : ${JOB_BASE}"
echo "  body_nmar_baseart      : ${JOB_NMAR}"
echo "  body_angle_baseart     : ${JOB_ANGLE}  ${INFER_DEPENDENCY:+(afterok: ${INFER_JOB})}"
echo "  body_nmar-angle_baseart: ${JOB_NMAR_ANGLE}  ${INFER_DEPENDENCY:+(afterok: ${INFER_JOB})}"
echo "  collect_metrics        : ${JOB_COLLECT}  (afterany powyższych)"
echo ""
echo " Tabela wynikowa: ${COLLECT_OUT}"
echo " Sprawdź status:  squeue -u \$USER"
echo "========================================================"
