#!/bin/bash
# ============================================================
# rerun_metrics_body.sh — przelicza TYLKO metryki dla już
# wytrenowanych modeli z podanego BATCH_DIR.
#
# Użycie:
#   bash rerun_metrics_body.sh <BATCH_DIR>
#
# Przykład:
#   bash rerun_metrics_body.sh \
#     /net/storage/pr3/plgrid/plggsmartudlteam/CircNet/runs/body_64f_20260521_2047
# ============================================================

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Użycie: $0 <BATCH_DIR>"
  exit 1
fi

BATCH_DIR="$1"
TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam
CIRCNET_MODEL="${CIRCNET_MODEL:-circnet_body_64f_16838558}"
REFINER_BASE="${REFINER_BASE:-128}"

echo "========================================================"
echo " rerun_metrics_body.sh"
echo " BATCH_DIR=${BATCH_DIR}"
echo " CIRCNET_MODEL=${CIRCNET_MODEL}  REFINER_BASE=${REFINER_BASE}"
echo "========================================================"

# Globalne tablice do zbierania ID zgłoszonych jobów i ich katalogów
SUBMITTED_JOB_IDS=()
SUBMITTED_RUN_DIRS=()

# ============================================================
# Funkcja pomocnicza: sbatch metryki dla jednego wariantu
# ============================================================
submit_metrics() {
  local name="$1"          # np. base, nmar, angle, nmar-angle
  local job_id="$2"        # SLURM job ID z treningu (do lokalizacji folderu)
  local extra_args="$3"    # dodatkowe argumenty dla metrics.py

  local RUN_DIR="${BATCH_DIR}/refiner_body_${name}_${REFINER_BASE}f_${job_id}"
  local CONTAINER_RUN_DIR="/app/CircNet${RUN_DIR#${TEAM_DIR}/CircNet}"
  local LOG="${RUN_DIR}/slurm_logs/02_metrics_rerun.log"

  if [[ ! -f "${RUN_DIR}/best_refiner.pth" ]]; then
    echo "[SKIP] Brak modelu: ${RUN_DIR}/best_refiner.pth"
    return
  fi

  local METRICS_JOB_ID
  METRICS_JOB_ID=$(sbatch --parsable \
    -N 1 -p plgrid-gpu-gh200 -A plgsmartudl-gpu-gh200 \
    --ntasks-per-node=1 --cpus-per-task=8 --mem=40GB --time=00:30:00 \
    --gres=gpu:1 --gpu-bind=closest \
    --job-name="metrics_body_${name}" \
    --output="${LOG}" --error="${LOG}" \
    --wrap="
      export PYTHONPATH=\"\$PYTHONPATH:/\"
      export APPTAINER_CACHEDIR=${TEAM_DIR}/apptainer_cache
      apptainer exec --nv \\
        --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet,${TEAM_DIR}/output:/app/output \\
        ${TEAM_DIR}/sif_images/sif_image.sif \\
        python3 /app/CircNet/circnet_refiner_metrics.py \\
          --model-path    ${CONTAINER_RUN_DIR}/best_refiner.pth \\
          --data-root     /app/data/RPI/body \\
          --test-folder   body5 \\
          --output-dir    ${CONTAINER_RUN_DIR}/metrics \\
          --base-features ${REFINER_BASE} \\
          ${extra_args}
    ")
  echo "[SUBMIT] metrics_body_${name}  job_id=${METRICS_JOB_ID}  (model: ${RUN_DIR##*/})"

  # Zapisz ID i katalog do globalnych tablic
  SUBMITTED_JOB_IDS+=("${METRICS_JOB_ID}")
  SUBMITTED_RUN_DIRS+=("${CONTAINER_RUN_DIR}")
}

# ============================================================
# Wykryj job IDs z nazw folderów w BATCH_DIR
# ============================================================
find_job_id() {
  local name="$1"
  local pattern="refiner_body_${name}_${REFINER_BASE}f_"
  local dir
  dir=$(find "${BATCH_DIR}" -maxdepth 1 -type d -name "${pattern}*" | head -1)
  if [[ -z "$dir" ]]; then
    echo ""
  else
    echo "${dir##*_}"
  fi
}

# ============================================================
# Submittuj metryki dla każdego wariantu
# ============================================================
for variant in base nmar angle nmar-angle; do
  job_id=$(find_job_id "${variant}")
  if [[ -z "$job_id" ]]; then
    echo "[SKIP] Nie znaleziono folderu dla wariantu: ${variant}"
    continue
  fi

  case "${variant}" in
    base)
      submit_metrics "base" "${job_id}" "--no-prior"
      ;;
    nmar)
      submit_metrics "nmar" "${job_id}" \
        "--prior-dir /app/output/aapm_inference_results --prior-suffix _nmar2"
      ;;
    angle)
      submit_metrics "angle" "${job_id}" \
        "--prior-dir /app/CircNet/${CIRCNET_MODEL}/corrected"
      ;;
    nmar-angle)
      submit_metrics "nmar-angle" "${job_id}" \
        "--prior-dir /app/output/aapm_inference_results --prior-suffix _nmar2 --angle-dir /app/CircNet/${CIRCNET_MODEL}/corrected"
      ;;
  esac
done

# ============================================================
# collect_metrics — po zakończeniu wszystkich metryk
# Używa SUBMITTED_JOB_IDS i SUBMITTED_RUN_DIRS zebranych przez submit_metrics
# ============================================================
if [[ ${#SUBMITTED_JOB_IDS[@]} -gt 0 ]]; then
  DEP=$(IFS=:; echo "afterany:${SUBMITTED_JOB_IDS[*]}")
  COLLECT_OUT="${BATCH_DIR}/body_metrics_summary.csv"
  CONTAINER_COLLECT_OUT="/app/CircNet${COLLECT_OUT#${TEAM_DIR}/CircNet}"

  JOB_COLLECT=$(sbatch --parsable \
    --dependency="${DEP}" \
    --job-name=body_collect_rerun \
    -N 1 -p plgrid-gpu-gh200 -A plgsmartudl-gpu-gh200 \
    --ntasks-per-node=1 --cpus-per-task=4 --mem=8GB --time=00:15:00 \
    --output="${BATCH_DIR}/collect_metrics_rerun.log" \
    --error="${BATCH_DIR}/collect_metrics_rerun.log" \
    --wrap="
      export PYTHONPATH=\"\$PYTHONPATH:/\"
      apptainer exec \\
        --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \\
        ${TEAM_DIR}/sif_images/sif_image.sif \\
        python3 /app/CircNet/collect_metrics.py \\
          --run-dirs ${SUBMITTED_RUN_DIRS[*]} \\
          --output ${CONTAINER_COLLECT_OUT}
    ")
  echo "[SUBMIT] collect_metrics  job_id=${JOB_COLLECT}  (po zakończeniu metryk)"
  echo ""
  echo "Sprawdź status: squeue -u \$USER"
  echo "Logi metryk:    ls ${BATCH_DIR}/refiner_body_*/slurm_logs/02_metrics_rerun.log"
  echo "Tabela CSV:     ${COLLECT_OUT}"
else
  echo ""
  echo "[INFO] Nie zgłoszono żadnych jobów metryk — collect_metrics pominięty."
  echo "Sprawdź status: squeue -u \$USER"
fi
