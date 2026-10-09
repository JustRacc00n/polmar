#!/bin/bash -l
# ============================================================
# run_circnet_sweep.sh — trening CircNet dla wielu wartości radius_features
#
# Dla każdej wartości w SWEEP_FEATURES uruchamia łańcuch:
#   1. circnet_body.sh       — trening (24h)
#   2. circnet_body_eval.sh  — inferencja body5 + metryki + sanity check (2h)
#                               zależny od zakończenia treningu (afterok)
#
# Wszystkie runy startują równolegle (różne wartości niezależne).
# Na końcu jeden job zbiorczy collect_metrics (afterany: wszystkie eval).
#
# Użycie:
#   bash run_circnet_sweep.sh
#   bash run_circnet_sweep.sh --with-infer-full   # + inferencja na body1-5 po eval
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_INFER_FULL=0

for arg in "$@"; do
  [[ "$arg" == "--with-infer-full" ]] && WITH_INFER_FULL=1
done

# ============================================================
# KONFIGURACJA — zmień tutaj
# ============================================================
export TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam
export TRAIN_EPOCHS=100
export TRAIN_BATCH=2

# Lista wartości radius_features do przetestowania
SWEEP_FEATURES="32 64 128 256 512"
# ============================================================

echo "========================================================"
echo " run_circnet_sweep.sh"
echo " Data: $(date)"
echo " SWEEP_FEATURES=${SWEEP_FEATURES}"
echo " TRAIN_EPOCHS=${TRAIN_EPOCHS}  TRAIN_BATCH=${TRAIN_BATCH}"
echo " With full body inference: ${WITH_INFER_FULL}"
echo "========================================================"

ALL_EVAL_JOBS=""   # zbieramy ID jobów eval do collect_metrics
ALL_RUN_DIRS=""    # zbieramy ścieżki do run_dirs

for RF in ${SWEEP_FEATURES}; do
  echo ""
  echo "---- radius_features=${RF} ----------------------------------------"

  export RADIUS_FEATURES=${RF}

  # --- Krok 1: Trening ---
  echo "[SUBMIT] Trening CircNet (rf=${RF})..."
  JOB_TRAIN=$(sbatch --parsable --export=ALL \
    ${SCRIPT_DIR}/train/circnet_body.sh)
  echo "  -> job id: ${JOB_TRAIN}"

  # Zbuduj nazwę runu (musi zgadzać się z circnet_body.sh)
  CIRCNET_MODEL="circnet_body_${RF}f_${JOB_TRAIN}"
  export CIRCNET_MODEL

  # --- Krok 2: Ewaluacja (inferencja body5 + metryki + sanity check) ---
  echo "[SUBMIT] Ewaluacja (rf=${RF}, model=${CIRCNET_MODEL})..."
  JOB_EVAL=$(sbatch --parsable --export=ALL \
    --dependency=afterok:${JOB_TRAIN} \
    ${SCRIPT_DIR}/train/circnet_body_eval.sh)
  echo "  -> job id: ${JOB_EVAL}"

  # --- Opcjonalnie: pełna inferencja body1-5 (do dalszego użycia z refinerem) ---
  if [[ $WITH_INFER_FULL -eq 1 ]]; then
    echo "[SUBMIT] Pełna inferencja body1-5 (rf=${RF})..."
    JOB_INFER=$(sbatch --parsable --export=ALL \
      --dependency=afterok:${JOB_TRAIN} \
      ${SCRIPT_DIR}/prep/00_circnet_infer_body.sh)
    echo "  -> job id: ${JOB_INFER}"
  fi

  # Zbierz do listy dla collect_metrics
  RUN_DIR="${TEAM_DIR}/CircNet/${CIRCNET_MODEL}"
  ALL_EVAL_JOBS="${ALL_EVAL_JOBS:+${ALL_EVAL_JOBS}:}${JOB_EVAL}"
  ALL_RUN_DIRS="${ALL_RUN_DIRS} ${RUN_DIR}"

  echo "  Train: ${JOB_TRAIN}  Eval: ${JOB_EVAL}  Model: ${CIRCNET_MODEL}"

done

# ============================================================
# Krok końcowy: tabela zbiorcza metryk po zakończeniu wszystkich eval
# ============================================================
COLLECT_OUT="${TEAM_DIR}/CircNet/sweep_metrics_$(date +%Y%m%d_%H%M).csv"

echo ""
echo "[SUBMIT] collect_metrics (po zakończeniu wszystkich eval jobów)..."
JOB_COLLECT=$(sbatch --parsable \
  --dependency=afterok:${ALL_EVAL_JOBS} \
  --job-name=sweep_collect \
  -N 1 -p plgrid-gpu-gh200 -A plgsmartudl-gpu-gh200 \
  --ntasks-per-node=1 --cpus-per-task=4 --mem=8GB --time=00:15:00 \
  --output=/dev/null --error=/dev/null \
  --wrap="
    export PYTHONPATH=\"\$PYTHONPATH:/\"
    exec > >(tee ${TEAM_DIR}/CircNet/sweep_collect_$(date +%Y%m%d_%H%M).log) 2>&1
    echo \"[INFO] collect_metrics started: \$(date)\"
    apptainer exec \
      --bind ${TEAM_DIR}/CircNet:/app/CircNet \
      ${TEAM_DIR}/sif_images/sif_image.sif \
      python3 /app/CircNet/collect_metrics.py \
        --run-dirs ${ALL_RUN_DIRS} \
        --output   ${COLLECT_OUT}
    echo \"[INFO] collect_metrics finished: \$(date)\"
    echo \"[INFO] Tabela: ${COLLECT_OUT}\"
  ")
echo "  -> job id: ${JOB_COLLECT}"

# ============================================================
# Podsumowanie zgłoszonych jobów
# ============================================================
echo ""
echo "========================================================"
echo " Zgłoszone joby:"
echo ""

for RF in ${SWEEP_FEATURES}; do
  echo "  radius_features=${RF}"
done

echo ""
echo " collect_metrics : ${JOB_COLLECT}  (afterok: wszystkie eval)"
echo " Tabela wynikowa : ${COLLECT_OUT}"
echo ""
echo " Sprawdź status: squeue -u \$USER"
echo "========================================================"
echo ""
echo " Następny krok po zakończeniu:"
echo "   Ustaw CIRCNET_MODEL w run_all_body.sh / run_all_jitter.sh"
echo "   i uruchom je."
echo "========================================================"
