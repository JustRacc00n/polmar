#!/bin/bash -l
# ============================================================
# Masterscrypt: pełny potok treningu CircNet
#
# Kolejność:
#   1. Trening CircNet (body)
#   2. Inferencja CircNet na body1-5
#   3. (opcjonalnie) Inferencja CircNet na jitter
#
# Po zakończeniu możesz odpalić run_all_body.sh / run_all_jitter.sh
# (bez flagi --with-infer, bo inferencja jest już gotowa).
#
# Użycie:
#   bash run_all_train.sh                   # trening + inferencja body
#   bash run_all_train.sh --with-jitter     # + inferencja jitter
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_JITTER=0

for arg in "$@"; do
  [[ "$arg" == "--with-jitter" ]] && WITH_JITTER=1
done

# ============================================================
# KONFIGURACJA — zmień tutaj
# ============================================================
export TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam
export RADIUS_FEATURES=64     # decyduje o nazwie modelu (32→32f, 64→64f)
export TRAIN_EPOCHS=100
export TRAIN_BATCH=2
# ============================================================

echo "========================================================"
echo " run_all_train.sh"
echo " Data: $(date)"
echo " RADIUS_FEATURES=${RADIUS_FEATURES}  TRAIN_EPOCHS=${TRAIN_EPOCHS}"
echo " With jitter inference: ${WITH_JITTER}"
echo "========================================================"

# ---- krok 1: trening CircNet (body) ------------------------
echo "[SUBMIT] CircNet training (body)..."
JOB_TRAIN=$(sbatch --parsable --export=ALL ${SCRIPT_DIR}/train/circnet_body.sh)
echo "  -> job id: ${JOB_TRAIN}"

# ---- krok 2: inferencja body (zależy od treningu) ----------
echo "[SUBMIT] CircNet inference body (po treningu)..."
JOB_INFER_BODY=$(sbatch --parsable --export=ALL \
  --dependency=afterok:${JOB_TRAIN} \
  ${SCRIPT_DIR}/prep/00_circnet_infer_body.sh)
echo "  -> job id: ${JOB_INFER_BODY}"

# ---- krok 3 (opcjonalny): inferencja jitter ----------------
JOB_INFER_JITTER=""
if [[ $WITH_JITTER -eq 1 ]]; then
  echo "[SUBMIT] CircNet inference jitter (po treningu)..."
  JOB_INFER_JITTER=$(sbatch --parsable --export=ALL \
    --dependency=afterok:${JOB_TRAIN} \
    ${SCRIPT_DIR}/prep/00_circnet_infer_jitter.sh)
  echo "  -> job id: ${JOB_INFER_JITTER}"
fi

echo "========================================================"
echo " Zgłoszone joby:"
echo "  circnet_train      : ${JOB_TRAIN}"
echo "  infer_body         : ${JOB_INFER_BODY}  (afterok: ${JOB_TRAIN})"
[[ -n $JOB_INFER_JITTER ]] && \
echo "  infer_jitter       : ${JOB_INFER_JITTER}  (afterok: ${JOB_TRAIN})"
echo ""
echo " Następny krok po zakończeniu:"
echo "   bash run_all_body.sh              # bez --with-infer"
[[ $WITH_JITTER -eq 1 ]] && \
echo "   bash run_all_jitter.sh            # bez --with-infer"
echo ""
echo " Sprawdź status: squeue -u \$USER"
echo "========================================================"
