#!/bin/bash -l
# ============================================================
# Masterscrypt: jitter — uruchamia wszystkie warianty refinera
#
# Kolejność:
#   0. Pre-flight check — sprawdza czy potrzebne priory istnieją:
#        CircNet jitter: auto-submit jeśli brakuje
#        NMAR jitter:    auto-submit jeśli brakuje
#        Graf jitter:    auto-submit jeśli brakuje (wymaga NIfTI+Seg)
#   1. Refiner: base, nmar, angle, graph, nmar-angle,
#               nmar-graph, angle-graph — równolegle po priorach
#   2. collect_metrics — zbiorcza tabela CSV po zakończeniu runów
#
# Użycie:
#   bash run_all_jitter.sh                # auto-wykrywanie priorów
#   bash run_all_jitter.sh --force-infer  # wymuś ponowne przeliczenie wszystkich priorów
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
echo " run_all_jitter.sh"
echo " Data: $(date)"
echo " REFINER_BASE=${REFINER_BASE}  REFINER_EPOCHS=${REFINER_EPOCHS}"
echo " CIRCNET_MODEL=${CIRCNET_MODEL}"
echo " Force infer: ${FORCE_INFER}"
echo "========================================================"

# ============================================================
# Funkcje pomocnicze sprawdzające priory
# ============================================================
has_raw_files() {
  # Sprawdza czy katalog zawiera JAKIEKOLWIEK pliki .raw (dla CircNet/Graf)
  local dir="$1"
  [[ -d "$dir" ]] && compgen -G "${dir}/*.raw" > /dev/null 2>&1
}

has_jitter_nmar_files() {
  # Sprawdza czy nmar_results zawiera SPECYFICZNIE pliki jitter_*_nmar2.raw
  # Samo istnienie innych .raw (np. z body) nie wystarczy
  local dir="$1"
  [[ -d "$dir" ]] && compgen -G "${dir}/jitter_Clean*_nmar2.raw" > /dev/null 2>&1
}

# ============================================================
# PRE-FLIGHT CHECK
# ============================================================
echo ""
echo "--- Pre-flight check ---"
MISSING=0

# Dependency IDs — będą składane w łańcuch afterok dla wariantów które ich wymagają
JOB_INFER_CIRCNET=""
JOB_INFER_NMAR=""
JOB_INFER_GRAPH=""

# ---- 1. CircNet jitter inference (jitter_angle, nmar-angle, angle-graph) ----
CIRCNET_JITTER_DIR="${TEAM_DIR}/CircNet/${CIRCNET_MODEL}/corrected_jitter"
if [[ $FORCE_INFER -eq 1 ]] || ! has_raw_files "${CIRCNET_JITTER_DIR}"; then
  [[ $FORCE_INFER -eq 1 ]] \
    && echo "[FORCE]   CircNet jitter inference..." \
    || echo "[MISSING] CircNet jitter inference — submituje automatycznie..."
  JOB_INFER_CIRCNET=$(sbatch --parsable --export=ALL \
    ${SCRIPT_DIR}/prep/00_circnet_infer_jitter.sh)
  echo "  -> job id: ${JOB_INFER_CIRCNET}"
else
  echo "[OK]      CircNet jitter: ${CIRCNET_JITTER_DIR}"
fi

# ---- 2. NMAR jitter (jitter_nmar, nmar-angle, nmar-graph) ----
# UWAGA: sprawdzamy *specyficznie* pliki jitter_Clean*_nmar2.raw, nie jakiekolwiek .raw
# (katalog nmar_results mógł być wypełniony plikami z innych datasetu — body, testy itp.)
NMAR_DIR="${TEAM_DIR}/output/nmar_results_2"
if [[ $FORCE_INFER -eq 1 ]] || ! has_jitter_nmar_files "${NMAR_DIR}"; then
  [[ $FORCE_INFER -eq 1 ]] \
    && echo "[FORCE]   NMAR jitter inference..." \
    || echo "[MISSING] NMAR jitter (brak jitter_Clean*_nmar2.raw) — submituje 01_nmar_infer_jitter.sh..."
  JOB_INFER_NMAR=$(sbatch --parsable --export=ALL \
    ${SCRIPT_DIR}/prep/01_nmar_infer_jitter.sh)
  echo "  -> job id: ${JOB_INFER_NMAR}"
else
  echo "[OK]      NMAR jitter: ${NMAR_DIR}  (znaleziono jitter_Clean*_nmar2.raw)"
fi

# ---- 3. Graf jitter — kanały segmentacji (jitter_graph, nmar-graph, angle-graph) ----
GRAPH_DIR="${TEAM_DIR}/data/seg_channels_jitter"
if [[ $FORCE_INFER -eq 1 ]] || ! has_raw_files "${GRAPH_DIR}"; then
  # Graf wymaga NIfTI CT i segmentacji jako wejścia — sprawdź czy są
  CT_NII_DIR="${TEAM_DIR}/output/3framesjitterRawRealNifti"
  SEG_NII_DIR="${TEAM_DIR}/output/3framesjitterRawRealSeg"
  if [[ ! -d "${CT_NII_DIR}" ]] || [[ ! -d "${SEG_NII_DIR}" ]]; then
    echo "[ERROR]   Graf jitter NIE MOŻE być wygenerowany:"
    [[ ! -d "${CT_NII_DIR}" ]] && echo "          Brak NIfTI CT: ${CT_NII_DIR}"
    [[ ! -d "${SEG_NII_DIR}" ]] && echo "          Brak segmentacji nnUNet: ${SEG_NII_DIR}"
    echo "          Uruchom pipeline nnUNet na datasecie jitter najpierw."
    MISSING=1
  else
    [[ $FORCE_INFER -eq 1 ]] \
      && echo "[FORCE]   Graf jitter (graph_channel_generator)..." \
      || echo "[MISSING] Graf jitter — submituje 02_graph_jitter.sh automatycznie..."
    JOB_INFER_GRAPH=$(sbatch --parsable --export=ALL \
      ${SCRIPT_DIR}/prep/02_graph_jitter.sh)
    echo "  -> job id: ${JOB_INFER_GRAPH}"
  fi
else
  echo "[OK]      Graf jitter: ${GRAPH_DIR}"
fi

if [[ $MISSING -eq 1 ]]; then
  echo ""
  echo "[ABORT] Brakujące wejścia uniemożliwiają uruchomienie. Przerwano."
  exit 1
fi

echo ""
echo "Pre-flight OK — zgłaszam joby refinera."
echo ""

# ============================================================
# Folder zbiorczy dla tego runu
# ============================================================
export BATCH_DIR=${TEAM_DIR}/CircNet/runs/jitter_circ${RADIUS_FEATURES}f_ref${REFINER_BASE}f_$(date +%Y%m%d_%H%M)
mkdir -p ${BATCH_DIR}
echo "Folder zbiorczy: ${BATCH_DIR}"
echo ""

# ============================================================
# Buduj łańcuchy zależności dla każdej grupy wariantów:
#   base          — brak zależności
#   nmar          — czeka na NMAR
#   graph         — czeka na Graf
#   nmar-graph    — czeka na NMAR + Graf
#   angle         — czeka na CircNet
#   nmar-angle    — czeka na NMAR + CircNet
#   angle-graph   — czeka na CircNet + Graf
# ============================================================

dep_for() {
  # Buduje "--dependency=afterok:J1:J2:..." z niepustych argumentów
  local jobs=""
  for j in "$@"; do
    [[ -n "$j" ]] && jobs="${jobs:+${jobs}:}${j}"
  done
  [[ -n "$jobs" ]] && echo "--dependency=afterok:${jobs}"
}

DEP_BASE=""
DEP_NMAR=$(dep_for "${JOB_INFER_NMAR}")
DEP_GRAPH=$(dep_for "${JOB_INFER_GRAPH}")
DEP_NMAR_GRAPH=$(dep_for "${JOB_INFER_NMAR}" "${JOB_INFER_GRAPH}")
DEP_ANGLE=$(dep_for "${JOB_INFER_CIRCNET}")
DEP_NMAR_ANGLE=$(dep_for "${JOB_INFER_NMAR}" "${JOB_INFER_CIRCNET}")
DEP_ANGLE_GRAPH=$(dep_for "${JOB_INFER_CIRCNET}" "${JOB_INFER_GRAPH}")

# ============================================================
# Refiner — wszystkie warianty
# ============================================================

echo "[SUBMIT] jitter_base..."
JOB_BASE=$(sbatch --parsable --export=ALL \
  ${DEP_BASE} ${SCRIPT_DIR}/jitter/jitter_base.sh)
echo "  -> job id: ${JOB_BASE}"

echo "[SUBMIT] jitter_nmar..."
JOB_NMAR=$(sbatch --parsable --export=ALL \
  ${DEP_NMAR} ${SCRIPT_DIR}/jitter/jitter_nmar.sh)
echo "  -> job id: ${JOB_NMAR}"

echo "[SUBMIT] jitter_graph..."
JOB_GRAPH=$(sbatch --parsable --export=ALL \
  ${DEP_GRAPH} ${SCRIPT_DIR}/jitter/jitter_graph.sh)
echo "  -> job id: ${JOB_GRAPH}"

echo "[SUBMIT] jitter_nmar-graph..."
JOB_NMAR_GRAPH=$(sbatch --parsable --export=ALL \
  ${DEP_NMAR_GRAPH} ${SCRIPT_DIR}/jitter/jitter_nmar-graph.sh)
echo "  -> job id: ${JOB_NMAR_GRAPH}"

echo "[SUBMIT] jitter_angle..."
JOB_ANGLE=$(sbatch --parsable --export=ALL \
  ${DEP_ANGLE} ${SCRIPT_DIR}/jitter/jitter_angle.sh)
echo "  -> job id: ${JOB_ANGLE}"

echo "[SUBMIT] jitter_nmar-angle..."
JOB_NMAR_ANGLE=$(sbatch --parsable --export=ALL \
  ${DEP_NMAR_ANGLE} ${SCRIPT_DIR}/jitter/jitter_nmar-angle.sh)
echo "  -> job id: ${JOB_NMAR_ANGLE}"

echo "[SUBMIT] jitter_angle-graph..."
JOB_ANGLE_GRAPH=$(sbatch --parsable --export=ALL \
  ${DEP_ANGLE_GRAPH} ${SCRIPT_DIR}/jitter/jitter_angle-graph.sh)
echo "  -> job id: ${JOB_ANGLE_GRAPH}"

echo "[SUBMIT] jitter_3f-base..."
JOB_3F_BASE=$(sbatch --parsable --export=ALL \
  ${SCRIPT_DIR}/jitter/jitter_3f-base.sh)
echo "  -> job id: ${JOB_3F_BASE}"

echo "[SUBMIT] jitter_3f-circnet..."
JOB_3F_CIRCNET=$(sbatch --parsable --export=ALL \
  ${DEP_ANGLE} ${SCRIPT_DIR}/jitter/jitter_3f-circnet.sh)
echo "  -> job id: ${JOB_3F_CIRCNET}"

# ============================================================
# collect_metrics po zakończeniu wszystkich runów
# ============================================================
ALL_JOBS="${JOB_BASE}:${JOB_NMAR}:${JOB_GRAPH}:${JOB_NMAR_GRAPH}:${JOB_ANGLE}:${JOB_NMAR_ANGLE}:${JOB_ANGLE_GRAPH}:${JOB_3F_BASE}:${JOB_3F_CIRCNET}"

RUN_BASE=${BATCH_DIR}/refiner_jitter_base_${REFINER_BASE}f_${JOB_BASE}
RUN_NMAR=${BATCH_DIR}/refiner_jitter_nmar_${REFINER_BASE}f_${JOB_NMAR}
RUN_GRAPH=${BATCH_DIR}/refiner_jitter_graph_${REFINER_BASE}f_${JOB_GRAPH}
RUN_NMAR_GRAPH=${BATCH_DIR}/refiner_jitter_nmar-graph_${REFINER_BASE}f_${JOB_NMAR_GRAPH}
RUN_ANGLE=${BATCH_DIR}/refiner_jitter_angle_${REFINER_BASE}f_${JOB_ANGLE}
RUN_NMAR_ANGLE=${BATCH_DIR}/refiner_jitter_nmar-angle_${REFINER_BASE}f_${JOB_NMAR_ANGLE}
RUN_ANGLE_GRAPH=${BATCH_DIR}/refiner_jitter_angle-graph_${REFINER_BASE}f_${JOB_ANGLE_GRAPH}
RUN_3F_BASE=${BATCH_DIR}/refiner_jitter_3f-base_${REFINER_BASE}f_${JOB_3F_BASE}
RUN_3F_CIRCNET=${BATCH_DIR}/refiner_jitter_3f-circnet_${REFINER_BASE}f_${JOB_3F_CIRCNET}

COLLECT_OUT=${BATCH_DIR}/jitter_metrics_summary.csv

echo "[SUBMIT] collect_metrics (po zakończeniu wszystkich runów)..."
JOB_COLLECT=$(sbatch --parsable \
  --dependency=afterany:${ALL_JOBS} \
  --job-name=jitter_collect \
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
          /app/CircNet${RUN_GRAPH#${TEAM_DIR}/CircNet} \
          /app/CircNet${RUN_NMAR_GRAPH#${TEAM_DIR}/CircNet} \
          /app/CircNet${RUN_ANGLE#${TEAM_DIR}/CircNet} \
          /app/CircNet${RUN_NMAR_ANGLE#${TEAM_DIR}/CircNet} \
          /app/CircNet${RUN_ANGLE_GRAPH#${TEAM_DIR}/CircNet} \
          /app/CircNet${RUN_3F_BASE#${TEAM_DIR}/CircNet} \
          /app/CircNet${RUN_3F_CIRCNET#${TEAM_DIR}/CircNet} \
        --output /app/CircNet${COLLECT_OUT#${TEAM_DIR}/CircNet}
    echo \"[INFO] collect_metrics finished: \$(date)\"
    echo \"[INFO] Tabela: ${COLLECT_OUT}\"
  ")
echo "  -> job id: ${JOB_COLLECT}"

echo "========================================================"
echo " Zgłoszone priory:"
[[ -n $JOB_INFER_CIRCNET ]] && echo "  CircNet infer  : ${JOB_INFER_CIRCNET}"
[[ -n $JOB_INFER_NMAR    ]] && echo "  NMAR infer     : ${JOB_INFER_NMAR}"
[[ -n $JOB_INFER_GRAPH   ]] && echo "  Graf generate  : ${JOB_INFER_GRAPH}"
echo ""
echo " Zgłoszone joby refinera:"
echo "  jitter_base       : ${JOB_BASE}"
echo "  jitter_nmar       : ${JOB_NMAR}       ${DEP_NMAR:+(dep: NMAR)}"
echo "  jitter_graph      : ${JOB_GRAPH}       ${DEP_GRAPH:+(dep: Graf)}"
echo "  jitter_nmar-graph : ${JOB_NMAR_GRAPH}  ${DEP_NMAR_GRAPH:+(dep: NMAR+Graf)}"
echo "  jitter_angle      : ${JOB_ANGLE}       ${DEP_ANGLE:+(dep: CircNet)}"
echo "  jitter_nmar-angle : ${JOB_NMAR_ANGLE}  ${DEP_NMAR_ANGLE:+(dep: NMAR+CircNet)}"
echo "  jitter_angle-graph: ${JOB_ANGLE_GRAPH} ${DEP_ANGLE_GRAPH:+(dep: CircNet+Graf)}"
echo "  jitter_3f-base    : ${JOB_3F_BASE}"
echo "  jitter_3f-circnet : ${JOB_3F_CIRCNET}  ${DEP_ANGLE:+(dep: CircNet)}"
echo "  collect_metrics   : ${JOB_COLLECT}     (afterany powyższych)"
echo ""
echo " Tabela wynikowa: ${COLLECT_OUT}"
echo " Sprawdź status:  squeue -u \$USER"
echo "========================================================"
