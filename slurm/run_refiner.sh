#!/bin/bash -l

#SBATCH -J circnet_refiner
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --output="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet/slurm_logs/refiner_%j.out"
#SBATCH --error="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet/slurm_logs/refiner_%j.err"

export CUDA_LAUNCH_BLOCKING=1
export APPTAINER_CACHEDIR=/net/storage/pr3/plgrid/plggsmartudlteam//apptainer_cache
export PYTHONPATH="$PYTHONPATH:/"

TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam

# =========================================================
# KONFIGURACJA — zmień przed uruchomieniem
# =========================================================
CIRCNET_RUN="mar_run_XXXXX"       # <-- run CircNetu (np. mar_run_16838558)
RADIUS_FEATURES=64                 # <-- zgodnie z użytym CircNetem (32 lub 64)
TRAIN_FOLDERS="body1 body2 body3"
VAL_FOLDER="body4"
TEST_FOLDER="body5"
REFINER_EPOCHS=500
REFINER_BATCH=4
REFINER_BASE=64                    # Bazowy wymiar kanałów UNet refinera
# =========================================================

CORR_DIR=${TEAM_DIR}/CircNet/${CIRCNET_RUN}/corrected
REFINER_DIR=${TEAM_DIR}/CircNet/refiner_run_${SLURM_JOB_ID}

mkdir -p ${TEAM_DIR}/CircNet/slurm_logs
mkdir -p ${REFINER_DIR}

echo "[INFO] Job started: $(date)"
echo "[INFO] Running on: $(hostname)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] CircNet run: ${CIRCNET_RUN}"
echo "[INFO] Corrected dir: ${CORR_DIR}"
echo "[INFO] Refiner output: ${REFINER_DIR}"

# =========================================================
# Krok 1: Inferencja CircNetu
# Generuje pliki .raw dla WSZYSTKICH folderów (train+val+test)
# Ustaw FORCE_INFERENCE=true żeby wymusić ponowną inferencję
# nawet jeśli katalog corrected już istnieje.
# =========================================================
FORCE_INFERENCE=true

if [ "${FORCE_INFERENCE}" = "false" ] && [ -d "${CORR_DIR}" ] && [ "$(ls -A ${CORR_DIR} 2>/dev/null)" ]; then
    echo "[INFO] Katalog corrected istnieje i nie jest pusty — pomijam inferencję CircNetu."
else
    echo "[INFO] Uruchamiam inferencję CircNetu dla wszystkich folderów..."
    apptainer exec --nv \
      --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
      ${TEAM_DIR}/sif_images/sif_image.sif \
      python3 /app/CircNet/inference.py \
        --model-path      /app/CircNet/${CIRCNET_RUN}/best_model.pth \
        --data-root       /app/data/RPI/body \
        --test-folders    ${TRAIN_FOLDERS} ${VAL_FOLDER} ${TEST_FOLDER} \
        --output-dir      /app/CircNet/${CIRCNET_RUN}/corrected \
        --polar-size      512x1024 \
        --radius-features ${RADIUS_FEATURES} \
      2>&1 | tee ${TEAM_DIR}/CircNet/slurm_logs/inference_for_refiner_${SLURM_JOB_ID}.log

    if [ $? -ne 0 ]; then
        echo "[ERROR] Inferencja CircNetu nie powiodła się. Przerywam."
        exit 1
    fi
    echo "[INFO] Inferencja CircNetu zakończona."
fi

# =========================================================
# Krok 2: Trening refinera (UNet)
# =========================================================
echo "[INFO] Uruchamiam trening refinera..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/circnet_refiner_train.py \
    --data-root        /app/data/RPI/body \
    --circnet-corr-dir /app/CircNet/${CIRCNET_RUN}/corrected \
    --train-folders    ${TRAIN_FOLDERS} \
    --val-folders      ${VAL_FOLDER} \
    --output-dir       /app/CircNet/refiner_run_${SLURM_JOB_ID} \
    --epochs           ${REFINER_EPOCHS} \
    --batch-size       ${REFINER_BATCH} \
    --num-workers      8 \
    --base-features    ${REFINER_BASE} \
    --lr               2e-4 \
    --lam-out          1.0 \
    --lam-in           0.25 \
  2>&1 | tee ${TEAM_DIR}/CircNet/slurm_logs/refiner_train_${SLURM_JOB_ID}.log

if [ $? -ne 0 ]; then
    echo "[ERROR] Trening refinera nie powiódł się. Przerywam."
    exit 1
fi
echo "[INFO] Trening refinera zakończony."

# =========================================================
# Krok 3: Metryki refinera na zbiorze testowym
# =========================================================
echo "[INFO] Liczę metryki refinera na ${TEST_FOLDER}..."
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/circnet_refiner_metrics.py \
    --model-path       /app/CircNet/refiner_run_${SLURM_JOB_ID}/best_refiner.pth \
    --data-root        /app/data/RPI/body \
    --prior-dir        /app/CircNet/${CIRCNET_RUN}/corrected \
    --test-folder      ${TEST_FOLDER} \
    --output-dir       /app/CircNet/refiner_run_${SLURM_JOB_ID}/metrics \
    --base-features    ${REFINER_BASE} \
  2>&1 | tee ${TEAM_DIR}/CircNet/slurm_logs/refiner_metrics_${SLURM_JOB_ID}.log

echo "[INFO] Metryki zakończone."
echo "[INFO] Job finished: $(date)"
echo "[INFO] Refiner zapisany w: ${REFINER_DIR}"
