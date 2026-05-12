#!/usr/bin/env bash
set -euo pipefail

# Paper-faithful LaBraM fine-tuning run for local FACED 8-class data.
#
# Override any setting from the command line, for example:
#   BATCH_SIZE=64 MAX_EPOCHS=50 LR=5e-4 bash src/biosignal/run_labram.sh

PYTHON_BIN="${PYTHON_BIN:-/data/scratch/linrui/conda_envs/neuromusic/bin/python}"

MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
LAYER_DECAY="${LAYER_DECAY:-0.65}"
MODEL_EMA_DECAY="${MODEL_EMA_DECAY:-0.9999}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.8}"
INPUT_DIVISOR="${INPUT_DIVISOR:-100.0}"
PATIENCE="${PATIENCE:-15}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"

CACHE_DIR="${CACHE_DIR:-data/processed/faced/labram_raw_float32}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/faced/run_${RUN_TIMESTAMP}}"
HF_CACHE_DIR="${HF_CACHE_DIR:-data/processed/hf_cache}"
LOG_PATH="${OUTPUT_DIR}/train.log"
CONFIG_PATH="${OUTPUT_DIR}/run_config.env"

BUILD_CACHE_ARG=()
if [[ "${BUILD_CACHE:-0}" == "1" ]]; then
  BUILD_CACHE_ARG=(--build-cache)
fi

FREEZE_ARG=()
if [[ "${FREEZE_ENCODER:-0}" == "1" ]]; then
  FREEZE_ARG=(--freeze-encoder)
fi

NO_PREPROCESSING_ARG=()
if [[ "${NO_PREPROCESSING:-0}" == "1" ]]; then
  NO_PREPROCESSING_ARG=(--no-preprocessing)
fi

DISABLE_EMA_ARG=()
if [[ "${DISABLE_MODEL_EMA:-0}" == "1" ]]; then
  DISABLE_EMA_ARG=(--disable-model-ema)
fi

ALLOW_CPU_ARG=()
if [[ "${ALLOW_CPU:-0}" == "1" ]]; then
  ALLOW_CPU_ARG=(--allow-cpu)
fi

mkdir -p "${OUTPUT_DIR}"

{
  echo "RUN_TIMESTAMP=${RUN_TIMESTAMP}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "MAX_EPOCHS=${MAX_EPOCHS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "LR=${LR}"
  echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
  echo "MIN_LR=${MIN_LR}"
  echo "WARMUP_EPOCHS=${WARMUP_EPOCHS}"
  echo "LAYER_DECAY=${LAYER_DECAY}"
  echo "MODEL_EMA_DECAY=${MODEL_EMA_DECAY}"
  echo "MAX_GRAD_NORM=${MAX_GRAD_NORM}"
  echo "INPUT_DIVISOR=${INPUT_DIVISOR}"
  echo "PATIENCE=${PATIENCE}"
  echo "SEED=${SEED}"
  echo "NUM_WORKERS=${NUM_WORKERS}"
  echo "CACHE_DIR=${CACHE_DIR}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "HF_CACHE_DIR=${HF_CACHE_DIR}"
  echo "BUILD_CACHE=${BUILD_CACHE:-0}"
  echo "FREEZE_ENCODER=${FREEZE_ENCODER:-0}"
  echo "NO_PREPROCESSING=${NO_PREPROCESSING:-0}"
  echo "DISABLE_MODEL_EMA=${DISABLE_MODEL_EMA:-0}"
  echo "LABEL_SMOOTHING=${LABEL_SMOOTHING}"
} > "${CONFIG_PATH}"

CMD=(
  "${PYTHON_BIN}" -m src.biosignal.train_faced_labram
  "${BUILD_CACHE_ARG[@]}"
  "${FREEZE_ARG[@]}"
  "${NO_PREPROCESSING_ARG[@]}"
  "${DISABLE_EMA_ARG[@]}"
  "${ALLOW_CPU_ARG[@]}"
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --hf-cache-dir "${HF_CACHE_DIR}"
  --max-epochs "${MAX_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --min-lr "${MIN_LR}"
  --warmup-epochs "${WARMUP_EPOCHS}"
  --layer-decay "${LAYER_DECAY}"
  --model-ema-decay "${MODEL_EMA_DECAY}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --input-divisor "${INPUT_DIVISOR}"
  --early-stopping-patience "${PATIENCE}"
  --seed "${SEED}"
  --num-workers "${NUM_WORKERS}"
  --label-smoothing "${LABEL_SMOOTHING}"
)

{
  echo "Command: ${CMD[*]}"
  "${CMD[@]}"
} 2>&1 | tee "${LOG_PATH}"
