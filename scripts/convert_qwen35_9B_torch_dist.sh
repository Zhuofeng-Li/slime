#!/bin/bash
# Usage: bash scripts/convert_qwen35_9B_torch_dist.sh
# Run inside slimerl/slime:latest container via scripts/launch.sh.
# Converts Qwen3.5-9B HF weights to Megatron torch_dist format.

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3.5-9B.sh"

BASE_FOLDER="${BASE_FOLDER:-/root}"
NUM_GPUS="${NUM_GPUS:-8}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
HF_CKPT="${HF_CKPT:-${BASE_FOLDER}/models/${MODEL_NAME}}"
SAVE_DIR="${SAVE_DIR:-${BASE_FOLDER}/models/${MODEL_NAME}_torch_dist}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"

echo "[convert] Model         : ${MODEL_NAME}"
echo "[convert] HF checkpoint : ${HF_CKPT}"
echo "[convert] Save to       : ${SAVE_DIR}"
echo "[convert] NUM_GPUS      : ${NUM_GPUS}"
echo "[convert] TP/PP         : ${TP_SIZE}/${PP_SIZE}"

if [[ ! -d "${HF_CKPT}" ]]; then
  echo "[convert] ERROR: HF checkpoint not found: ${HF_CKPT}"
  exit 1
fi

MEGATRON_PATH="/root/Megatron-LM"
if [ ! -d "${MEGATRON_PATH}" ]; then
  echo "[convert] ${MEGATRON_PATH} not found, using system-installed Megatron"
  MEGATRON_PATH=""
fi

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

PYTHONPATH="${MEGATRON_PATH}" torchrun --nproc-per-node "${NUM_GPUS}" tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --tensor-model-parallel-size "${TP_SIZE}" \
    --pipeline-model-parallel-size "${PP_SIZE}" \
    --disable-auto-pipeline-parallel \
    --hf-checkpoint "${HF_CKPT}" \
    --save "${SAVE_DIR}"

echo "[convert] Done."
