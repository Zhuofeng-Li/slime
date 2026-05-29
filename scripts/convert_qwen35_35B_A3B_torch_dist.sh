#!/bin/bash
# Usage: bash scripts/convert_qwen35_35B_A3B_torch_dist.sh
# Run inside slimerl/slime:latest container via launch.sh or docker run directly.
# Converts Qwen3.5-35B-A3B HF weights to Megatron torch_dist format.

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3.5-35B-A3B.sh"

BASE_FOLDER="${BASE_FOLDER:-/root}"
NUM_GPUS="${NUM_GPUS:-8}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
HF_CKPT="${HF_CKPT:-${BASE_FOLDER}/models/${MODEL_NAME}}"
SAVE_DIR="${SAVE_DIR:-${BASE_FOLDER}/models/${MODEL_NAME}_torch_dist}"

echo "[convert] Model         : ${MODEL_NAME}"
echo "[convert] HF checkpoint : ${HF_CKPT}"
echo "[convert] Save to       : ${SAVE_DIR}"
echo "[convert] NUM_GPUS      : ${NUM_GPUS}"

if [[ ! -d "${HF_CKPT}" ]]; then
  echo "[convert] ERROR: HF checkpoint not found: ${HF_CKPT}"
  exit 1
fi

# /root/Megatron-LM is only valid when a custom Megatron is checked out under BASE_DIR.
# The slimerl/slime:latest image ships Megatron system-wide, so fall back gracefully.
MEGATRON_PATH="/root/Megatron-LM"
if [ ! -d "${MEGATRON_PATH}" ]; then
  echo "[convert] ${MEGATRON_PATH} not found, using system-installed Megatron"
  MEGATRON_PATH=""
fi

PYTHONPATH="${MEGATRON_PATH}" torchrun --nproc-per-node "${NUM_GPUS}" tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CKPT}" \
    --save "${SAVE_DIR}"

echo "[convert] Done."
