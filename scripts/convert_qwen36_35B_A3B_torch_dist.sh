#!/bin/bash
# Usage: bash scripts/convert_qwen36_35B_A3B_torch_dist.sh
# Run inside slimerl/slime:latest container via scripts/launch.sh.
# Converts Qwen3.6-35B-A3B HF weights to Megatron torch_dist format.

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3.6-35B-A3B.sh"

BASE_FOLDER="${BASE_FOLDER:-/root}"
NUM_GPUS="${NUM_GPUS:-8}"
HF_CKPT="${BASE_FOLDER}/models/Qwen3.6-35B-A3B"
SAVE_DIR="${BASE_FOLDER}/models/Qwen3.6-35B-A3B_torch_dist"

echo "[convert] HF checkpoint : ${HF_CKPT}"
echo "[convert] Save to       : ${SAVE_DIR}"
echo "[convert] NUM_GPUS      : ${NUM_GPUS}"

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
