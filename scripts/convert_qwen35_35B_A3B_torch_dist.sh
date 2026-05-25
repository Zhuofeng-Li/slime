#!/bin/bash
# Usage: bash scripts/convert_qwen35_35B_A3B_torch_dist.sh
# Run inside slimerl/slime:latest container via launch.sh or docker run directly.
# Converts Qwen3.5-35B-A3B HF weights to Megatron torch_dist format.

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3.5-35B-A3B.sh"

BASE_FOLDER="${BASE_FOLDER:-/root}"

echo "[convert] HF checkpoint : ${BASE_FOLDER}/Qwen3.5-35B-A3B"
echo "[convert] Save to       : ${BASE_FOLDER}/Qwen3.5-35B-A3B_torch_dist"

# /root/Megatron-LM is only valid when a custom Megatron is checked out under BASE_DIR.
# The slimerl/slime:latest image ships Megatron system-wide, so fall back gracefully.
MEGATRON_PATH="/root/Megatron-LM"
if [ ! -d "${MEGATRON_PATH}" ]; then
  echo "[convert] ${MEGATRON_PATH} not found, using system-installed Megatron"
  MEGATRON_PATH=""
fi

PYTHONPATH="${MEGATRON_PATH}" python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${BASE_FOLDER}/Qwen3.5-35B-A3B" \
    --save "${BASE_FOLDER}/Qwen3.5-35B-A3B_torch_dist"

echo "[convert] Done."
