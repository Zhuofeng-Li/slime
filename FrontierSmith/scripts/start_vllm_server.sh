#!/bin/bash
# Start vLLM server for Frontier-CS evaluation.
#
# Usage:
#   MODEL_PATH=Qwen/Qwen3.5-9B bash scripts/start_vllm_server.sh
#   MODEL_PATH=models/qwen35_9b_grpo_step105 PORT=8000 bash scripts/start_vllm_server.sh
#
# Uses CUDA_VISIBLE_DEVICES (default 4,5,6,7). Stop with Ctrl+C.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
PORT=${PORT:-8000}
TP=${TP:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-24576}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}

echo "Starting vLLM server: model=$MODEL_PATH port=$PORT TP=$TP"
exec vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    "$@"
