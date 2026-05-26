#!/bin/bash
# Start VERL inference server (OpenAI-compatible API, like vllm serve)
#
# For Qwen3.5-VL with DTensor issues, use text-only model:
#   python scripts/merge_fsdp_to_hf.py --ckpt <ckpt> --output models/xxx_text_only --text-only
#   MODEL_PATH=models/xxx_text_only bash scripts/run_verl_inference_server.sh
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/run_verl_inference_server.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH=${MODEL_PATH:-$PROJECT_ROOT/models/qwen35_9b_grpo_step105}
TP=${TP:-1}
NGPU=${NGPU:-4}

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found: $MODEL_PATH"
    exit 1
fi

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

python scripts/run_verl_inference_server.py \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    "$@"
