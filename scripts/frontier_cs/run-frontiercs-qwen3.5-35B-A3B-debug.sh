#!/bin/bash
# Usage: bash scripts/frontier_cs/run-frontiercs-qwen3.5-35B-A3B-debug.sh
#
# FrontierCS GRPO debug training on Qwen3.5-35B-A3B (8x GPU, 1 data item).
# Parameters aligned with FrontierCS Qwen3.5-27B VERL config:
#   lr=5e-7, kl_loss_coef=0.001, n_samples=8, batch_size=8, max_response=32000
#
# Prerequisites:
#   1. uv run python scripts/frontier_cs/prepare_frontiercs_jsonl.py --full-for-both
#   2. Frontier-CS judge running: cd FrontierSmith/Frontier-CS/algorithmic && ./run_judge.sh
#   3. Model weights at ${BASE_FOLDER}/models/Qwen3.5-35B-A3B (HF format)
#   4. torch_dist weights at ${BASE_FOLDER}/models/Qwen3.5-35B-A3B_torch_dist/
#      (convert with: bash scripts/convert_qwen35_35B_A3B_torch_dist.sh if needed)

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

export PYTHONBUFFERED=16

BASE_FOLDER="${BASE_FOLDER:-/root}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-frontiercs-debug}"
CKPT_DIR="${CKPT_DIR:-${BASE_FOLDER}/models/${RUN_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${BASE_FOLDER}/logs/frontiercs_rollouts/${RUN_NAME}}"

mkdir -p "${CKPT_DIR}" "${ROLLOUT_DIR}"
echo "RUN_NAME: ${RUN_NAME}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Debug rollout dir: ${ROLLOUT_DIR}"

# FrontierSmith must be importable for the custom RM.
# /root/slime is the mount point used by launch.sh.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/root/slime:/root/slime/FrontierSmith"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

# ── Checkpoints ───────────────────────────────────────────────────────────
CKPT_ARGS=(
   --hf-checkpoint "${BASE_FOLDER}/models/${MODEL_NAME}"
   --ref-load "${BASE_FOLDER}/models/${MODEL_NAME}_torch_dist"
   --load "${BASE_FOLDER}/models/${MODEL_NAME}_torch_dist"
   --save "${BASE_FOLDER}/models/${MODEL_NAME}_torch_dist"
   --save-interval 200
)

# ── Data ──────────────────────────────────────────────────────────────────
# prompt format: [{"role": "user", "content": "...C++ problem..."}]
# label: problem_id string (passed to judge as pid)
# No --rm-type; reward is fully handled by --custom-rm-path.
ROLLOUT_ARGS=(
   --prompt-data "${BASE_FOLDER}/data/frontiercs/train.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --custom-rm-path "FrontierSmith.slime_rm.frontiercs_rm.batched_custom_rm"
   --custom-rollout-log-function-path "scripts.frontier_cs.rollout_wandb.log_rollout_data"
   --save-debug-rollout-data "${ROLLOUT_DIR}/{rollout_id}.pt"

   # Fast debug profile: keep one prompt per rollout, but reduce total samples
   # and max generation length so judge/eval/training cycles finish sooner.
   --num-rollout 32
   --rollout-batch-size 1
   --n-samples-per-prompt 32
   --rollout-max-response-len 81920
   --rollout-temperature 1.0

   --global-batch-size 32
   --balance-data
)

# ── Eval ──────────────────────────────────────────────────────────────────
# Eval also goes through the same judge via --custom-rm-path (set globally above).
# val.jsonl contains all problems (--full-for-both) so we get full coverage.
EVAL_ARGS=(
   --eval-prompt-data frontiercs "${BASE_FOLDER}/data/frontiercs/val.jsonl"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 81920
   --eval-top-p 1.0
)

# ── PERF (same as run-qwen3.5-35B-A3B.sh) ────────────────────────────────
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 5000
)

# ── GRPO (from FrontierCS 27B) ────────────────────────────────────────────
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-tis
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# ── Optimizer (from FrontierCS 27B: lr=5e-7) ─────────────────────────────
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-7
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

# ── WandB ─────────────────────────────────────────────────────────────────
WANDB_ARGS=(
   --use-wandb
   --wandb-project frontiercs-slime
   --wandb-group "${RUN_NAME}"
   --wandb-key "${WANDB_KEY}"
)

# ── SGLang (same as run-qwen3.5-35B-A3B.sh) ──────────────────────────────
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-max-running-requests 64
   --sglang-server-concurrency 64
   --sglang-disable-custom-all-reduce
   --sglang-mamba-scheduler-strategy extra_buffer
)

# ── Misc ──────────────────────────────────────────────────────────────────
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --log-probs-chunk-size 4096
   --train-memory-margin-bytes 0
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"max_split_size_mb:512"}'
)

DEBUG_ARGS=()
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]]; then
   DEBUG_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
fi
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE:-}" ]]; then
   DEBUG_ARGS+=(--load-debug-rollout-data-subsample "${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE}")
fi
if [[ "${DEBUG_TRAIN_ONLY:-0}" == "1" ]]; then
   DEBUG_ARGS+=(--debug-train-only)
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/slime:/root/slime/FrontierSmith\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"FRONTIER_JUDGE_URL\": \"http://localhost:8082\",
    \"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK\": \"false\"
  }
}"

JOB_SUBMIT_OUT=$(ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --no-wait \
   -- python3 "${SLIME_DIR}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DEBUG_ARGS[@]}")
JOB_ID=$(echo "$JOB_SUBMIT_OUT" | grep -oP "raysubmit_\w+" | head -1)
echo "Job submitted: $JOB_ID"

# Follow logs; retry if WebSocket drops (1006)
while true; do
    ray job logs --follow "$JOB_ID" 2>/dev/null || true
    STATUS=$(ray job status "$JOB_ID" 2>/dev/null | grep -oP "Status:\s*\K\w+")
    echo "Job status: $STATUS"
    if [[ "$STATUS" == "SUCCEEDED" || "$STATUS" == "FAILED" || "$STATUS" == "STOPPED" ]]; then
        echo "Job ended with status: $STATUS"
        [[ "$STATUS" == "SUCCEEDED" ]] && exit 0 || exit 1
    fi
    echo "Log stream dropped, retrying in 10s..."
    sleep 10
done
