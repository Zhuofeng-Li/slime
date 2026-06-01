#!/bin/bash
# Usage:
#   MASTER_ADDR=10.0.0.1 \
#   HOSTFILE=/shared/dev/zhuofeng/slime/hostfile \
#   WANDB_KEY=<your_key> \
#   BASE_FOLDER=/shared/dev/zhuofeng/slime \
#   bash scripts/frontier_cs/run-frontiercs-qwen3.5-27B_debug-64_h200.sh
#
# FrontierCS GRPO debug training on Qwen3.5-27B, 8 nodes × 8 H200 = 64 GPU.
# Debug variant: cuda-graph disabled, custom-all-reduce disabled, DEBUG_ARGS supported.
#
# Training parallelism: TP=4, PP=2, CP=4 → 32 GPU per replica, DP=2
# Rollout: rollout_num_gpus_per_engine=4 → 16 SGLang engines (TP=4 each)
# H200-specific: max-tokens-per-gpu=10000, sglang-mem-fraction-static=0.80
#
# Required env vars:
#   MASTER_ADDR   IP of worker-0 (hostname -I | awk '{print $1}')
#   HOSTFILE      file with one worker IP per line (all 8 nodes)
#   WANDB_KEY     WandB API key
#
# Optional env vars (with defaults):
#   BASE_FOLDER                  /root
#   MODEL_NAME                   Qwen3.5-27B
#   ACTOR_NUM_NODES              8
#   ACTOR_NUM_GPUS_PER_NODE      8
#   NUM_ROLLOUT                  2000
#   ROLLOUT_BATCH_SIZE           32
#   N_SAMPLES_PER_PROMPT         8
#   ROLLOUT_MAX_RESPONSE_LEN     81920
#   GLOBAL_BATCH_SIZE            256
#   SGLANG_MEM_FRACTION_STATIC   0.80
#   SAVE_INTERVAL                (unset = no saving)
#   SOCKET_IFNAME                eth0
#   LOAD_DEBUG_ROLLOUT_DATA      path to .pt file (skip rollout, load cached)
#   LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE  number of samples to subsample
#   DEBUG_TRAIN_ONLY             1 = skip rollout entirely
#
# Prerequisites:
#   1. uv run python scripts/frontier_cs/prepare_frontiercs_jsonl.py --full-for-both
#   2. Frontier-CS judge running on this node (port 8082)
#   3. Model weights at ${BASE_FOLDER}/models/Qwen3.5-27B (HF format)
#   4. torch_dist weights at ${BASE_FOLDER}/models/Qwen3.5-27B_torch_dist/
#   5. mkdir -p ${BASE_FOLDER}/logs/frontiercs_rollouts/

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

# unset proxy to avoid distributed startup issues
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

BASE_FOLDER="${BASE_FOLDER:-/root}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-frontiercs-debug-64-h200}"
CKPT_DIR="${CKPT_DIR:-${BASE_FOLDER}/models/${RUN_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${BASE_FOLDER}/logs/frontiercs_rollouts/${RUN_NAME}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-2000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-81920}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
SOCKET_IFNAME="${SOCKET_IFNAME:-eth0}"

MASTER_ADDR="${MASTER_ADDR:-}"
if [ -z "${MASTER_ADDR}" ]; then
  echo "MASTER_ADDR is not set. Please set it to the master node IP."
  exit 1
fi

mkdir -p "${CKPT_DIR}" "${ROLLOUT_DIR}"
echo "RUN_NAME: ${RUN_NAME}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Debug rollout dir: ${ROLLOUT_DIR}"

# FrontierSmith must be importable for the custom RM.
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
source "${SLIME_DIR}/scripts/models/qwen3.5-27B.sh"

# ── Checkpoints ───────────────────────────────────────────────────────────
CKPT_ARGS=(
   --hf-checkpoint "${BASE_FOLDER}/models/${MODEL_NAME}"
   --ref-load "${BASE_FOLDER}/models/${MODEL_NAME}_torch_dist"
   --load "${CKPT_DIR}/"
)
if [[ -n "${SAVE_INTERVAL:-}" ]]; then
   CKPT_ARGS+=(
      --save "${CKPT_DIR}/"
      --save-interval "${SAVE_INTERVAL}"
   )
fi

# ── Data ──────────────────────────────────────────────────────────────────
ROLLOUT_ARGS=(
   --prompt-data "${BASE_FOLDER}/data/frontiercs_full/train.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --custom-rm-path "FrontierSmith.slime_rm.frontiercs_rm.batched_custom_rm"
   --custom-rollout-log-function-path "scripts.frontier_cs.rollout_wandb.log_rollout_data"
   --save-debug-rollout-data "${ROLLOUT_DIR}/{rollout_id}.pt"

   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1.0

   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)
if [[ -n "${APPLY_CHAT_TEMPLATE_KWARGS:-}" ]]; then
   ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi
if [[ -n "${DYNAMIC_SAMPLING_FILTER_PATH:-}" ]]; then
   ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi
if [[ -n "${OVER_SAMPLING_BATCH_SIZE:-}" ]]; then
   ROLLOUT_ARGS+=(--over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}")
fi
if [[ "${PARTIAL_ROLLOUT:-0}" == "1" ]]; then
   ROLLOUT_ARGS+=(--partial-rollout)
fi

# ── Eval ──────────────────────────────────────────────────────────────────
EVAL_ARGS=(
   --eval-prompt-data frontiercs "${BASE_FOLDER}/data/frontiercs/val.jsonl"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 81920
   --eval-top-p 1.0
)

# ── PERF ──────────────────────────────────────────────────────────────────
# 8 nodes × 8 H200 = 64 GPU total
# TP=4, PP=2, CP=4 → 32 GPU per training replica → DP=2
# rollout: 64 GPU / 4 GPU per engine = 16 SGLang engines (TP=4 each)
# H200 141GB HBM3e: max-tokens-per-gpu bumped to 10000 vs 5000 on A100
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --decoder-last-pipeline-num-layers 30
   --context-parallel-size 4
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 10000
)

# ── GRPO ──────────────────────────────────────────────────────────────────
GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
   --eps-clip 0.2
)

# ── Optimizer ─────────────────────────────────────────────────────────────
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
if [[ -n "${WANDB_ENTITY:-}" ]]; then
   WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

# ── SGLang ────────────────────────────────────────────────────────────────
# debug: cuda-graph and custom-all-reduce disabled for stability
# H200: mem-fraction bumped to 0.80 (141GB vs 80GB on A100)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.80}"
)
if [[ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]]; then
   SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
fi
if [[ -n "${SGLANG_SERVER_CONCURRENCY:-}" ]]; then
   SGLANG_ARGS+=(--sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}")
fi

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
if [[ "${CHECK_WEIGHT_UPDATE_EQUAL:-0}" == "1" ]]; then
   MISC_ARGS+=(--check-weight-update-equal)
fi

# ── Debug ─────────────────────────────────────────────────────────────────
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
if [[ "${DEBUG_ROLLOUT_ONLY:-0}" == "1" ]]; then
   DEBUG_ARGS+=(--debug-rollout-only)
fi

# ── Ray cluster startup ────────────────────────────────────────────────────
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

HOSTFILE="${HOSTFILE:-}"
if [ -n "${HOSTFILE}" ]; then
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then
      continue
    fi
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh root@"${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; \
       ray start --address=${MASTER_ADDR}:6379 \
         --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} \
         --node-ip-address ${WORKER_IP} \
         --disable-usage-stats" &
  done
  wait
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM/:/root/slime:/root/slime/FrontierSmith",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "FRONTIER_JUDGE_URL": "${FRONTIER_JUDGE_URL:-http://localhost:8082}",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false"
  }
}
EOF_JSON
)

JOB_SUBMIT_OUT=$(ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --no-wait \
   -- python3 "${SLIME_DIR}/train.py" \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
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

# Follow logs; retry if WebSocket drops (1006).
while true; do
    ray job logs --follow "$JOB_ID" 2>/dev/null || true
    STATUS=$(ray job status "$JOB_ID" 2>/dev/null | grep -oP "Status:\s*\K\w+")
    echo "Job status: ${STATUS}"
    if [[ "$STATUS" == "SUCCEEDED" || "$STATUS" == "FAILED" || "$STATUS" == "STOPPED" ]]; then
        echo "Job ended with status: ${STATUS}"
        [[ "$STATUS" == "SUCCEEDED" ]] && exit 0 || exit 1
    fi
    echo "Log stream dropped, retrying in 10s..."
    sleep 10
done
