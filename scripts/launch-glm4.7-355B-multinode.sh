#!/bin/bash
# Usage: bash scripts/launch-glm4.7-355B-multinode.sh
# Run on worker-0 (master node) inside the compute pod.
# Starts ray head, then triggers ray workers on other pods via kubectl exec.
# No SSH required — uses kubectl exec instead.
#
# Prerequisites:
#   - Model at $BASE_DIR/GLM-4.7-355B-A32B/
#   - torch_dist ref at $BASE_DIR/GLM-4.7-355B-A32B_torch_dist/
#   - Datasets at $BASE_DIR/dapo-math-17k/ and $BASE_DIR/rl_data/
#
# Example:
#   kubectl exec xliucr-slime-8n-may27-worker-0 -n application-nonprod -- \
#     bash /shared/dev/zhuofeng/slime/scripts/launch-glm4.7-355B-multinode.sh

set -ex

BASE_DIR="${BASE_DIR:-/shared/dev/zhuofeng/slime}"
SLIME_DIR="${SLIME_DIR:-/shared/dev/zhuofeng/slime}"
NAMESPACE="${NAMESPACE:-application-nonprod}"
CLUSTER_PREFIX="${CLUSTER_PREFIX:-xliucr-slime-8n-may27}"
ACTOR_NUM_NODES=8
ACTOR_NUM_GPUS_PER_NODE=8

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

MASTER_ADDR=$(hostname -I | awk '{print $1}')
echo "MASTER_ADDR: ${MASTER_ADDR}"

# ── Stop stale processes ───────────────────────────────────────────────────────
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3

# ── Install slime from shared dir (no Docker, use pip) ────────────────────────
cd "${SLIME_DIR}"
pip install -e . --no-deps -q

# ── Start Ray head ────────────────────────────────────────────────────────────
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

sleep 3

# ── Start Ray workers on other pods via kubectl exec (no SSH) ─────────────────
for i in $(seq 1 $((ACTOR_NUM_NODES - 1))); do
  POD="${CLUSTER_PREFIX}-worker-${i}"
  echo "Starting ray worker on ${POD} ..."
  kubectl exec "${POD}" -n "${NAMESPACE}" -- bash -c "
    pkill -9 sglang 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 2
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    export no_proxy='127.0.0.1,${MASTER_ADDR}'
    WORKER_IP=\$(hostname -I | awk '{print \$1}')
    ray start --address='${MASTER_ADDR}:6379' \
      --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} \
      --node-ip-address \"\${WORKER_IP}\" \
      --disable-usage-stats
    echo \"Worker \${WORKER_IP} joined\"
  " &
done
wait
echo "All ray workers started. Waiting 10s for cluster to stabilize..."
sleep 10
ray status

# ── Detect NVLink ─────────────────────────────────────────────────────────────
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=0
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1
echo "HAS_NVLINK: ${HAS_NVLINK}"

# ── Source model args ─────────────────────────────────────────────────────────
source "${SLIME_DIR}/scripts/models/glm4.5-355B-A32B.sh"

# ── Training args (from run-glm4.7-355B-A32B.sh) ─────────────────────────────
CKPT_ARGS=(
   --hf-checkpoint "${BASE_DIR}/GLM-4.7-355B-A32B"
   --ref-load "${BASE_DIR}/GLM-4.7-355B-A32B_torch_dist/"
)

ROLLOUT_ARGS=(
   --prompt-data "${BASE_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 1
   --over-sampling-batch-size 256
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --num-steps-per-rollout 4
   --balance-data
   --rollout-stop-token-ids 151329 151336 151338
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "${BASE_DIR}/rl_data/aime-2024.jsonl"
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 32768
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 2
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 1e-4
   --eps-clip-high 2e-4
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 32
   --sglang-mem-fraction-static 0.7
   --sglang-enable-dp-attention
   --sglang-dp-size 4
   --sglang-ep-size 32
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
)

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "PYTHONBUFFERED": "16"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
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
   "${MISC_ARGS[@]}"
