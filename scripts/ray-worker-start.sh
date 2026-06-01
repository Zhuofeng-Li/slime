#!/bin/bash
# Usage: bash scripts/ray-worker-start.sh <MASTER_ADDR>
# Run on each worker node (not master) to join the Ray cluster.
# Called via kubectl exec from master node (no SSH needed).

MASTER_ADDR="${1:?MASTER_ADDR required}"
NUM_GPUS="${NUM_GPUS:-8}"

pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 2

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="127.0.0.1,${MASTER_ADDR}"

WORKER_IP=$(hostname -I | awk '{print $1}')
echo "Starting Ray worker: IP=${WORKER_IP}, MASTER=${MASTER_ADDR}, GPUs=${NUM_GPUS}"

ray start \
  --address="${MASTER_ADDR}:6379" \
  --num-gpus "${NUM_GPUS}" \
  --node-ip-address "${WORKER_IP}" \
  --disable-usage-stats

echo "Ray worker started on ${WORKER_IP}"
