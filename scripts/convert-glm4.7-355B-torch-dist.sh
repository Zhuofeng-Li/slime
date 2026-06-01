#!/bin/bash
# Usage: bash scripts/convert-glm4.7-355B-torch-dist.sh
# Run on worker-0. Uses 2 nodes x 8 GPUs via kubectl exec for node-1.
# Converts GLM-4.7-355B HF checkpoint to torch_dist format for slime training.
#
# Example:
#   kubectl exec xliucr-slime-8n-may27-worker-0 -n application-nonprod -- \
#     bash /shared/dev/zhuofeng/slime/scripts/convert-glm4.7-355B-torch-dist.sh

set -ex

BASE_DIR="${BASE_DIR:-/shared/dev/zhuofeng/slime}"
SLIME_DIR="${SLIME_DIR:-/shared/dev/zhuofeng/slime}"
NAMESPACE="${NAMESPACE:-application-nonprod}"
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=12345

echo "MASTER_ADDR: ${MASTER_ADDR}"

# Install slime (no Docker, pip only)
cd "${SLIME_DIR}"
pip install -e . --no-deps -q

source "${SLIME_DIR}/scripts/models/glm4.5-355B-A32B.sh"

# Start node-1 torchrun via kubectl exec (no SSH)
WORKER_POD="xliucr-slime-8n-may27-worker-1"
echo "Starting torchrun on ${WORKER_POD} (NODE_RANK=1)..."
kubectl exec "${WORKER_POD}" -n "${NAMESPACE}" -- bash -c "
  cd ${SLIME_DIR}
  pip install -e . --no-deps -q 2>/dev/null
  source ${SLIME_DIR}/scripts/models/glm4.5-355B-A32B.sh
  PYTHONPATH=/root/Megatron-LM/ torchrun \
    --nproc-per-node 8 \
    --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} \
    --nnodes=2 --node-rank 1 \
    ${SLIME_DIR}/tools/convert_hf_to_torch_dist.py \
    \${MODEL_ARGS[@]} \
    --hf-checkpoint ${BASE_DIR}/GLM-4.7-355B-A32B/ \
    --save ${BASE_DIR}/GLM-4.7-355B-A32B_torch_dist/
" > "${BASE_DIR}/logs/convert_node1.log" 2>&1 &
NODE1_PID=$!
echo "Node-1 torchrun PID: ${NODE1_PID}"

sleep 5

# Run node-0 (master, NODE_RANK=0)
PYTHONPATH=/root/Megatron-LM/ torchrun \
  --nproc-per-node 8 \
  --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT}" \
  --nnodes=2 --node-rank 0 \
  "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${BASE_DIR}/GLM-4.7-355B-A32B/" \
  --save "${BASE_DIR}/GLM-4.7-355B-A32B_torch_dist/"

wait $NODE1_PID
echo "Conversion complete: ${BASE_DIR}/GLM-4.7-355B-A32B_torch_dist/"
