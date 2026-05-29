# debug
RUN_NAME=qwen3.5-35B-A3B-frontiercs-debug-n32-use-tis

bash scripts/frontier_cs/launch.sh \
  --script scripts/frontier_cs/run-frontiercs-qwen3.5-35B-A3B-debug.sh \
  --run-name "${RUN_NAME}" \
  --skip-data \
  --skip-judge | tee "logs/frontiercs/${RUN_NAME}-$(date +%Y%m%d-%H%M%S).log"

RUN_NAME=qwen3.5-27B-frontiercs-debug-n32
bash scripts/frontier_cs/launch.sh \
    --base-dir /fsx-alignment/home/zhuofeng/slime \
    --num-gpus 8 \
    --script scripts/frontier_cs/run-frontiercs-qwen3.5-27B-debug.sh \
    --skip-data \
    --skip-judge | tee "logs/frontiercs/${RUN_NAME}-$(date +%Y%m%d-%H%M%S).log"

