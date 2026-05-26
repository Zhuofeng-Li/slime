#!/bin/bash
# Usage: bash scripts/frontier_cs/launch.sh [--base-dir DIR] [--num-gpus N] [--model MODEL] [--skip-data] [--skip-judge]
#        bash scripts/frontier_cs/launch.sh --base-dir /fsx-alignment/home/zhuofeng/slime --num-gpus 8 | tee logs/$(date +%Y%m%d-%H%M%S).log
#        bash scripts/frontier_cs/launch.sh --model qwen3.6 --skip-data --skip-judge | tee logs/frontiercs/run-qwen36-36B-$(date +%Y%m%d-%H%M%S).log 
# One-click launcher for FrontierCS:
#   1. Prepare data (JSONL from FrontierSmith problems)
#   2. Build & start Frontier-CS judge (Docker, port 8082)
#   3. Launch SLIME training via scripts/launch.sh
# sudo kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -u)

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
FRONTIER_DIR="${SLIME_DIR}/FrontierSmith/Frontier-CS/algorithmic"

# ── defaults ──────────────────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-${SLIME_DIR}}"
NUM_GPUS="${NUM_GPUS:-8}"
MODEL="${MODEL:-qwen3.6}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"

# ── arg parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)   BASE_DIR="$2";  shift 2 ;;
    --num-gpus)   NUM_GPUS="$2";  shift 2 ;;
    --model)      MODEL="$2";     shift 2 ;;
    --skip-data)  SKIP_DATA=1;    shift   ;;
    --skip-judge) SKIP_JUDGE=1;   shift   ;;
    *) echo "Unknown option: $1"; exit 1  ;;
  esac
done

# ── resolve training script from model name ────────────────────────────────
case "${MODEL}" in
  qwen3.5) RUN_SCRIPT="scripts/frontier_cs/run-frontiercs-qwen3.5-35B-A3B.sh" ;;
  qwen3.6) RUN_SCRIPT="scripts/frontier_cs/run-frontiercs-qwen3.6-35B-A3B.sh" ;;
  *) echo "Unknown --model '${MODEL}'. Choose: qwen3.5, qwen3.6"; exit 1 ;;
esac

DATA_DIR="${BASE_DIR}/data/frontiercs"

echo "============================================================"
echo "  FrontierCS One-Click Launcher"
echo "  BASE_DIR : ${BASE_DIR}"
echo "  DATA_DIR : ${DATA_DIR}"
echo "  NUM_GPUS : ${NUM_GPUS}"
echo "  MODEL    : ${MODEL}  (${RUN_SCRIPT})"
echo "============================================================"

# ── Step 1: Data preparation ──────────────────────────────────────────────
if [[ $SKIP_DATA -eq 0 ]]; then
  echo ""
  echo "[1/3] Preparing FrontierCS JSONL data..."
  uv run python "${SCRIPT_DIR}/prepare_frontiercs_jsonl.py" \
    --output-dir "${DATA_DIR}" \
    --full-for-both
  echo "[1/3] Data ready at ${DATA_DIR}"
else
  echo "[1/3] Skipping data preparation (--skip-data)"
fi

# ── Step 2: Judge ─────────────────────────────────────────────────────────
if [[ $SKIP_JUDGE -eq 0 ]]; then
  echo ""
  echo "[2/3] Starting Frontier-CS judge (port 8082)..."

  # If judge is already healthy, skip docker operations entirely
  if curl -sf http://localhost:8082/health >/dev/null 2>&1; then
    echo "  Judge already healthy at http://localhost:8082 — skipping Docker build/start."
  else
    # Build image if not present (use sudo docker)
    if ! sudo docker image inspect frontiercs-judge >/dev/null 2>&1; then
      echo "  Building frontiercs-judge Docker image..."
      sudo docker build -t frontiercs-judge "${FRONTIER_DIR}"
    fi

    # Start judge (run_judge.sh already handles remove + restart)
    bash "${FRONTIER_DIR}/run_judge.sh"

    # Wait for judge to be healthy
    echo "  Waiting for judge at http://localhost:8082/health ..."
    for i in $(seq 1 30); do
      if curl -sf http://localhost:8082/health >/dev/null 2>&1; then
        echo "  Judge is healthy."
        break
      fi
      if [[ $i -eq 30 ]]; then
        echo "  ERROR: Judge did not become healthy after 60s. Check: sudo docker logs Competitive-Programming"
        exit 1
      fi
      sleep 2
    done
  fi
  echo "[2/3] Judge running at http://localhost:8082"
else
  echo "[2/3] Skipping judge startup (--skip-judge)"
fi

# ── Step 3: Training ──────────────────────────────────────────────────────
echo ""
echo "[3/3] Launching SLIME training..."
bash "${SLIME_DIR}/scripts/launch.sh" \
  --base-dir "${BASE_DIR}" \
  --num-gpus "${NUM_GPUS}" \
  --skip-download \
  --skip-convert \
  --script "${RUN_SCRIPT}"
