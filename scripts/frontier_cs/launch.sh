#!/bin/bash
# Usage: bash scripts/frontier_cs/launch.sh [--base-dir DIR] [--num-gpus N] [--skip-data] [--skip-judge]
#
# One-click launcher for FrontierCS + Qwen3.5-35B-A3B:
#   1. Prepare data (JSONL from FrontierSmith problems)
#   2. Build & start Frontier-CS judge (Docker, port 8082)
#   3. Launch SLIME training via scripts/launch.sh

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
FRONTIER_DIR="${SLIME_DIR}/FrontierSmith/Frontier-CS/algorithmic"

# ── defaults ──────────────────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-${SLIME_DIR}}"
NUM_GPUS="${NUM_GPUS:-8}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"

# ── arg parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)   BASE_DIR="$2";  shift 2 ;;
    --num-gpus)   NUM_GPUS="$2";  shift 2 ;;
    --skip-data)  SKIP_DATA=1;    shift   ;;
    --skip-judge) SKIP_JUDGE=1;   shift   ;;
    *) echo "Unknown option: $1"; exit 1  ;;
  esac
done

DATA_DIR="${BASE_DIR}/data/frontiercs"

echo "============================================================"
echo "  FrontierCS One-Click Launcher"
echo "  BASE_DIR : ${BASE_DIR}"
echo "  DATA_DIR : ${DATA_DIR}"
echo "  NUM_GPUS : ${NUM_GPUS}"
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

  # Build image if not present
  if ! docker image inspect frontiercs-judge >/dev/null 2>&1; then
    echo "  Building frontiercs-judge Docker image..."
    docker build -t frontiercs-judge "${FRONTIER_DIR}"
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
      echo "  ERROR: Judge did not become healthy after 60s. Check: docker logs Competitive-Programming"
      exit 1
    fi
    sleep 2
  done
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
  --script "scripts/frontier_cs/run-frontiercs-qwen3.5-35B-A3B.sh"
