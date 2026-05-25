#!/bin/bash
# Usage: bash scripts/launch.sh [--base-dir DIR] [--num-gpus N] [--skip-download] [--skip-convert] [--image IMAGE] [--script SCRIPT]
#
# One-click Docker launcher. Defaults to quickstart (GLM-Z1-9B).
# Use --script to run any other training script inside the same container.
# Run this after SSH-ing into a GPU compute node.
#
# Examples:
#   bash scripts/launch.sh --base-dir /fsx-alignment/home/zhuofeng/slime --num-gpus 8 --script scripts/run-qwen3.5-35B-A3B.sh
#   bash scripts/launch.sh --base-dir /fsx-alignment/home/zhuofeng/slime --script scripts/run-qwen3.5-35B-A3B.sh --skip-download --skip-convert

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

read_zshrc_var() {
  local name="$1"
  if command -v zsh >/dev/null 2>&1 && [[ -r "${HOME}/.zshrc" ]]; then
    zsh -fc "source ~/.zshrc >/dev/null 2>&1; print -r -- \${${name}:-}"
  fi
}

[[ -z "${HF_TOKEN:-}" ]] && HF_TOKEN="$(read_zshrc_var HF_TOKEN)"
[[ -z "${WANDB_KEY:-}" ]] && WANDB_KEY="$(read_zshrc_var WANDB_KEY)"
[[ -z "${GPU_IDS:-}" ]] && GPU_IDS="$(read_zshrc_var GPU_IDS)"

# ── defaults ─────────────────────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-${SLIME_DIR}}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-}"          # set in ~/.zshrc or override with --gpu-ids
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
IMAGE="${IMAGE:-slimerl/slime:latest}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/quickstart.sh}"   # override with --script

# ── arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)      BASE_DIR="$2";     shift 2 ;;
    --num-gpus)      NUM_GPUS="$2";     shift 2 ;;
    --image)         IMAGE="$2";        shift 2 ;;
    --gpu-ids)       GPU_IDS="$2";      shift 2 ;;
    --script)        RUN_SCRIPT="$2";   shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1;   shift   ;;
    --skip-convert)  SKIP_CONVERT=1;    shift   ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── build extra args ──────────────────────────────────────────────────────────
QUICKSTART_ARGS=(
  --base-dir /root
  --num-gpus "${NUM_GPUS}"
)
[[ $SKIP_DOWNLOAD -eq 1 ]] && QUICKSTART_ARGS+=(--skip-download)
[[ $SKIP_CONVERT  -eq 1 ]] && QUICKSTART_ARGS+=(--skip-convert)

echo "============================================================"
echo "  slime Docker Launcher"
echo "  Image    : ${IMAGE}"
echo "  Script   : ${RUN_SCRIPT}"
echo "  BASE_DIR : ${BASE_DIR}  (mounted as /data inside container)"
echo "  NUM_GPUS : ${NUM_GPUS}"
echo "  Slime src: ${SLIME_DIR}  (mounted as /root/slime)"
echo "============================================================"

# ── ensure data dir exists ────────────────────────────────────────────────────
mkdir -p "${BASE_DIR}"

# ── pull image ────────────────────────────────────────────────────────────────
echo "[0/3] Pulling image ${IMAGE} ..."
sudo docker pull "${IMAGE}"

# ── Secrets (read from environment, set these in ~/.zshrc) ────────────────────
[[ -z "${HF_TOKEN:-}"   ]] && echo "[launch.sh] WARNING: HF_TOKEN not set. Gated models/datasets will fail."
[[ -z "${WANDB_KEY:-}"  ]] && echo "[launch.sh] WARNING: WANDB_KEY not set. wandb logging will be unavailable."

SECRET_ARGS=()
[[ -n "${HF_TOKEN:-}" ]] && SECRET_ARGS+=(-e "HF_TOKEN=${HF_TOKEN}")
[[ -n "${WANDB_KEY:-}" ]] && SECRET_ARGS+=(-e "WANDB_KEY=${WANDB_KEY}")

# ── GPU selection ─────────────────────────────────────────────────────────────
if [[ -n "${GPU_IDS}" ]]; then
  GPUS_ARG=(--gpus all)
  GPU_ENV=(-e "CUDA_VISIBLE_DEVICES=${GPU_IDS}")
else
  GPUS_ARG=(--gpus all)
  GPU_ENV=()
fi

# ── kill stale Ray/sglang from previous Docker runs (--network=host leaks them) ──
echo "[pre-flight] Stopping any stale Ray/sglang processes on host..."
sudo pkill -9 -f "gcs_server"  2>/dev/null || true
sudo pkill -9 -f "raylet"      2>/dev/null || true
sudo pkill -9 -f "ray/_private" 2>/dev/null || true
sudo pkill -9 -f "ray/dashboard" 2>/dev/null || true
sudo pkill -9 -f "ray/autoscaler" 2>/dev/null || true
sudo pkill -9 -f "ray/util/client" 2>/dev/null || true
sudo pkill -9 -f "sglang"      2>/dev/null || true
sudo rm -rf /tmp/ray           2>/dev/null || true
sleep 3

# ── build script args ─────────────────────────────────────────────────────────
# quickstart.sh takes CLI flags; other scripts read BASE_FOLDER env var.
if [[ "${RUN_SCRIPT}" == "scripts/quickstart.sh" ]]; then
  SCRIPT_CMD=(bash "${RUN_SCRIPT}" "${QUICKSTART_ARGS[@]}")
else
  SCRIPT_CMD=(bash "${RUN_SCRIPT}")
fi

# ── run ───────────────────────────────────────────────────────────────────────
# Mounts:
#   BASE_DIR  → /data            (models, datasets, checkpoints)
#   SLIME_DIR → /root/slime      (your local dev code replaces the image's copy)
sudo docker run --rm "${GPUS_ARG[@]}" \
  --shm-size=64g \
  --network=host \
  -v "${BASE_DIR}:/root" \
  -v "${SLIME_DIR}:/root/slime" \
  -w /root/slime \
  -e "BASE_FOLDER=/root" \
  -e "MASTER_ADDR=127.0.0.1" \
  "${GPU_ENV[@]}" \
  "${SECRET_ARGS[@]}" \
  "${IMAGE}" \
  "${SCRIPT_CMD[@]}"
