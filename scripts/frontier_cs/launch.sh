#!/bin/bash
# Usage: bash scripts/frontier_cs/launch.sh [--base-dir DIR] [--num-gpus N] [--model MODEL] [--script SCRIPT] [--run-name NAME] [--skip-data] [--skip-judge] [--skip-model]
#        bash scripts/frontier_cs/launch.sh --base-dir /fsx-alignment/home/zhuofeng/slime --num-gpus 8 | tee logs/$(date +%Y%m%d-%H%M%S).log
#        bash scripts/frontier_cs/launch.sh --model qwen3.6 --skip-data --skip-judge | tee logs/frontiercs/run-qwen36-36B-$(date +%Y%m%d-%H%M%S).log
#        bash scripts/frontier_cs/launch.sh --model qwen3.5 --skip-data --skip-judge | tee logs/frontiercs/run-qwen35-36B-$(date +%Y%m%d-%H%M%S).log
#        bash scripts/frontier_cs/launch.sh --model qwen3.5-9B --skip-data --skip-judge
#        bash scripts/frontier_cs/launch.sh --model qwen3.5-27B --skip-data --skip-judge
#        bash scripts/frontier_cs/launch.sh --script scripts/frontier_cs/run-frontiercs-qwen3.5-35B-A3B-debug.sh --skip-data --skip-judge
# One-click launcher for FrontierCS:
#   1. Prepare data (JSONL from FrontierSmith problems)
#   2. Build & start Frontier-CS judge (Docker, port 8082)
#   3. Download HF model weights if missing
#   4. Convert HF weights to Megatron torch_dist if missing
#   5. Launch SLIME training via scripts/launch.sh
# sudo kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -u)

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
FRONTIER_DIR="${SLIME_DIR}/FrontierSmith/Frontier-CS/algorithmic"

read_zshrc_var() {
  local name="$1"
  if command -v zsh >/dev/null 2>&1 && [[ -r "${HOME}/.zshrc" ]]; then
    zsh -fc "source ~/.zshrc >/dev/null 2>&1; print -r -- \${${name}:-}"
  fi
}

[[ -z "${HF_TOKEN:-}" ]] && HF_TOKEN="$(read_zshrc_var HF_TOKEN)"
[[ -z "${WANDB_KEY:-}" ]] && WANDB_KEY="$(read_zshrc_var WANDB_KEY)"
[[ -z "${GPU_IDS:-}" ]] && GPU_IDS="$(read_zshrc_var GPU_IDS)"
if [[ -z "${WANDB_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then
  WANDB_MODE="offline"
fi

# ── defaults ──────────────────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-${SLIME_DIR}}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_IDS="${GPU_IDS:-}"
MODEL="${MODEL:-qwen3.6}"
RUN_SCRIPT="${RUN_SCRIPT:-}"
RUN_NAME="${RUN_NAME:-}"
CKPT_DIR="${CKPT_DIR:-}"
ROLLOUT_DIR="${ROLLOUT_DIR:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
IMAGE="${IMAGE:-slimerl/slime:latest}"
MODEL_NAME="${MODEL_NAME:-}"
HF_REPO="${HF_REPO:-}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
SKIP_MODEL="${SKIP_MODEL:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"

# ── arg parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)   BASE_DIR="$2";  shift 2 ;;
    --num-gpus)   NUM_GPUS="$2";  shift 2 ;;
    --gpu-ids)    GPU_IDS="$2";   shift 2 ;;
    --model)      MODEL="$2";     shift 2 ;;
    --script)     RUN_SCRIPT="$2"; shift 2 ;;
    --run-name)   RUN_NAME="$2";  shift 2 ;;
    --ckpt-dir)   CKPT_DIR="$2";  shift 2 ;;
    --rollout-dir) ROLLOUT_DIR="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --image)      IMAGE="$2";     shift 2 ;;
    --skip-data)  SKIP_DATA=1;    shift   ;;
    --skip-judge) SKIP_JUDGE=1;   shift   ;;
    --skip-model) SKIP_MODEL=1; SKIP_DOWNLOAD=1; SKIP_CONVERT=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --skip-convert) SKIP_CONVERT=1; shift ;;
    *) echo "Unknown option: $1"; exit 1  ;;
  esac
done

# ── resolve training script ────────────────────────────────────────────────
if [[ -z "${RUN_SCRIPT}" ]]; then
  case "${MODEL}" in
    qwen3.5)
      RUN_SCRIPT="scripts/frontier_cs/run-frontiercs-qwen3.5-35B-A3B.sh"
      MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
      HF_REPO="${HF_REPO:-Qwen/Qwen3.5-35B-A3B}"
      CONVERT_SCRIPT="${CONVERT_SCRIPT:-scripts/convert_qwen35_35B_A3B_torch_dist.sh}"
      ;;
    qwen3.5-9B)
      RUN_SCRIPT="scripts/frontier_cs/run-frontiercs-qwen3.5-9B-debug.sh"
      MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
      HF_REPO="${HF_REPO:-Qwen/Qwen3.5-9B}"
      CONVERT_SCRIPT="${CONVERT_SCRIPT:-scripts/convert_qwen35_9B_torch_dist.sh}"
      ;;
    qwen3.5-27B)
      RUN_SCRIPT="scripts/frontier_cs/run-frontiercs-qwen3.5-27B-debug.sh"
      MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B}"
      HF_REPO="${HF_REPO:-Qwen/Qwen3.5-27B}"
      CONVERT_SCRIPT="${CONVERT_SCRIPT:-scripts/convert_qwen36_27B_torch_dist.sh}"
      ;;
    qwen3.6)
      RUN_SCRIPT="scripts/frontier_cs/run-frontiercs-qwen3.6-35B-A3B.sh"
      MODEL_NAME="${MODEL_NAME:-Qwen3.6-35B-A3B}"
      HF_REPO="${HF_REPO:-Qwen/Qwen3.6-35B-A3B}"
      CONVERT_SCRIPT="${CONVERT_SCRIPT:-scripts/convert_qwen36_35B_A3B_torch_dist.sh}"
      ;;
    *) echo "Unknown --model '${MODEL}'. Choose: qwen3.5, qwen3.5-9B, qwen3.5-27B, qwen3.6"; exit 1 ;;
  esac
else
  MODEL="custom"
fi

DATA_DIR="${BASE_DIR}/data/frontiercs"
MODEL_DIR="${BASE_DIR}/models/${MODEL_NAME}"
TORCH_DIST_DIR="${BASE_DIR}/models/${MODEL_NAME}_torch_dist"

if [[ "${SKIP_MODEL}" -eq 0 && "${MODEL}" == "custom" ]]; then
  if [[ -z "${MODEL_NAME}" || -z "${HF_REPO}" || -z "${CONVERT_SCRIPT}" ]]; then
    echo "ERROR: --script custom mode needs MODEL_NAME, HF_REPO, and CONVERT_SCRIPT env vars, or pass --skip-model."
    exit 1
  fi
fi

docker_env_args() {
  [[ -n "${HF_TOKEN:-}" ]] && printf '%s\0%s\0' -e "HF_TOKEN=${HF_TOKEN}"
  [[ -n "${WANDB_KEY:-}" ]] && printf '%s\0%s\0' -e "WANDB_KEY=${WANDB_KEY}"
  [[ -n "${WANDB_MODE:-}" ]] && printf '%s\0%s\0' -e "WANDB_MODE=${WANDB_MODE}"
  [[ -n "${GPU_IDS:-}" ]] && printf '%s\0%s\0' -e "CUDA_VISIBLE_DEVICES=${GPU_IDS}"
  [[ -n "${MODEL_NAME:-}" ]] && printf '%s\0%s\0' -e "MODEL_NAME=${MODEL_NAME}"
  [[ -n "${HF_REPO:-}" ]] && printf '%s\0%s\0' -e "HF_REPO=${HF_REPO}"
  [[ -n "${HF_CKPT:-}" ]] && printf '%s\0%s\0' -e "HF_CKPT=${HF_CKPT}"
  [[ -n "${SAVE_DIR:-}" ]] && printf '%s\0%s\0' -e "SAVE_DIR=${SAVE_DIR}"
}

run_in_slime_container() {
  local -a env_args=()
  while IFS= read -r -d '' item; do
    env_args+=("${item}")
  done < <(docker_env_args)

  sudo docker run --rm --gpus all \
    --shm-size=64g \
    --network=host \
    -v "${BASE_DIR}:/root" \
    -v "${SLIME_DIR}:/root/slime" \
    -w /root/slime \
    -e "BASE_FOLDER=/root" \
    -e "NUM_GPUS=${NUM_GPUS}" \
    "${env_args[@]}" \
    "${IMAGE}" \
    "$@"
}

run_prepare_data() {
  if command -v python3 >/dev/null 2>&1; then
    python3 "${SCRIPT_DIR}/prepare_frontiercs_jsonl.py" "$@"
  elif command -v python >/dev/null 2>&1; then
    python "${SCRIPT_DIR}/prepare_frontiercs_jsonl.py" "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv run python "${SCRIPT_DIR}/prepare_frontiercs_jsonl.py" "$@"
  else
    echo "  No host python found; running data preparation inside ${IMAGE}..."
    sudo docker pull "${IMAGE}"
    run_in_slime_container python3 /root/slime/scripts/frontier_cs/prepare_frontiercs_jsonl.py \
      --output-dir /root/data/frontiercs \
      --full-for-both
  fi
}

echo "============================================================"
echo "  FrontierCS One-Click Launcher"
echo "  BASE_DIR : ${BASE_DIR}"
echo "  DATA_DIR : ${DATA_DIR}"
echo "  IMAGE    : ${IMAGE}"
echo "  NUM_GPUS : ${NUM_GPUS}"
[[ -n "${GPU_IDS}" ]] && echo "  GPU_IDS  : ${GPU_IDS}"
echo "  MODEL    : ${MODEL}  (${RUN_SCRIPT})"
[[ -n "${MODEL_NAME}" ]] && echo "  MODEL_NAME: ${MODEL_NAME}"
[[ -n "${HF_REPO}" ]] && echo "  HF_REPO  : ${HF_REPO}"
[[ -n "${RUN_NAME}" ]] && echo "  RUN_NAME : ${RUN_NAME}"
[[ -n "${CKPT_DIR}" ]] && echo "  CKPT_DIR : ${CKPT_DIR}"
[[ -n "${ROLLOUT_DIR}" ]] && echo "  ROLLOUT_DIR: ${ROLLOUT_DIR}"
[[ -n "${SAVE_INTERVAL}" ]] && echo "  SAVE_INTERVAL: ${SAVE_INTERVAL}"
echo "============================================================"

# ── Step 1: Data preparation ──────────────────────────────────────────────
if [[ $SKIP_DATA -eq 0 ]]; then
  echo ""
  echo "[1/5] Preparing FrontierCS JSONL data..."
  run_prepare_data \
    --output-dir "${DATA_DIR}" \
    --full-for-both
  echo "[1/5] Data ready at ${DATA_DIR}"
else
  echo "[1/5] Skipping data preparation (--skip-data)"
fi

# ── Step 2: Judge ─────────────────────────────────────────────────────────
if [[ $SKIP_JUDGE -eq 0 ]]; then
  echo ""
  echo "[2/5] Starting Frontier-CS judge (port 8082)..."

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
  echo "[2/5] Judge running at http://localhost:8082"
else
  echo "[2/5] Skipping judge startup (--skip-judge)"
fi

# ── Step 3: Model download ────────────────────────────────────────────────
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
  echo ""
  echo "[3/5] Ensuring HF model weights exist at ${MODEL_DIR}..."
  if [[ -f "${MODEL_DIR}/model.safetensors.index.json" || -f "${MODEL_DIR}/config.json" && -n "$(find "${MODEL_DIR}" -maxdepth 1 -name '*.safetensors' -print -quit 2>/dev/null)" ]]; then
    echo "  HF model already present; skipping download."
  else
    mkdir -p "${MODEL_DIR}"
    echo "  Pulling Docker image ${IMAGE} for model setup..."
    sudo docker pull "${IMAGE}"
    echo "  Downloading ${HF_REPO} ..."
    run_in_slime_container bash -lc '
      set -euo pipefail
      if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "${HF_REPO}" --local-dir "/root/models/${MODEL_NAME}"
      elif python3 -c "import huggingface_hub" >/dev/null 2>&1; then
        python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${HF_REPO}", local_dir="/root/models/${MODEL_NAME}", local_dir_use_symlinks=False)
PY
      else
        echo "ERROR: neither huggingface-cli nor huggingface_hub is available in ${IMAGE}."
        exit 1
      fi
    '
  fi
  echo "[3/5] HF model ready at ${MODEL_DIR}"
else
  echo "[3/5] Skipping model download (--skip-download/--skip-model)"
fi

# ── Step 4: torch_dist conversion ─────────────────────────────────────────
if [[ $SKIP_CONVERT -eq 0 ]]; then
  echo ""
  echo "[4/5] Ensuring torch_dist weights exist at ${TORCH_DIST_DIR}..."
  if [[ -d "${TORCH_DIST_DIR}" && -n "$(find "${TORCH_DIST_DIR}" -type f -print -quit 2>/dev/null)" ]]; then
    echo "  torch_dist model already present; skipping conversion."
  else
    echo "  Pulling Docker image ${IMAGE} for conversion..."
    sudo docker pull "${IMAGE}"
    MODEL_NAME="${MODEL_NAME}" HF_CKPT="/root/models/${MODEL_NAME}" SAVE_DIR="/root/models/${MODEL_NAME}_torch_dist" \
      run_in_slime_container bash "${CONVERT_SCRIPT}"
  fi
  echo "[4/5] torch_dist model ready at ${TORCH_DIST_DIR}"
else
  echo "[4/5] Skipping torch_dist conversion (--skip-convert/--skip-model)"
fi

# ── Step 5: Training ──────────────────────────────────────────────────────
echo ""
echo "[5/5] Launching SLIME training..."
[[ -n "${RUN_NAME}" ]] && export RUN_NAME
[[ -n "${CKPT_DIR}" ]] && export CKPT_DIR
[[ -n "${ROLLOUT_DIR}" ]] && export ROLLOUT_DIR
[[ -n "${SAVE_INTERVAL}" ]] && export SAVE_INTERVAL
[[ -n "${MODEL_NAME}" ]] && export MODEL_NAME
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN
[[ -n "${WANDB_KEY:-}" ]] && export WANDB_KEY
[[ -n "${WANDB_MODE:-}" ]] && export WANDB_MODE
[[ -n "${GPU_IDS}" ]] && export GPU_IDS
for name in APPLY_CHAT_TEMPLATE_KWARGS ROLLOUT_TEMPERATURE ROLLOUT_TOP_P ROLLOUT_TOP_K NUM_ROLLOUT ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT ROLLOUT_MAX_RESPONSE_LEN GLOBAL_BATCH_SIZE SGLANG_MEM_FRACTION_STATIC SGLANG_MAX_RUNNING_REQUESTS SGLANG_SERVER_CONCURRENCY DYNAMIC_SAMPLING_FILTER_PATH OVER_SAMPLING_BATCH_SIZE PARTIAL_ROLLOUT; do
  value="${!name:-}"
  [[ -n "${value}" ]] && export "${name}=${value}"
done
bash "${SLIME_DIR}/scripts/launch.sh" \
  --base-dir "${BASE_DIR}" \
  --num-gpus "${NUM_GPUS}" \
  --image "${IMAGE}" \
  --script "${RUN_SCRIPT}"
