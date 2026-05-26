#!/bin/bash
# Frontier-CS-synthetic: one-shot environment setup and activation.
#
# Usage:
#   source setup-env.sh           # create .venv (if missing), install deps, activate
#   source setup-env.sh --skip    # activate only, skip install checks
#
# Everything stays inside this project directory:
#   .venv/          — Python virtual environment
#   .cache/         — HuggingFace, vLLM, Ray caches
#   checkpoints/    — training checkpoints
#   outputs/        — rollout data and logs

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
SKIP_INSTALL=false
[[ "${1:-}" == "--skip" ]] && SKIP_INSTALL=true

# ── 1. Create venv if missing ───────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# ── 2. Activate ─────────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── 3. Install / upgrade deps ───────────────────────────────────────────────
if [ "$SKIP_INSTALL" = false ]; then
    # Marker to avoid re-running on every source
    MARKER="$VENV_DIR/.deps-installed"
    if [ ! -f "$MARKER" ]; then
        echo "[setup] Installing dependencies (first time) ..."
        pip install --upgrade pip

        # Core: vllm pulls torch + CUDA wheels automatically
        pip install "vllm>=0.19,<1"

        # verl training deps (skip verl's numpy<2 pin — vllm needs numpy>=2)
        pip install accelerate codetiming datasets dill hydra-core pandas peft \
            pylatexenc "ray[default]>=2.41.0" torchdata \
            "tensordict>=0.8.0,<=0.10.0,!=0.9.0" "transformers>=5.2.0" \
            wandb tensorboard pybind11 math-verify liger-kernel \
            latex2sympy2_extended fastapi uvicorn "pyarrow>=19.0.0"

        # verl itself (editable, no-deps to avoid numpy<2 conflict)
        pip install --no-deps -e "$PROJECT_ROOT/verl"

        # ALE-Bench validator
        pip install -e "$PROJECT_ROOT/ALE-Bench"

        # Test utilities
        pip install pytest pytest-asyncio pytest-rerunfailures py-spy pre-commit

        # Project-level deps
        pip install -r "$PROJECT_ROOT/requirements.txt"

        touch "$MARKER"
        echo "[setup] Dependencies installed."
    else
        echo "[setup] Dependencies already installed (rm $MARKER to force reinstall)."
    fi
fi

# ── 4. Environment variables ────────────────────────────────────────────────
export PYTHONPATH="$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export FRONTIER_CS_PROBLEMS_DIR="$PROJECT_ROOT/Frontier-CS/algorithmic/problems"

# All caches in project dir (override with env vars if needed)
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$PROJECT_ROOT/.cache/vllm}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
export ALE_BENCH_CACHE_DIR="${ALE_BENCH_CACHE_DIR:-$PROJECT_ROOT/.cache/ale-bench}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_DIR" "$RAY_TMPDIR" "$ALE_BENCH_CACHE_DIR"

# ── 5. Summary ──────────────────────────────────────────────────────────────
echo "[setup] Environment ready."
echo "  python : $(which python)"
echo "  torch  : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "  vllm   : $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'not installed')"
echo "  GPUs   : $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'N/A')"
