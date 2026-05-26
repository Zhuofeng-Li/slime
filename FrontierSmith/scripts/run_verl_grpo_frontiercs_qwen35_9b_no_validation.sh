#!/bin/bash
# GRPO training for Qwen3.5-9B on Frontier-CS (VERL) — NO VALIDATION VERSION
#
# VERL still constructs a validation dataset even when trainer.test_freq=0.
# This script therefore provides a 1-row dummy validation parquet with the
# same schema as TRAIN_DATA, and generates it automatically if missing.

set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Environment ──────────────────────────────────────────────────────────────
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "ERROR: .venv not found. Run 'source setup-env.sh' first."
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$PROJECT_ROOT/.cache/vllm}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}

TRAIN_DATA=${TRAIN_DATA:-$PROJECT_ROOT/data/frontiercs/train.parquet}
VAL_DATA=${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/dummy_val.parquet}

TP=${TP:-2}
NGPU=${NGPU:-8}

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Run: python scripts/prepare_frontiercs_parquet.py"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    mkdir -p "$(dirname "$VAL_DATA")"
    python - "$TRAIN_DATA" "$VAL_DATA" <<'PY'
import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

df = pd.read_parquet(src)
if len(df) == 0:
    raise SystemExit(f"ERROR: {src} is empty, cannot build dummy val parquet")

df.iloc[[0]].copy().to_parquet(dst, index=False)
print(f"[dummy-val] Created {dst} from {src} with columns {df.columns.tolist()}")
PY
fi

TRAIN_FILES="['$TRAIN_DATA']"
VAL_FILES="['$VAL_DATA']"

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ── Checkpoints & outputs ───────────────────────────────────────────────────
CKPT_DIR="$PROJECT_ROOT/checkpoints/verl_frontiercs_qwen35_9b/qwen35_9b_grpo_no_val"
ROLLOUT_DIR="$PROJECT_ROOT/outputs/rollout_data_no_val"
mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR"

FRESH_START_ARGS=""
if [ "${FRESH_START:-0}" = "1" ]; then
    if [ -d "$CKPT_DIR" ]; then
        BACKUP_DIR="${CKPT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
        echo "[FRESH_START] Backing up checkpoints: $CKPT_DIR -> $BACKUP_DIR"
        mv "$CKPT_DIR" "$BACKUP_DIR"
        echo "[FRESH_START] Backup done."
    fi
    FRESH_START_ARGS="trainer.resume_mode=disable"
    echo "[FRESH_START] Training will start from scratch."
fi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=16000 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_frontiercs_qwen35_9b \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.rollout_data_dir=$ROLLOUT_DIR \
    trainer.experiment_name=qwen35_9b_grpo_no_val \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.total_epochs=60 \
    trainer.save_freq=10 \
    trainer.test_freq=0 \
    $FRESH_START_ARGS \
    "$@"
