#!/bin/bash
# GRPO on Qwen3.5-9B with random rewards only (no Frontier-CS or other task data).
#
# VERL still requires a parquet with prompts; this script materializes a tiny
# placeholder parquet (repeated dummy prompts). Rewards come entirely from
# scripts/random_reward_fn.py (see RANDOM_REWARD_SEED).
#
# Usage:
#   bash scripts/run_verl_grpo_randomreward_qwen35_9b.sh
#   DUMMY_ROWS=2048 RANDOM_REWARD_SEED=123 bash scripts/run_verl_grpo_randomreward_qwen35_9b.sh
#
# Start from scratch:
#   FRESH_START=1 bash scripts/run_verl_grpo_randomreward_qwen35_9b.sh

set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}

# Placeholder parquet: many identical rows so train_batch_size can be > 1 (DataLoader drop_last).
DUMMY_PARQUET=${DUMMY_PARQUET:-$PROJECT_ROOT/data/random_reward/dummy.parquet}
DUMMY_ROWS=${DUMMY_ROWS:-1024}
export RANDOM_REWARD_SEED=${RANDOM_REWARD_SEED:-0}

REWARD_FN="$PROJECT_ROOT/scripts/random_reward_fn.py"

mkdir -p "$(dirname "$DUMMY_PARQUET")"
python3 - "$DUMMY_PARQUET" "$DUMMY_ROWS" <<'PY'
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
n = int(sys.argv[2])
prompt = [{"role": "user", "content": "Reply with a short friendly sentence."}]
rows = [
    {
        "prompt": prompt,
        "reward_model": {"ground_truth": "dummy"},
        "data_source": "random_reward",
    }
    for _ in range(n)
]
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(out, index=False)
print(f"Wrote {n} placeholder rows to {out}")
PY

TRAIN_FILES="['$DUMMY_PARQUET']"
VAL_FILES="['$DUMMY_PARQUET']"

TP=${TP:-1}
NGPU=${NGPU:-4}

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

CKPT_DIR="$PROJECT_ROOT/checkpoints/verl_randomreward_qwen35_9b/qwen35_9b_grpo"
FRESH_START_ARGS=""
if [ "${FRESH_START:-0}" = "1" ]; then
    if [ -d "$CKPT_DIR" ]; then
        BACKUP_DIR="${CKPT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
        echo "[FRESH_START] Backing up checkpoints: $CKPT_DIR -> $BACKUP_DIR"
        mv "$CKPT_DIR" "$BACKUP_DIR"
        echo "[FRESH_START] Backup done."
    fi
    FRESH_START_ARGS="trainer.resume_mode=disable"
    echo "[FRESH_START] Training will start from scratch (resume_mode=disable)."
fi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=32000 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.prompt_key=prompt \
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_randomreward_qwen35_9b \
    trainer.rollout_data_dir=$PROJECT_ROOT/outputs/rollout_data_randomreward \
    trainer.experiment_name=qwen35_9b_grpo_randomreward \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.total_epochs=60 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    $FRESH_START_ARGS \
    "$@"
