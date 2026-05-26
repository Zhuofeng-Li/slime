#!/bin/bash
# GRPO training for Qwen3.5-9B on Frontier-CS prompts (VERL)
# with random continuous training reward in [0, 1).
#
# This keeps the same:
#   - base model
#   - prompts / train parquet
#   - eval set
#   - rollout.n
#   - batch size
#   - learning rate
#   - KL coefficient
#   - max response length
#   - number of training steps / epochs
# as scripts/run_verl_grpo_frontiercs_qwen35_9b.sh
#
# Training rewards come from random reward; validation uses the normal
# Frontier-CS / ALE-Bench judge rewards, matching run_verl_grpo_frontiercs_qwen35_9b.sh.
# Set RANDOM_REWARD_SEED to control reproducibility.

set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

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
export ALE_BENCH_CACHE_DIR="${ALE_BENCH_CACHE_DIR:-$PROJECT_ROOT/.cache/ale-bench}"
export ALEBENCH_JUDGE_VERSION=${ALEBENCH_JUDGE_VERSION:-202301}
export RANDOM_REWARD_SEED=${RANDOM_REWARD_SEED:-0}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
REWARD_FN=${REWARD_FN:-$PROJECT_ROOT/scripts/random_train_default_val_reward_fn.py}

TRAIN_DATA=${TRAIN_DATA:-$PROJECT_ROOT/data/frontiercs/train.parquet}
RANDOM_TRAIN_DATA=${RANDOM_TRAIN_DATA:-${TRAIN_DATA%.parquet}_randomreward.parquet}
RANDOM_TRAIN_DATA_SOURCE=${RANDOM_TRAIN_DATA_SOURCE:-frontiercs_random_reward}
export RANDOM_REWARD_DATA_SOURCES="${RANDOM_REWARD_DATA_SOURCES:-$RANDOM_TRAIN_DATA_SOURCE,random_reward}"
VAL_DATA=${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}
ALEBENCH_VAL_DATA=${ALEBENCH_VAL_DATA:-$PROJECT_ROOT/data/alebench/val.parquet}

TP=${TP:-2}
NGPU=${NGPU:-8}

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Run: python scripts/prepare_frontiercs_parquet.py"
    exit 1
fi

if [ ! -f "$RANDOM_TRAIN_DATA" ] || [ "$TRAIN_DATA" -nt "$RANDOM_TRAIN_DATA" ]; then
    python scripts/prepare_random_reward_train_parquet.py \
        --input "$TRAIN_DATA" \
        --output "$RANDOM_TRAIN_DATA" \
        --data-source "$RANDOM_TRAIN_DATA_SOURCE"
fi

TRAIN_FILES="['$RANDOM_TRAIN_DATA']"

VAL_LIST=()
if [ -f "$VAL_DATA" ]; then
    VAL_LIST+=("'$VAL_DATA'")
else
    VAL_LIST+=("'$TRAIN_DATA'")
fi
if [ -f "$ALEBENCH_VAL_DATA" ]; then
    VAL_LIST+=("'$ALEBENCH_VAL_DATA'")
    echo "[ALE-Bench] Including $ALEBENCH_VAL_DATA in validation."
else
    echo "[ALE-Bench] $ALEBENCH_VAL_DATA not found; run scripts/prepare_alebench_parquet.py to enable."
fi
VAL_FILES="[$(IFS=,; echo "${VAL_LIST[*]}")]"

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

CKPT_DIR="$PROJECT_ROOT/checkpoints/verl_frontiercs_qwen35_9b_randomreward/qwen35_9b_grpo_randomreward"
ROLLOUT_DIR="$PROJECT_ROOT/outputs/rollout_data_randomreward"
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
    echo "[FRESH_START] Training will start from scratch (resume_mode=disable)."
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
    reward.custom_reward_function.path="$REWARD_FN" \
    reward.custom_reward_function.name=compute_score \
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
    actor_rollout_ref.rollout.val_kwargs.n=5 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_frontiercs_qwen35_9b_randomreward \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.rollout_data_dir=$ROLLOUT_DIR \
    trainer.experiment_name=qwen35_9b_grpo_randomreward \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.total_epochs=60 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    $FRESH_START_ARGS \
    "$@"
