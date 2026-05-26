#!/bin/bash
# GRPO training for Qwen3.5-27B on Frontier-CS (VERL)
# Tuned for 4× B200 (192GB per GPU).
#
# Critical memory notes for 27B on 4 GPUs:
#   - FSDP shards along data-parallel dim: DP = NGPU / TP
#   - TP=4 with NGPU=4 gives DP=1 (NO sharding) → each GPU holds full 27B
#     → params (54GB) + grads (54GB) + activations easily OOM
#   - TP=2 with NGPU=4 gives DP=2 → params/grads sharded to 27GB each → fits
#   - optimizer_offload=True offloads Adam m/v (216GB for 27B) to CPU
#
# Prerequisites:
#   1. python scripts/prepare_frontiercs_parquet.py
#   2. Frontier-CS judge on :8082 (cd Frontier-CS/algorithmic && ./run_judge.sh)
#   3. 4× B200 GPUs (192GB each)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/run_verl_grpo_frontiercs_qwen35_27b.sh
#
# Start from scratch:
#   FRESH_START=1 CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/run_verl_grpo_frontiercs_qwen35_27b.sh
#
# To debug response length spikes, add:
#   trainer.dump_long_responses_dir=outputs/dump_long_responses \
#   trainer.dump_long_responses_threshold=20000 \

set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-27B}

TRAIN_DATA=${TRAIN_DATA:-$PROJECT_ROOT/data/frontiercs/train.parquet}
VAL_DATA=${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}

# TP=2 so FSDP can shard params/grads across DP=2 groups.
TP=${TP:-2}
NGPU=${NGPU:-4}

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Run: python scripts/prepare_frontiercs_parquet.py"
    exit 1
fi

TRAIN_FILES="['$TRAIN_DATA']"
if [ -f "$VAL_DATA" ]; then
    VAL_FILES="['$VAL_DATA']"
else
    VAL_FILES="['$TRAIN_DATA']"
fi

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# FRESH_START=1: backup existing checkpoints and start from scratch
CKPT_DIR="$PROJECT_ROOT/checkpoints/verl_frontiercs_qwen35_27b/qwen35_27b_grpo"
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
    data.max_response_length=32000 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=5e-7 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.val_kwargs.n=5 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_frontiercs_qwen35_27b \
    trainer.rollout_data_dir=$PROJECT_ROOT/outputs/rollout_data_27b \
    trainer.experiment_name=qwen35_27b_grpo \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.total_epochs=60 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    $FRESH_START_ARGS \
    "$@"
