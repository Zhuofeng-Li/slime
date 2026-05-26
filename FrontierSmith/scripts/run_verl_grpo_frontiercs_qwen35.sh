#!/bin/bash
# GRPO training for Qwen3.5-27B on Frontier-CS (VERL)
#
# Prerequisites:
#   1. python scripts/prepare_frontiercs_parquet.py
#   2. Frontier-CS judge on :8082 (cd Frontier-CS/algorithmic && ./run_judge.sh)
#   3. 4+ GPUs (e.g. 4x H100 or 8x A100)
#
# Usage:
#   bash scripts/run_verl_grpo_frontiercs_qwen35.sh
#   bash scripts/run_verl_grpo_frontiercs_qwen35.sh  # with custom args appended

set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-27B}
TRAIN_DATA=${TRAIN_DATA:-$PROJECT_ROOT/data/frontiercs/train.parquet}
VAL_DATA=${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/val.parquet}

TP=${TP:-2}
NGPU=${NGPU:-4}

# Ensure data exists
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

# Ensure Ray workers inherit CUDA_VISIBLE_DEVICES (e.g. 4,5,6,7).
# Without this, Ray may override it and workers could use wrong GPUs.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
# Note: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is incompatible with vLLM's CuMem memory pool
# Lengths aligned with Frontier-CS API (max_tokens=32000) and Qwen3.5-27B (max_position=262144):
# - max_prompt_length=8192: longest statement ~20K chars (~5-6K tokens); 2048 would filter 8+ problems
# - max_response_length=32000: solutions can be 500-1000+ lines; 2048 caused CE and reward 0
# If OOM: reduce data.train_batch_size (e.g. 32) or actor.ppo_mini_batch_size (e.g. 16)

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.prompt_key=prompt \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_frontiercs_qwen35 \
    trainer.experiment_name=qwen35_27b_grpo \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.total_epochs=5 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    "$@"
