#!/bin/bash
# VERL inference: load checkpoint, sync to vLLM, run validation only, then exit.
#
# This uses VERL's internal path: FSDP loads checkpoint -> update_weights() syncs to vLLM.
# No disk HF loading by vLLM, so avoids DTensor errors.
#
# Prerequisites:
#   1. Frontier-CS judge on :8082 (for validation reward)
#   2. Same GPU/TP as training (e.g. 4 GPUs, TP=1)
#
# Usage:
#   bash scripts/run_verl_inference_from_ckpt.sh
#   CKPT_PATH=checkpoints/.../global_step_90  bash scripts/run_verl_inference_from_ckpt.sh

set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Checkpoint to load (must contain actor/ subdir, i.e. global_step_XXX)
CKPT_PATH=${CKPT_PATH:-$PROJECT_ROOT/checkpoints/verl_frontiercs_qwen35_9b/qwen35_9b_grpo/global_step_105}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
TRAIN_DATA=${TRAIN_DATA:-$PROJECT_ROOT/data/frontiercs/train.parquet}
VAL_DATA=${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/val.parquet}

TP=${TP:-1}
NGPU=${NGPU:-4}

if [ ! -d "$CKPT_PATH/actor" ]; then
    echo "Checkpoint not found: $CKPT_PATH (need actor/ subdir)"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    VAL_DATA=$TRAIN_DATA
fi

TRAIN_FILES="['$TRAIN_DATA']"
VAL_FILES="['$VAL_DATA']"

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Key overrides for inference-only:
#   resume_mode=resume_path, resume_from_path=$CKPT_PATH
#   val_only=True  -> run validation once and exit (no training)
#   val_before_train=True -> validation runs after checkpoint load
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_frontiercs_qwen35_9b \
    trainer.experiment_name=qwen35_9b_grpo_inference \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=$CKPT_PATH \
    trainer.val_only=True \
    trainer.val_before_train=True \
    "$@"
