#!/bin/bash
# GRPO on synthetic data (800 L2-top), warm-starting from hardtest_sampled checkpoint at global_step_40.
#
# Loads weights (and dataloader state if present) from:
#   checkpoints/verl_hardtest_sampled_qwen35_9b/qwen35_9b_grpo_hardtest_sampled/global_step_40
# New checkpoints / wandb run use a distinct experiment name so they do not collide with pure-synthetic runs.
#
# Prerequisites: same as run_verl_grpo_synthetic_qwen35_9b.sh (train_synthetic.parquet, full.parquet, judge).
# You must have completed hardtest_sampled training through at least step 40 (save_freq=20 -> step 40 exists).
#
# Override checkpoint path:
#   RESUME_CKPT=/abs/path/to/global_step_40 bash scripts/run_verl_grpo_synthetic_from_hardtest_sampled_s40_qwen35_9b.sh
#
# Usage:
#   bash scripts/run_verl_grpo_synthetic_from_hardtest_sampled_s40_qwen35_9b.sh

set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export FRONTIER_CS_PROBLEMS_DIR="$PROJECT_ROOT/Frontier-CS/algorithmic/problems"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
TRAIN_DATA=${TRAIN_DATA:-$PROJECT_ROOT/data/frontiercs/train_synthetic.parquet}
VAL_DATA=${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}

DEFAULT_RESUME="$PROJECT_ROOT/checkpoints/verl_hardtest_sampled_qwen35_9b/qwen35_9b_grpo_hardtest_sampled/global_step_40"
RESUME_CKPT=${RESUME_CKPT:-$DEFAULT_RESUME}

TP=${TP:-1}
NGPU=${NGPU:-4}

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Missing $TRAIN_DATA — run filter + prepare_synthetic_parquet"
    exit 1
fi
if [ ! -f "$VAL_DATA" ]; then
    echo "Missing $VAL_DATA — run prepare_frontiercs_parquet.py --full-for-both"
    exit 1
fi
if [ ! -d "$RESUME_CKPT" ]; then
    echo "Missing checkpoint dir: $RESUME_CKPT"
    echo "Train hardtest_sampled to at least global_step_40, or set RESUME_CKPT to an existing global_step_* folder."
    exit 1
fi

TRAIN_FILES="['$TRAIN_DATA']"
VAL_FILES="['$VAL_DATA']"

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Do not use FRESH_START=1 here — it would disable resume.
if [ "${FRESH_START:-0}" = "1" ]; then
    echo "Do not set FRESH_START=1 for this script; it is for resume from hardtest_sampled. Abort."
    exit 1
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
    trainer.project_name=verl_synthetic_qwen35_9b \
    trainer.rollout_data_dir=$PROJECT_ROOT/outputs/rollout_data_synthetic_from_hsampled_s40 \
    trainer.experiment_name=qwen35_9b_grpo_synthetic_from_hsampled_s40 \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.total_epochs=60 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$RESUME_CKPT" \
    "$@"
