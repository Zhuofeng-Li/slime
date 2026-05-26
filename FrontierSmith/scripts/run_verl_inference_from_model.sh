#!/bin/bash
# Run inference using VERL's hybrid flow (FSDP loads model, syncs to vLLM).
# This avoids the vLLM DTensor error when loading Qwen3.5-VL from disk directly.
#
# Prerequisites:
#   1. Merged model at models/qwen35_9b_grpo_step105 (from merge_fsdp_to_hf.py)
#   2. Frontier-CS/.venv activated, or verl installed
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/run_verl_inference_from_model.sh
#   MODEL_PATH=/path/to/model bash scripts/run_verl_inference_from_model.sh

set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Use merged model by default
MODEL_PATH=${MODEL_PATH:-$PROJECT_ROOT/models/qwen35_9b_grpo_step105}
TRAIN_DATA=${TRAIN_DATA:-$PROJECT_ROOT/data/frontiercs/train.parquet}

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found: $MODEL_PATH"
    echo "Run: python scripts/merge_fsdp_to_hf.py --ckpt <checkpoint> --output $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Data not found: $TRAIN_DATA"
    echo "Run: python scripts/prepare_frontiercs_parquet.py"
    exit 1
fi

TP=${TP:-1}
NGPU=${NGPU:-4}
TRAIN_FILES="['$TRAIN_DATA']"
VAL_FILES="['$TRAIN_DATA']"

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Run 1 epoch with minimal batch to verify model loads and inference works
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=2 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=dummy \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=verl_inference_test \
    trainer.experiment_name=inference_from_model \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=999 \
    trainer.test_freq=999 \
    "$@"
