#!/bin/bash
# 评估基座模型 (checkpoint 0) 在 Frontier-CS full.parquet 上的 accuracy
#
# Prerequisites:
#   1. Frontier-CS judge 在 :8082 运行 (cd Frontier-CS/algorithmic && ./run_judge.sh)
#   2. 与训练相同的 GPU 配置 (如 4 GPUs, TP=1)
#
# Usage:
#   bash scripts/eval_base_model_frontiercs.sh
#   VAL_DATA=data/frontiercs/val.parquet bash scripts/eval_base_model_frontiercs.sh  # 仅 17 题
#   NGPU=2 bash scripts/eval_base_model_frontiercs.sh  # 使用 2 张 GPU

set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 基座模型，不做 resume
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
VAL_DATA=${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}

TP=${TP:-1}
NGPU=${NGPU:-4}

if [ ! -f "$VAL_DATA" ]; then
    echo "Val data not found: $VAL_DATA"
    echo "Run: python scripts/prepare_frontiercs_parquet.py --full-for-both"
    exit 1
fi

TRAIN_FILES="['$VAL_DATA']"   # 仅用于占位，不会训练
VAL_FILES="['$VAL_DATA']"

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# val_only=True: 只跑 validation，不训练
# val_before_train=True: 加载模型后立即跑 validation 然后退出
# 不设置 resume_mode/resume_from_path，使用 MODEL_PATH 作为基座模型
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
    trainer.logger='["console"]' \
    trainer.project_name=verl_frontiercs_qwen35_9b \
    trainer.experiment_name=eval_base_model \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.val_only=True \
    trainer.val_before_train=True \
    "$@"
