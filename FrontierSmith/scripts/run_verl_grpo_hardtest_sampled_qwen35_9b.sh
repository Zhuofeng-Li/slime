#!/bin/bash
# GRPO for Qwen3.5-9B on HardTests: N problems random-sampled from ALL local hardtest_* pools
# (not the same 800 as the synthetic L2 pipeline).
#
# Problem packages use prefix hardtest_smp_* so they do not overwrite hardtest_orig_* (pipeline-800).
#
# Prerequisites:
#   1. Local tiers populated: data/problems/hardtest_medium_hard, hardtest_hard, hardtest_very_hard
#   2. python scripts/sample_hardtest_problems.py --n 800 --seed 42
#   3. python scripts/install_hardtest_frontier_packages.py \
#        --filtered-json results/hardtest_sampled_800.json --package-prefix hardtest_smp_
#   4. python scripts/prepare_hardtest_original_parquet.py \
#        --package-prefix hardtest_smp_ --output data/frontiercs/train_hardtest_sampled.parquet
#   5. python scripts/prepare_frontiercs_parquet.py --full-for-both
#   6. Frontier-CS judge :8082 (cd Frontier-CS/algorithmic && ./run_judge.sh)
#   7. 4–8 GPUs
#
# Reproducibility: fix --seed in step 2; change SEED / MANIFEST / TRAIN_DATA via env if needed.
#
# Usage:
#   bash scripts/run_verl_grpo_hardtest_sampled_qwen35_9b.sh
#   SAMPLE_SEED=123 bash scripts/run_verl_grpo_hardtest_sampled_qwen35_9b.sh   # only if you re-run steps 2–4 with same seed
#
# Start from scratch:
#   FRESH_START=1 bash scripts/run_verl_grpo_hardtest_sampled_qwen35_9b.sh

set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export FRONTIER_CS_PROBLEMS_DIR="$PROJECT_ROOT/Frontier-CS/algorithmic/problems"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}

TRAIN_DATA=${TRAIN_DATA:-$PROJECT_ROOT/data/frontiercs/train_hardtest_sampled.parquet}
VAL_DATA=${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}

TP=${TP:-1}
NGPU=${NGPU:-4}

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Missing $TRAIN_DATA"
    echo "Run: python scripts/sample_hardtest_problems.py --n 800 --seed \${SAMPLE_SEED:-42}"
    echo "     python scripts/install_hardtest_frontier_packages.py \\"
    echo "       --filtered-json results/hardtest_sampled_800.json --package-prefix hardtest_smp_"
    echo "     python scripts/prepare_hardtest_original_parquet.py \\"
    echo "       --package-prefix hardtest_smp_ --output data/frontiercs/train_hardtest_sampled.parquet"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "Run: python scripts/prepare_frontiercs_parquet.py --full-for-both"
    exit 1
fi

TRAIN_FILES="['$TRAIN_DATA']"
VAL_FILES="['$VAL_DATA']"

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

CKPT_DIR="$PROJECT_ROOT/checkpoints/verl_hardtest_sampled_qwen35_9b/qwen35_9b_grpo_hardtest_sampled"
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
    trainer.project_name=verl_hardtest_sampled_qwen35_9b \
    trainer.rollout_data_dir=$PROJECT_ROOT/outputs/rollout_data_hardtest_sampled \
    trainer.experiment_name=qwen35_9b_grpo_hardtest_sampled \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.total_epochs=60 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    $FRESH_START_ARGS \
    "$@"
