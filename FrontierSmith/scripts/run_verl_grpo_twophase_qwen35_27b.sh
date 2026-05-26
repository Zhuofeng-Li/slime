#!/bin/bash
# Two-phase GRPO curriculum for Qwen3.5-27B:
#   Phase 1 (steps 1–100): 2 epochs on 400 hardtest problems  (400 / 8 = 50 steps/epoch × 2 = 100 steps)
#   Phase 2 (steps 101–200): 2 epochs on 400 synthetic problems (400 / 8 = 50 steps/epoch × 2 = 100 steps)
#
# Phase 2 loads weights from the Phase 1 checkpoint (global_step_100) via resume_path,
# and continues the global_step counter (so wandb / progress shows 101–200 continuously).
#
# Key adjustments vs 9B twophase:
#   - TP=2 (27B needs tensor parallelism)
#   - NGPU=8 (more GPUs for larger model)
#   - train_batch_size=8 (halved to fit 27B in memory)
#   - max_response_length=16384 (reduced from 32000, still generous for code)
#   - ppo_mini_batch_size=8 (match train_batch_size)
#   - lr=5e-7 (lower for larger model stability)
#   - rollout.n=8 (keep sampling diversity for GRPO)
#   - gpu_memory_utilization=0.7
#
# Prerequisites:
#   1. python scripts/prepare_mixed_400_parquets.py [--seed 42]
#      -> data/frontiercs/train_hardtest_400.parquet
#      -> data/frontiercs/train_synthetic_400.parquet
#   2. python scripts/prepare_frontiercs_parquet.py --full-for-both  (for full.parquet val)
#   3. Frontier-CS judge on :8082
#   4. 8 GPUs (e.g. 8x H100 or 8x A100-80G)
#
# Usage:
#   bash scripts/run_verl_grpo_twophase_qwen35_27b.sh            # run both phases
#   PHASE=1 bash scripts/run_verl_grpo_twophase_qwen35_27b.sh    # phase 1 only
#   PHASE=2 bash scripts/run_verl_grpo_twophase_qwen35_27b.sh    # phase 2 only (phase 1 must have finished)
#
# Start from scratch:
#   FRESH_START=1 bash scripts/run_verl_grpo_twophase_qwen35_27b.sh

set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export FRONTIER_CS_PROBLEMS_DIR="$PROJECT_ROOT/Frontier-CS/algorithmic/problems"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-27B}
VAL_DATA=${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}

HARDTEST_DATA=${HARDTEST_DATA:-$PROJECT_ROOT/data/frontiercs/train_hardtest_400.parquet}
SYNTHETIC_DATA=${SYNTHETIC_DATA:-$PROJECT_ROOT/data/frontiercs/train_synthetic_400.parquet}

TP=${TP:-2}
NGPU=${NGPU:-8}

# batch_size=8, 400 problems => 50 steps/epoch, 2 epochs => 100 steps per phase
PHASE1_EPOCHS=2
PHASE1_STEPS=100
# Phase 2 total_epochs must be large enough so the epoch loop doesn't exhaust before
# total_training_steps is reached. VERL computes current_epoch = global_steps // len(dataloader);
# after resume at step 100 with a 50-batch dataloader, current_epoch=2, so total_epochs must be > 2+2=4.
PHASE2_EPOCHS=10
TOTAL_STEPS=200

EXPERIMENT="qwen35_27b_grpo_twophase"
PROJECT="verl_twophase_qwen35_27b"
CKPT_BASE="$PROJECT_ROOT/checkpoints/$PROJECT/$EXPERIMENT"

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ── Validation ────────────────────────────────────────────────────────────
for f in "$HARDTEST_DATA" "$SYNTHETIC_DATA"; do
    if [ ! -f "$f" ]; then
        echo "Missing $f — run: python scripts/prepare_mixed_400_parquets.py"
        exit 1
    fi
done
if [ ! -f "$VAL_DATA" ]; then
    echo "Missing $VAL_DATA — run: python scripts/prepare_frontiercs_parquet.py --full-for-both"
    exit 1
fi

# ── Fresh start ───────────────────────────────────────────────────────────
if [ "${FRESH_START:-0}" = "1" ]; then
    if [ -d "$CKPT_BASE" ]; then
        BACKUP="${CKPT_BASE}_backup_$(date +%Y%m%d_%H%M%S)"
        echo "[FRESH_START] Backing up $CKPT_BASE -> $BACKUP"
        mv "$CKPT_BASE" "$BACKUP"
    fi
fi

PHASE=${PHASE:-0}  # 0 = both

# ── Common VERL args (everything except data, epochs, resume, experiment) ─
COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_batch_size=8
    data.max_prompt_length=8192
    data.max_response_length=16384
    data.filter_overlong_prompts=True
    data.truncation=error
    data.prompt_key=prompt
    actor_rollout_ref.model.path=$MODEL_PATH
    actor_rollout_ref.model.use_remove_padding=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.actor.optim.lr=5e-7
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT
    trainer.n_gpus_per_node=$NGPU
    trainer.nnodes=1
    trainer.save_freq=50
    trainer.test_freq=10
)

# ── Phase 1: Hardtest 400 × 2 epochs = 100 steps ────────────────────────
run_phase1() {
    echo "===== Phase 1: Hardtest (steps 1–$PHASE1_STEPS) ====="
    RESUME_ARGS=""
    if [ "${FRESH_START:-0}" = "1" ]; then
        RESUME_ARGS="trainer.resume_mode=disable"
    fi
    python -m verl.trainer.main_ppo \
        "${COMMON_ARGS[@]}" \
        data.train_files="['$HARDTEST_DATA']" \
        data.val_files="['$VAL_DATA']" \
        trainer.total_epochs=$PHASE1_EPOCHS \
        trainer.total_training_steps=$PHASE1_STEPS \
        trainer.experiment_name=$EXPERIMENT \
        trainer.rollout_data_dir=$PROJECT_ROOT/outputs/rollout_data_twophase_27b \
        $RESUME_ARGS \
        "$@"
}

# ── Phase 2: Synthetic 400 × 2 epochs = 100 steps (resume from phase 1) ─
run_phase2() {
    PHASE1_CKPT="$CKPT_BASE/global_step_$PHASE1_STEPS"
    if [ ! -d "$PHASE1_CKPT" ]; then
        echo "Phase 1 checkpoint not found: $PHASE1_CKPT"
        echo "Run phase 1 first, or set PHASE=1"
        exit 1
    fi
    echo "===== Phase 2: Synthetic (steps $((PHASE1_STEPS+1))–$TOTAL_STEPS) ====="
    python -m verl.trainer.main_ppo \
        "${COMMON_ARGS[@]}" \
        data.train_files="['$SYNTHETIC_DATA']" \
        data.val_files="['$VAL_DATA']" \
        trainer.total_epochs=$PHASE2_EPOCHS \
        trainer.total_training_steps=$TOTAL_STEPS \
        trainer.experiment_name=$EXPERIMENT \
        trainer.rollout_data_dir=$PROJECT_ROOT/outputs/rollout_data_twophase_27b \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="$PHASE1_CKPT" \
        "$@"
}

# ── Dispatch ──────────────────────────────────────────────────────────────
if [ "$PHASE" = "1" ]; then
    run_phase1 "$@"
elif [ "$PHASE" = "2" ]; then
    run_phase2 "$@"
else
    run_phase1 "$@" || echo "[WARN] Phase 1 exited with code $?, proceeding to Phase 2..."
    run_phase2 "$@"
fi
