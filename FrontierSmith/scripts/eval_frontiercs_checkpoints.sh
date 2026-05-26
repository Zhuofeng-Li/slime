#!/bin/bash
# Evaluate base model (Qwen/Qwen3.5-9B) + Frontier-CS checkpoints (steps 0,10,...,180) on 172 problems.
# Uses vLLM server + OpenAI API for generation, Frontier-CS judge for scoring.
# Reports: score@1, avg_score@5, best_score@5 for each step.
#
# Per-GPU pipeline: each GPU runs a loop (start vLLM -> eval -> kill -> next step).
# No batch wait, no PID_DIR. Each GPU is continuously busy.
#
# Prerequisites:
#   1. Frontier-CS judge on :8082 (cd Frontier-CS/algorithmic && ./run_judge.sh)
#   2. data/frontiercs/full.parquet (python scripts/prepare_frontiercs_parquet.py --full-for-both)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/eval_frontiercs_checkpoints.sh
#   STEPS="0 10 20 30" bash scripts/eval_frontiercs_checkpoints.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

# base = Qwen/Qwen3.5-9B (no training); 0,10,...,180 = checkpoints
STEPS=${STEPS:-"0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500"}
CKPT_BASE="$PROJECT_ROOT/checkpoints/verl_frontiercs_qwen35_9b/qwen35_9b_grpo"
MODELS_DIR="$PROJECT_ROOT/models"
VAL_DATA=${VAL_DATA:-$PROJECT_ROOT/data/frontiercs/full.parquet}
MODEL_PATH_BASE=${MODEL_PATH_BASE:-Qwen/Qwen3.5-9B}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-16}
BASE_PORT=${BASE_PORT:-8010}
OUTPUT_CSV=${OUTPUT_CSV:-$PROJECT_ROOT/outputs/frontiercs_checkpoint_evals.csv}
GPU_CLEANUP_SLEEP=${GPU_CLEANUP_SLEEP:-30}

if [ ! -f "$VAL_DATA" ]; then
    echo "Val data not found: $VAL_DATA"
    echo "Run: python scripts/prepare_frontiercs_parquet.py --full-for-both"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"
RESULT_BASE="${OUTPUT_CSV%.csv}"

get_model_path() {
    local step=$1
    if [ "$step" = "base" ] || [ "$step" = "0" ]; then
        echo "$MODEL_PATH_BASE"
    else
        local merged="$MODELS_DIR/qwen35_9b_grpo_step${step}"
        if [ -d "$merged" ]; then
            echo "$merged"
        else
            local ckpt="$CKPT_BASE/global_step_$step"
            if [ ! -d "$ckpt/actor" ]; then
                echo ""
                return
            fi
            echo "MERGE:$ckpt:$merged"
        fi
    fi
}

wait_for_port() {
    local port=$1
    local i=0
    while [ $i -lt 120 ]; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1 || \
           curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep 2
        i=$((i + 1))
    done
    return 1
}

kill_vllm_by_port() {
    local port=$1
    fuser -k "${port}/tcp" 2>/dev/null || true
}

eval_one_step() {
    local port=$1
    local step=$2
    local base_url="http://localhost:${port}/v1"
    local out
    if [ "${EVAL_DEBUG:-0}" = "1" ]; then
        out=$(python scripts/eval_frontiercs_via_vllm.py \
            --data "$VAL_DATA" \
            --base-url "$base_url" \
            --n-samples 5 \
            --concurrency "$EVAL_CONCURRENCY" \
            --print-csv-row \
            --print-progress \
            --progress-prefix "[step $step] " \
            2>&1 | tee /dev/stderr | grep "EVAL_CSV_ROW:" || true)
    else
        out=$(python scripts/eval_frontiercs_via_vllm.py \
            --data "$VAL_DATA" \
            --base-url "$base_url" \
            --n-samples 5 \
            --concurrency "$EVAL_CONCURRENCY" \
            --print-csv-row \
            --print-progress \
            --progress-prefix "[step $step] " \
            2>&1 | tee /dev/stderr | grep "EVAL_CSV_ROW:" || true)
    fi
    local score_at_1 avg_score_at_5 best_score_at_5
    score_at_1=$(echo "$out" | sed 's/EVAL_CSV_ROW://' | cut -d, -f1)
    avg_score_at_5=$(echo "$out" | sed 's/EVAL_CSV_ROW://' | cut -d, -f2)
    best_score_at_5=$(echo "$out" | sed 's/EVAL_CSV_ROW://' | cut -d, -f3)
    score_at_1=${score_at_1:-"NA"}
    avg_score_at_5=${avg_score_at_5:-"NA"}
    best_score_at_5=${best_score_at_5:-"NA"}
    if [ "$score_at_1" = "NA" ]; then
        echo "  [GPU $3 step $step] NA" >&2
    fi
    echo "$step,$score_at_1,$avg_score_at_5,$best_score_at_5"
}

# Per-GPU worker: runs a loop over its assigned steps. No wait for other GPUs.
gpu_worker() {
    local gpu_idx=$1
    local port=$2
    local result_file=$3
    shift 3
    local steps=("$@")
    local gpu_id=${GPU_ARR[$gpu_idx]}

    for step in "${steps[@]}"; do
        model_path=$(get_model_path "$step")
        if [ -z "$model_path" ]; then
            echo "$step,SKIP,SKIP,SKIP" >> "$result_file"
            continue
        fi

        echo "[GPU $gpu_idx] Step $step: starting vLLM on port $port"
        CUDA_VISIBLE_DEVICES=$gpu_id vllm serve "$model_path" \
            --host 127.0.0.1 \
            --port "$port" \
            --tensor-parallel-size 1 \
            --max-model-len 24576 &
        local vllm_pid=$!

        if ! wait_for_port "$port"; then
            echo "[GPU $gpu_idx] Step $step: vLLM failed to start" >&2
            kill $vllm_pid 2>/dev/null || true
            kill_vllm_by_port "$port"
            echo "$step,ERR,ERR,ERR" >> "$result_file"
            sleep "$GPU_CLEANUP_SLEEP"
            continue
        fi

        result=$(eval_one_step "$port" "$step" "$gpu_idx") || result="$step,NA,NA,NA"
        echo "$result" >> "$result_file"
        echo "[GPU $gpu_idx] Step $step done: $result"

        kill -TERM $vllm_pid 2>/dev/null || true
        sleep 2
        kill -9 $vllm_pid 2>/dev/null || true
        pkill -9 -P $vllm_pid 2>/dev/null || true
        kill_vllm_by_port "$port"

        sleep "$GPU_CLEANUP_SLEEP"
    done
}

# 1. Pre-merge all models that need merging (sequential, avoids races)
STEPS_ARR=($STEPS)
echo "========== Pre-merge phase =========="
for step in "${STEPS_ARR[@]}"; do
    model_path=$(get_model_path "$step")
    if [[ "$model_path" == MERGE:* ]]; then
        rest="${model_path#MERGE:}"
        ckpt="${rest%%:*}"
        merged="${rest#*:}"
        if [ ! -d "$merged" ]; then
            echo "Merging $ckpt -> $merged ..."
            python -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "$ckpt/actor" \
                --target_dir "$merged"
        fi
    fi
done

# 2. Assign steps to GPUs: GPU i gets steps at indices i, i+NGPU, i+2*NGPU, ...
# 3. Launch one worker per GPU (each writes to its own result file)
echo "========== Launching $NGPU GPU workers =========="
echo "step,score_at_1,avg_score_at_5,best_score_at_5" > "$OUTPUT_CSV"

for gpu_idx in $(seq 0 $((NGPU - 1))); do
    port=$((BASE_PORT + gpu_idx))
    result_file="${RESULT_BASE}_gpu${gpu_idx}.csv"
    : > "$result_file"

    steps_for_gpu=()
    for (( idx=gpu_idx; idx<${#STEPS_ARR[@]}; idx+=NGPU )); do
        steps_for_gpu+=("${STEPS_ARR[$idx]}")
    done

    if [ ${#steps_for_gpu[@]} -eq 0 ]; then
        continue
    fi
    echo "GPU $gpu_idx (port $port): steps ${steps_for_gpu[*]}"
    gpu_worker "$gpu_idx" "$port" "$result_file" "${steps_for_gpu[@]}" &
done

# 4. Wait for all GPU workers to finish
wait || true

# 5. Merge per-GPU result files into final CSV
for gpu_idx in $(seq 0 $((NGPU - 1))); do
    result_file="${RESULT_BASE}_gpu${gpu_idx}.csv"
    if [ -f "$result_file" ] && [ -s "$result_file" ]; then
        cat "$result_file" >> "$OUTPUT_CSV"
        rm -f "$result_file"
    fi
done

# 6. Sort by step (base first, then 0,10,...,180)
( head -1 "$OUTPUT_CSV"; tail -n +2 "$OUTPUT_CSV" | sed 's/^base,/-1,/' | sort -t, -k1 -n | sed 's/^-1,/base,/' ) > "${OUTPUT_CSV}.tmp" && mv "${OUTPUT_CSV}.tmp" "$OUTPUT_CSV"

echo ""
echo "Results saved to $OUTPUT_CSV"
echo ""
echo "Summary:"
column -t -s, < "$OUTPUT_CSV"
