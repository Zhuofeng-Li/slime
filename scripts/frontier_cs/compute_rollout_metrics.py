"""
Compute epoch-level FrontierCS metrics from saved rollout .pt files.

Each sample's data epoch is determined by sample['index'] // (dataset_size * n_samples_per_prompt),
NOT by rollout_id boundaries — because a single rollout can straddle two data epochs when
dataset_size is not divisible by rollout_batch_size.

Usage:
    python scripts/frontier_cs/compute_rollout_metrics.py \
        --rollout-dir /shared/dev/zhuofeng/slime/logs/frontiercs_rollouts/Qwen3.5-27B-frontiercs-debug-64-h200 \
        --dataset-size 188 \
        --n-samples-per-prompt 8

Mirrors the logic in scripts/frontier_cs/rollout_wandb.py.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch


def best_of_k_estimate(rewards: list[float], k: int) -> float:
    sorted_rewards = sorted(rewards)
    n = len(sorted_rewards)
    expected = 0.0
    prev_cdf = 0.0
    for i, reward in enumerate(sorted_rewards, start=1):
        cdf = (i / n) ** k
        expected += reward * (cdf - prev_cdf)
        prev_cdf = cdf
    return expected


def compute_epoch_metrics(epoch_groups: dict[str, list[float]]) -> dict[str, float]:
    metrics = {}
    group_lists = list(epoch_groups.values())
    per_problem_avg = [sum(g) / len(g) for g in group_lists]
    metrics["epoch_avg_score"] = sum(per_problem_avg) / len(per_problem_avg)

    min_n = min(len(g) for g in group_lists)
    k = min_n
    while k >= 1:
        if k == min_n:
            per_problem_best = [max(g) for g in group_lists]
        else:
            per_problem_best = [best_of_k_estimate(g, k) for g in group_lists]
        metrics[f"epoch_max_score_n{k}"] = sum(per_problem_best) / len(per_problem_best)
        k //= 2
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--dataset-size", type=int, required=True)
    parser.add_argument("--n-samples-per-prompt", type=int, default=8)
    parser.add_argument("--reward-key", default=None, help="key to extract from reward dict (if reward is dict)")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    samples_per_epoch = args.dataset_size * args.n_samples_per_prompt
    print(f"dataset_size={args.dataset_size}, n_samples_per_prompt={args.n_samples_per_prompt}")
    print(f"samples_per_epoch={samples_per_epoch}  (epoch boundary at index multiples of {samples_per_epoch})")
    print()

    pt_files = sorted(
        [f for f in rollout_dir.glob("*.pt") if not f.stem.startswith("eval_")],
        key=lambda f: int(f.stem),
    )
    print(f"Found {len(pt_files)} rollout files (rollout 0..{int(pt_files[-1].stem) if pt_files else -1})")
    print()

    # epoch_groups[epoch_id][label] = [rewards...]
    epoch_groups: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False)
        rollout_id = data["rollout_id"]
        samples = data["samples"]

        for s in samples:
            reward = s["reward"]
            if reward is None:
                continue
            if isinstance(reward, dict):
                reward = reward[args.reward_key] if args.reward_key else next(iter(reward.values()))
            reward = float(reward)
            label = str(s.get("label") or "unknown")
            # Determine which data epoch this sample belongs to
            sample_index = s["index"]
            data_epoch = sample_index // samples_per_epoch
            epoch_groups[data_epoch][label].append(reward)

    # Report per-epoch metrics
    for epoch_id in sorted(epoch_groups.keys()):
        groups = epoch_groups[epoch_id]
        all_rewards = [r for rs in groups.values() for r in rs]
        print(f"=== EPOCH {epoch_id + 1} ===")
        print(f"  problems covered : {len(groups)} / {args.dataset_size}")
        print(f"  total samples    : {len(all_rewards)}")

        metrics = compute_epoch_metrics(groups)
        print(f"  epoch_avg_score          = {metrics['epoch_avg_score']:.4f}")
        for key, val in sorted(metrics.items()):
            if key.startswith("epoch_max_score"):
                print(f"  {key:30s} = {val:.4f}")
        print()


if __name__ == "__main__":
    main()
