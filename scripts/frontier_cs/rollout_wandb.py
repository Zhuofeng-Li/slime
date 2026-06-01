from collections import defaultdict

import math

import wandb
from slime.utils.metric_utils import compute_rollout_step

_metrics_defined = False
_dataset_size: int | None = None
_epoch_groups: dict[str, list[float]] = defaultdict(list)


def _get_dataset_size(args) -> int:
    global _dataset_size
    if _dataset_size is None:
        with open(args.prompt_data) as f:
            _dataset_size = sum(1 for _ in f)
    return _dataset_size


def _ensure_metrics_defined():
    global _metrics_defined
    if not _metrics_defined and wandb.run is not None:
        wandb.define_metric("frontiercs/*", step_metric="rollout/step")
        _metrics_defined = True


def _best_of_k_estimate(rewards: list[float], k: int) -> float:
    sorted_rewards = sorted(rewards)
    n = len(sorted_rewards)
    expected = 0.0
    prev_cdf = 0.0
    for i, reward in enumerate(sorted_rewards, start=1):
        cdf = (i / n) ** k
        expected += reward * (cdf - prev_cdf)
        prev_cdf = cdf
    return expected



def _best_of_halves(rewards: list[float], prefix: str) -> dict[str, float]:
    metrics = {}
    k = len(rewards)
    while k >= 1:
        value = max(rewards) if k == len(rewards) else _best_of_k_estimate(rewards, k)
        metrics[f"{prefix}/max_score_n{k}"] = value
        k //= 2
    return metrics


def _reward_values(args, samples):
    values = []
    for sample in samples:
        reward = sample.get_reward_value(args)
        if reward is not None:
            values.append(float(reward))
    return values


def _group_rewards_by_label(args, samples) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        reward = sample.get_reward_value(args)
        if reward is not None:
            label = sample.label or "unknown"
            groups[label].append(float(reward))
    return groups


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    _ensure_metrics_defined()
    rewards = _reward_values(args, samples)
    if not rewards:
        return False

    # # Overall metrics
    # metrics = {
    #     "frontiercs/avg_score": sum(rewards) / len(rewards),
    #     "frontiercs/max_score": max(rewards),
    # }
    # metrics.update(_best_of_halves(rewards, prefix="frontiercs"))
    metrics = {}

    # Per-problem metrics: only log problems that appear in this rollout,
    # and only max_score + avg_score (no best-of-k expansion).
    groups = _group_rewards_by_label(args, samples)
    for label, group_rewards in groups.items():
        prefix = f"frontiercs/by_problem/{label}"
        metrics[f"{prefix}/avg_score"] = sum(group_rewards) / len(group_rewards)
        metrics[f"{prefix}/max_score"] = max(group_rewards)

    # Epoch-level metrics: accumulate per-problem rewards, flush once per epoch.
    global _epoch_groups
    for label, group_rewards in groups.items():
        _epoch_groups[label].extend(group_rewards)
    dataset_size = _get_dataset_size(args)
    steps_per_epoch = max(1, math.ceil(dataset_size / args.rollout_batch_size))
    # rollout_id is 0-indexed; flush at the last rollout of each epoch
    if (rollout_id + 1) % steps_per_epoch == 0:
        # Split each problem's rewards into [this epoch | carry-over for next epoch].
        # Boundary rollouts may contain samples from both epochs; each problem
        # should contribute exactly n_samples_per_prompt rewards per epoch.
        n = args.n_samples_per_prompt
        flush_groups: dict[str, list[float]] = {}
        carry_over: dict[str, list[float]] = defaultdict(list)
        for label, reward_list in _epoch_groups.items():
            flush_groups[label] = reward_list[:n]
            if len(reward_list) > n:
                carry_over[label] = reward_list[n:]

        # Sanity check: total flushed == dataset_size * n_samples_per_prompt
        total_flushed = sum(len(g) for g in flush_groups.values())
        assert total_flushed == dataset_size * n, (
            f"epoch flush count mismatch: got {total_flushed}, "
            f"expected {dataset_size * n}"
        )

        group_lists = list(flush_groups.values())
        per_problem_avg = [sum(g) / len(g) for g in group_lists]
        metrics["frontiercs/epoch_avg_score"] = sum(per_problem_avg) / len(per_problem_avg)
        min_n = min(len(g) for g in group_lists)
        k = min_n
        while k >= 1:
            if k == min_n:
                per_problem_best = [max(g) for g in group_lists]
            else:
                per_problem_best = [_best_of_k_estimate(g, k) for g in group_lists]
            metrics[f"frontiercs/epoch_max_score_n{k}"] = sum(per_problem_best) / len(per_problem_best)
            k //= 2
        _epoch_groups = carry_over

    # Buffer frontiercs metrics without committing — the default _log_rollout_data
    # will call wandb.log() with rollout/step and commit=True, so both sets of
    # metrics land in the same wandb step and the step counter advances only once.
    if args.use_wandb and wandb.run is not None:
        wandb.log(metrics, commit=False)

    # Return False so the default rollout/perf logging still runs.
    return False
