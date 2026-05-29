import wandb
from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step

_metrics_defined = False


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


def _best_of_halves(rewards: list[float]) -> dict[str, float]:
    metrics = {}
    k = len(rewards)
    while k >= 1:
        value = max(rewards) if k == len(rewards) else _best_of_k_estimate(rewards, k)
        metrics[f"frontiercs/max_score_n{k}"] = value
        k //= 2
    return metrics


def _reward_values(args, samples):
    values = []
    for sample in samples:
        reward = sample.get_reward_value(args)
        if reward is not None:
            values.append(float(reward))
    return values


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    _ensure_metrics_defined()
    rewards = _reward_values(args, samples)
    if not rewards:
        return False

    metrics = {
        "frontiercs/avg_score": sum(rewards) / len(rewards),
        "frontiercs/max_score": max(rewards),
        "rollout/step": compute_rollout_step(args, rollout_id),
    }
    metrics.update(_best_of_halves(rewards))
    logging_utils.log(args, metrics, step_key="rollout/step")

    # Keep the default rollout/perf logging in slime.ray.rollout._log_rollout_data.
    return False
