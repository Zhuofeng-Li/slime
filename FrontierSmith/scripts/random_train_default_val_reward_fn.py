"""Random reward for tagged training rows, default task reward for validation rows."""

from __future__ import annotations

import os

import numpy as np

from verl.utils.reward_score import default_compute_score

DEFAULT_RANDOM_REWARD_DATA_SOURCES = "frontiercs_random_reward,random_reward"

_rng: np.random.Generator | None = None


def _rng_get() -> np.random.Generator:
    global _rng
    if _rng is None:
        seed = int(os.environ.get("RANDOM_REWARD_SEED", "0"))
        _rng = np.random.default_rng(seed)
    return _rng


def _random_reward_data_sources() -> set[str]:
    raw_value = os.environ.get("RANDOM_REWARD_DATA_SOURCES", DEFAULT_RANDOM_REWARD_DATA_SOURCES)
    return {item.strip() for item in raw_value.split(",") if item.strip()}


def _uses_random_reward(data_source: str) -> bool:
    data_source = str(data_source)
    return data_source in _random_reward_data_sources() or data_source.endswith("_random_reward")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Use random training rewards and the normal VERL reward for validation data."""
    if _uses_random_reward(data_source):
        return float(_rng_get().random())

    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
