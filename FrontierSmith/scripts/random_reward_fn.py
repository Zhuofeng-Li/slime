"""Random reward for VERL custom_reward_function (ignores task / ground truth)."""

from __future__ import annotations

import os

import numpy as np

_rng: np.random.Generator | None = None


def _rng_get() -> np.random.Generator:
    global _rng
    if _rng is None:
        seed = int(os.environ.get("RANDOM_REWARD_SEED", "0"))
        _rng = np.random.default_rng(seed)
    return _rng


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Return a random score in [0, 1). Same signature as other VERL reward fns."""
    return float(_rng_get().random())
