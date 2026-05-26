# Copyright 2025 - ALE-Bench reward for VERL RL training (validation only)
"""ALE-Bench Docker judge reward: run C++ code via ALE-Bench private eval."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from verl.utils.reward_score.frontiercs import extract_cpp

logger = logging.getLogger(__name__)

_LITE = os.environ.get("ALEBENCH_LITE", "true").lower() in ("1", "true")
_JUDGE_VERSION = os.environ.get("ALEBENCH_JUDGE_VERSION", "202301")
_CODE_LANGUAGE = "cpp17"
_NUM_WORKERS = int(os.environ.get("ALEBENCH_NUM_WORKERS", "4"))


def _empty_result() -> dict[str, float | None]:
    return {
        "score": 0.0,
        "performance": 0.0,
        "rank": None,
        "overall_absolute_score": 0.0,
        "overall_relative_score": None,
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _start_session(problem_id: str):
    import ale_bench

    # private_eval is limited to one call per session, so each scored sample
    # must get a fresh session instead of reusing a cached public-eval session.
    return ale_bench.start(
        problem_id=problem_id,
        lite_version=_LITE,
        num_workers=_NUM_WORKERS,
        run_visualization_server=False,
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Evaluate C++ solution on ALE-Bench and return native ALE metrics.

    Returns dict with:
        score: ALE performance, used as the scalar reward/validation core metric
        performance: same as score, for explicit logging
        rank: estimated standings rank, lower is better
        overall_absolute_score: private absolute score
        overall_relative_score: private relative score when available
    """
    if data_source != "alebench":
        raise ValueError(f"data_source must be 'alebench', got {data_source}")

    problem_id = str(ground_truth)
    code = extract_cpp(solution_str)
    if not code:
        return _empty_result()

    session = None
    try:
        session = _start_session(problem_id)
        result, rank, performance = session.private_eval(code, _CODE_LANGUAGE, judge_version=_JUDGE_VERSION)
        performance = float(performance)
        return {
            "score": performance,
            "performance": performance,
            "rank": float(rank),
            "overall_absolute_score": float(result.overall_absolute_score),
            "overall_relative_score": _optional_float(result.overall_relative_score),
        }
    except Exception:
        logger.exception("ALE-Bench eval failed for %s", problem_id)
        return _empty_result()
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                logger.exception("ALE-Bench session close failed for %s", problem_id)
