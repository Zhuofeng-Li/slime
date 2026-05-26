# Copyright 2025 - ALE-Bench reward for VERL RL training
"""ALE-Bench Docker judge reward for the full problem set via public eval."""

from __future__ import annotations

import atexit
import logging
import os
from typing import Any, Optional

from verl.utils.reward_score.frontiercs import extract_cpp

logger = logging.getLogger(__name__)

_LITE = False
_JUDGE_VERSION = os.environ.get("ALEBENCH_JUDGE_VERSION", "202301")
_CODE_LANGUAGE = "cpp17"
_NUM_WORKERS = int(os.environ.get("ALEBENCH_NUM_WORKERS", "4"))
_SESSION_DURATION_SECONDS = float(os.environ.get("ALEBENCH_PUBLIC_SESSION_DURATION_SECONDS", 365 * 24 * 3600))
_SESSION_CACHE: dict[str, Any] = {}


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


def _start_session(problem_id: str) -> Any:
    import ale_bench

    return ale_bench.start(
        problem_id=problem_id,
        lite_version=_LITE,
        num_workers=_NUM_WORKERS,
        run_visualization_server=False,
        session_duration=_SESSION_DURATION_SECONDS,
    )


def _get_session(problem_id: str) -> Any:
    session = _SESSION_CACHE.get(problem_id)
    if session is None:
        session = _start_session(problem_id)
        _SESSION_CACHE[problem_id] = session
    return session


def _close_cached_sessions() -> None:
    for problem_id, session in list(_SESSION_CACHE.items()):
        try:
            session.close()
        except Exception:
            logger.exception("ALE-Bench full cached session close failed for %s", problem_id)
    _SESSION_CACHE.clear()


atexit.register(_close_cached_sessions)


def _drop_session(problem_id: str) -> None:
    session = _SESSION_CACHE.pop(problem_id, None)
    if session is not None:
        try:
            session.close()
        except Exception:
            logger.exception("ALE-Bench full cached session close failed for %s", problem_id)


def _public_eval(problem_id: str, code: str) -> Any:
    session = _get_session(problem_id)
    return session.public_eval(
        code,
        _CODE_LANGUAGE,
        judge_version=_JUDGE_VERSION,
        skip_local_visualization=True,
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Evaluate C++ solution on ALE-Bench full train set using public eval."""
    if data_source != "alebench_full":
        raise ValueError(f"data_source must be 'alebench_full', got {data_source}")

    problem_id = str(ground_truth)
    code = extract_cpp(solution_str)
    if not code:
        return _empty_result()

    try:
        try:
            result = _public_eval(problem_id, code)
        except Exception as exc:
            if "session is finished" not in str(exc).lower():
                raise
            _drop_session(problem_id)
            result = _public_eval(problem_id, code)
        score = float(result.overall_absolute_score)
        return {
            "score": score,
            "performance": score,
            "rank": None,
            "overall_absolute_score": float(result.overall_absolute_score),
            "overall_relative_score": _optional_float(result.overall_relative_score),
        }
    except Exception:
        logger.exception("ALE-Bench full public eval failed for %s", problem_id)
        return _empty_result()
