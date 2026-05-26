# Copyright 2025 - Frontier-CS reward for VERL RL training
"""Frontier-CS judge reward: submit C++ code to judge API, return score."""

import re
import time
from typing import Any, Optional

import requests

DEFAULT_JUDGE_URL = "http://localhost:8082"
POLL_INTERVAL = 2.0
MAX_WAIT = 300.0
HARDTEST_PREFIXES = ("hardtest_smp_", "hardtest_orig_")
HARDTEST_FULL_SCORE = 100.0


def strip_think(response: str) -> str:
    """Remove everything through the last closing think tag."""
    if not response:
        return response
    _, sep, suffix = response.rpartition("</think>")
    return suffix if sep else response


def extract_cpp(response_text: str) -> str:
    """Extract C++ code from model response (markdown or raw), ignoring <think> blocks."""
    if not response_text:
        return ""

    response_text = strip_think(response_text)
    if not response_text:
        return ""

    code = response_text.strip()

    # Try to extract from ```cpp blocks
    cpp_pattern = r'```(?:cpp|c\+\+)?\s*\n(.*?)```'
    matches = re.findall(cpp_pattern, code, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Fallback: strip markdown if present
    if code.startswith("```cpp"):
        code = code[6:].strip()
    elif code.startswith("```c++"):
        code = code[6:].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    return code


def is_hardtest_problem(ground_truth: Any) -> bool:
    """Whether this problem id is one of the packaged hardtest variants."""
    return str(ground_truth).startswith(HARDTEST_PREFIXES)


def normalize_frontiercs_score(raw_score: float, ground_truth: Any) -> float:
    """Map hardtest judge scores to binary reward while preserving default behavior otherwise."""
    if is_hardtest_problem(ground_truth):
        return 100.0 if raw_score >= HARDTEST_FULL_SCORE else 0.0
    return raw_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    judge_url: Optional[str] = None,
    **kwargs,
) -> float:
    """
    Submit C++ solution to Frontier-CS judge and return score.

    Hardtest package ids (hardtest_smp_* / hardtest_orig_*) are treated as
    strict binary reward: only full score counts as 100, partial score counts
    as 0.0. All other Frontier-CS problems keep the original 0-100 score.

    Args:
        data_source: Must be "frontiercs"
        solution_str: Model's raw response (may contain ```cpp ... ```)
        ground_truth: problem_id (str)
        judge_url: Judge API base URL (default: http://localhost:8082)

    Returns:
        float: Reward score, or 0 on error.
    """
    if data_source != "frontiercs":
        raise ValueError(f"data_source must be 'frontiercs', got {data_source}")

    url = (judge_url or extra_info.get("judge_url") if extra_info else None) or DEFAULT_JUDGE_URL
    url = url.rstrip("/")
    problem_id = str(ground_truth)
    code = extract_cpp(solution_str)

    if not code:
        return 0.0

    try:
        r = requests.post(
            f"{url}/submit",
            files={"code": ("sol.cpp", code)},
            data={"pid": problem_id, "lang": "cpp"},
            timeout=30,
        )
        r.raise_for_status()
        sid = r.json().get("sid")
        if not sid:
            return 0.0

        start = time.time()
        while time.time() - start < MAX_WAIT:
            r2 = requests.get(f"{url}/result/{sid}", timeout=10)
            if r2.status_code == 404:
                time.sleep(POLL_INTERVAL)
                continue
            r2.raise_for_status()
            res = r2.json()
            status = res.get("status")
            if status == "done":
                raw_score = float(res.get("score", 0))
                return normalize_frontiercs_score(raw_score, problem_id)
            if status == "error":
                return 0.0
            time.sleep(POLL_INTERVAL)
        return 0.0
    except Exception:
        return 0.0
