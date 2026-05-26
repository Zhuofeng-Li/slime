"""
FrontierCS reward adapter for SLIME.

Wraps the synchronous Frontier-CS judge (HTTP polling) in an async batched
interface that matches SLIME's --custom-rm-path contract:

    async def batched_custom_rm(args, samples: list[Sample]) -> list[float]

Judge must be running at DEFAULT_JUDGE_URL (default: http://localhost:8082).
Override via FRONTIER_JUDGE_URL env var or per-sample metadata["judge_url"].

Scores are returned in [0, 1] (raw 0-100 score divided by 100).
Hardtest problems (hardtest_smp_* / hardtest_orig_*) get binary reward:
  full score (100) -> 1.0, anything else -> 0.0

Usage (set in run script):
  --custom-rm-path FrontierSmith.slime_rm.frontiercs_rm.batched_custom_rm
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from slime.utils.types import Sample

DEFAULT_JUDGE_URL = os.environ.get("FRONTIER_JUDGE_URL", "http://localhost:8082")
POLL_INTERVAL = 2.0
MAX_WAIT = 300.0
HARDTEST_PREFIXES = ("hardtest_smp_", "hardtest_orig_")


# ── Code extraction (mirrors FrontierSmith/verl reward) ──────────────────

def _strip_think(text: str) -> str:
    _, sep, suffix = text.rpartition("</think>")
    return suffix if sep else text


def _extract_cpp(response: str) -> str:
    if not response:
        return ""
    response = _strip_think(response).strip()
    matches = re.findall(r'```(?:cpp|c\+\+)?\s*\n(.*?)```', response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    for prefix in ("```cpp", "```c++", "```"):
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
            break
    if response.endswith("```"):
        response = response[:-3].strip()
    return response


# ── Judge interaction ─────────────────────────────────────────────────────

def _compute_score_sync(problem_id: str, solution_str: str, judge_url: str) -> float:
    code = _extract_cpp(solution_str)
    if not code:
        return 0.0

    url = judge_url.rstrip("/")
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

        deadline = time.time() + MAX_WAIT
        while time.time() < deadline:
            r2 = requests.get(f"{url}/result/{sid}", timeout=10)
            if r2.status_code == 404:
                time.sleep(POLL_INTERVAL)
                continue
            r2.raise_for_status()
            res = r2.json()
            status = res.get("status")
            if status == "done":
                raw = float(res.get("score", 0))
                # hardtest: binary reward
                if str(problem_id).startswith(HARDTEST_PREFIXES):
                    return 1.0 if raw >= 100.0 else 0.0
                return raw / 100.0
            if status == "error":
                return 0.0
            time.sleep(POLL_INTERVAL)
    except Exception:
        pass
    return 0.0


async def _score_one(sample: "Sample") -> float:
    problem_id = sample.label or ""
    judge_url = (
        (sample.metadata or {}).get("judge_url") or DEFAULT_JUDGE_URL
    )
    return await asyncio.to_thread(_compute_score_sync, problem_id, sample.response, judge_url)


# ── SLIME batched interface ───────────────────────────────────────────────

async def batched_custom_rm(args: Any, samples) -> "list[float] | float":
    """Score samples against the Frontier-CS judge.

    Accepts either a single Sample (called via async_rm) or a list of Samples
    (called via batched_async_rm).
    """
    if not isinstance(samples, list):
        return await _score_one(samples)
    tasks = [_score_one(s) for s in samples]
    return list(await asyncio.gather(*tasks))
