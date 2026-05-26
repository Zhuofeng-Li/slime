#!/usr/bin/env python3
"""Run ALE-Bench private evaluation for a directory of C++ solutions.

This helper is intended to run inside an ALE Harbor task's `main` container,
where `/opt/ALE-Bench` and `/opt/ale-bench-data` are already mounted by the
existing Harbor environment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import traceback
from itertools import combinations
from pathlib import Path
from typing import Any


def clipped01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return min(1.0, max(0.0, v))


def normalized_pairwise_l2(vectors: list[list[float]]) -> dict[str, Any]:
    n = len(vectors)
    if n < 2:
        return {
            "num_solutions": n,
            "num_pairs": 0,
            "m": len(vectors[0]) if vectors else 0,
            "pairwise_normalized_L2": [],
            "avg_normalized_pairwise_L2": None,
            "avg_pairwise_L2": None,
            "effective_avg_pairwise_L2": None,
        }

    m = max(len(v) for v in vectors)
    if m == 0:
        return {
            "num_solutions": n,
            "num_pairs": 0,
            "m": 0,
            "pairwise_normalized_L2": [],
            "avg_normalized_pairwise_L2": None,
            "avg_pairwise_L2": None,
            "effective_avg_pairwise_L2": None,
        }

    padded = [v + [0.0] * (m - len(v)) for v in vectors]
    distances = []
    for i, j in combinations(range(n), 2):
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(padded[i], padded[j])) / m)
        distances.append(round(d, 6))
    return {
        "num_solutions": n,
        "num_pairs": len(distances),
        "m": m,
        "pairwise_normalized_L2": distances,
        "avg_normalized_pairwise_L2": round(sum(distances) / len(distances), 6),
        "avg_pairwise_L2": round(sum(distances) / len(distances), 6),
        "effective_avg_pairwise_L2": round(sum(distances) / len(distances), 6),
    }


def case_to_dict(case: Any) -> dict[str, Any]:
    judge_result = getattr(case, "judge_result", None)
    return {
        "judge_result": getattr(judge_result, "value", str(judge_result)),
        "absolute_score": getattr(case, "absolute_score", 0),
        "relative_score": getattr(case, "relative_score", None),
        "execution_time": getattr(case, "execution_time", 0),
        "memory_usage": getattr(case, "memory_usage", 0),
    }


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def image_exists(name: str) -> bool:
    return run(["docker", "image", "inspect", name], check=False).returncode == 0


def ensure_image(alias: str, source: str | None = None) -> None:
    if image_exists(alias):
        return
    if source is None:
        source = alias
    print(f"Pulling {source}", file=sys.stderr, flush=True)
    print(run(["docker", "pull", source], check=True).stdout, file=sys.stderr)
    if source != alias:
        print(f"Tagging {source} -> {alias}", file=sys.stderr, flush=True)
        run(["docker", "tag", source, alias], check=True)


def ensure_docker_images(code_language: str, judge_version: str) -> None:
    judge_alias = f"ale-bench:{code_language}-{judge_version}"
    judge_source = f"yimjk/ale-bench:{code_language}-{judge_version}"
    ensure_image(judge_alias, judge_source)
    ensure_image("rust:1.79.0-buster")


def result_to_record(solution_path: Path, result: Any, rank: int, performance: int) -> dict[str, Any]:
    judge_result = getattr(result.overall_judge_result, "value", str(result.overall_judge_result))
    cases = [case_to_dict(case) for case in result.case_results]
    return {
        "path": str(solution_path),
        "status": "done",
        "overall_judge_result": judge_result,
        "overall_absolute_score": result.overall_absolute_score,
        "overall_relative_score": result.overall_relative_score,
        "rank": rank,
        "performance": performance,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALE private eval for a solution directory.")
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--solutions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-language", default="cpp20")
    parser.add_argument("--judge-version", default="202301")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--session-hours", type=float, default=12.0)
    parser.add_argument("--relative-max-score", type=float, default=None)
    args = parser.parse_args()

    ensure_docker_images(args.code_language, args.judge_version)

    import ale_bench
    from ale_bench.result import ResourceUsage

    solutions = sorted(args.solutions_dir.glob("*.cpp"))
    records: list[dict[str, Any]] = []

    session = ale_bench.start(
        problem_id=args.problem_id,
        lite_version=True,
        use_same_time_scale=False,
        maximum_num_call_public_eval=0,
        session_duration=dt.timedelta(hours=args.session_hours),
        num_workers=args.num_workers,
        run_visualization_server=False,
    )
    try:
        # ALE-Bench defaults to one private evaluation per session. This helper
        # is not simulating a contest agent; it is an offline scoring pass over
        # a fixed solution pool, so raise only this resource cap.
        session._maximum_resource_usage = ResourceUsage(  # noqa: SLF001
            num_case_gen=session._maximum_resource_usage.num_case_gen,  # noqa: SLF001
            num_case_eval=session._maximum_resource_usage.num_case_eval,  # noqa: SLF001
            num_call_public_eval=session._maximum_resource_usage.num_call_public_eval,  # noqa: SLF001
            num_call_private_eval=max(1, len(solutions)),
            execution_time_case_eval=session._maximum_resource_usage.execution_time_case_eval,  # noqa: SLF001
        )

        for solution in solutions:
            try:
                code = solution.read_text(encoding="utf-8")
                result, rank, performance = session.private_eval(
                    code,
                    code_language=args.code_language,
                    judge_version=args.judge_version,
                )
                records.append(result_to_record(solution, result, rank, performance))
            except Exception as exc:
                records.append(
                    {
                        "path": str(solution),
                        "status": "error",
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=20),
                        "cases": [],
                    }
                )
    finally:
        session.close()

    raw_vectors: list[list[float]] = []
    normalized_vectors: list[list[float]] = []
    vector_source = "relative_score"
    scale = args.relative_max_score
    if scale is None or scale <= 0:
        vector_source = "casewise_max_absolute_score"

    # If relative_max_score is unavailable, fall back to case-wise max absolute
    # score across this fixed pool. This keeps each case on a comparable [0,1]
    # scale without relying on raw AHC magnitudes.
    casewise_max: list[float] = []
    if vector_source == "casewise_max_absolute_score":
        max_m = max((len(r.get("cases") or []) for r in records if r.get("status") == "done"), default=0)
        casewise_max = [0.0] * max_m
        for record in records:
            if record.get("status") != "done":
                continue
            for idx, case in enumerate(record.get("cases") or []):
                try:
                    casewise_max[idx] = max(casewise_max[idx], float(case.get("absolute_score") or 0.0))
                except Exception:
                    pass

    for record in records:
        if record.get("status") != "done":
            continue
        raw_vec = []
        norm_vec = []
        for idx, case in enumerate(record.get("cases") or []):
            if vector_source == "relative_score":
                raw = case.get("relative_score")
                if raw is None:
                    raw = case.get("absolute_score", 0.0)
                try:
                    norm = float(raw) / float(scale)
                except Exception:
                    norm = 0.0
            else:
                raw = case.get("absolute_score", 0.0)
                denom = casewise_max[idx] if idx < len(casewise_max) else 0.0
                try:
                    norm = float(raw) / denom if denom > 0 else 0.0
                except Exception:
                    norm = 0.0
            try:
                raw_vec.append(float(raw or 0.0))
            except Exception:
                raw_vec.append(0.0)
            norm_vec.append(clipped01(norm))
        raw_vectors.append(raw_vec)
        normalized_vectors.append(norm_vec)

    quality = normalized_pairwise_l2(normalized_vectors)
    quality.update(
        {
            "metric": "avg_pairwise_l2_div_sqrt_m",
            "vector_source": vector_source,
            "relative_max_score": args.relative_max_score,
            "num_attempted_solutions": len(solutions),
            "num_done_solutions": len(normalized_vectors),
        }
    )

    payload = {
        "problem_id": args.problem_id,
        "code_language": args.code_language,
        "judge_version": args.judge_version,
        "raw_score_vectors": raw_vectors,
        "normalized_score_vectors": normalized_vectors,
        "judge_results": records,
        "quality": quality,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
