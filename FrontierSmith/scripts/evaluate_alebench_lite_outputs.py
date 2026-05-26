#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NODE_ORDER = [
    "baseline",
    "train_on_frontiercs",
    "train_on_synthetic",
    "train_on_hardtest",
]

PROBLEM_ORDER = [
    "ahc008",
    "ahc011",
    "ahc015",
    "ahc016",
    "ahc024",
    "ahc025",
    "ahc026",
    "ahc027",
    "ahc039",
    "ahc046",
]


@dataclass(frozen=True)
class SampleTask:
    node: str
    problem_id: str
    sample_index: int
    sample_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path(
            "eval_work/alebench_output_unpack/outputs/alebench_lite_qwen35_27b_oneshot/samples"
        ),
    )
    parser.add_argument("--results-root", type=Path, default=Path("eval_work/alebench_private_eval"))
    parser.add_argument("--ale-bench-src", type=Path, default=Path("ALE-Bench/src"))
    parser.add_argument("--ale-bench-data", type=Path, default=Path("data/alebench/local_data"))
    parser.add_argument("--ale-bench-cache", type=Path, default=Path("eval_work/alebench_cache"))
    parser.add_argument("--nodes", nargs="*", default=NODE_ORDER)
    parser.add_argument("--problems", nargs="*", default=PROBLEM_ORDER)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--judge-version", default="202301")
    parser.add_argument("--code-language", default="cpp20")
    parser.add_argument("--extractor", choices=["robust", "verl"], default="robust")
    parser.add_argument(
        "--verl-frontiercs-path",
        type=Path,
        default=Path("verl/verl/utils/reward_score/frontiercs.py"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_ale_bench_to_path(src: Path) -> None:
    src = src.resolve()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def extract_code(response_text: str) -> tuple[str, dict[str, Any]]:
    text = response_text or ""
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []

    closed_fence_re = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    for idx, match in enumerate(closed_fence_re.finditer(text)):
        lang = match.group(1).strip().lower()
        body = match.group(2).strip()
        quality = 0
        if any(token in lang for token in ("cpp", "c++", "cxx")) or lang == "":
            quality += 100
        if "#include" in body:
            quality += 30
        if re.search(r"\bmain\s*\(", body):
            quality += 30
        if "using namespace std" in body:
            quality += 5
        candidates.append((quality, idx, body, {"method": "closed_fence", "language": lang, "fence_index": idx}))

    open_fence_re = re.compile(r"```([^\n`]*)\n", re.DOTALL)
    for idx, match in enumerate(open_fence_re.finditer(text)):
        lang = match.group(1).strip().lower()
        if "```" in text[match.end() :]:
            continue
        body = text[match.end() :].strip()
        quality = 0
        if any(token in lang for token in ("cpp", "c++", "cxx")) or lang == "":
            quality += 80
        if "#include" in body:
            quality += 30
        if re.search(r"\bmain\s*\(", body):
            quality += 30
        candidates.append((quality, idx, body, {"method": "open_fence", "language": lang, "fence_index": idx}))

    include_idx = text.find("#include")
    if include_idx >= 0:
        body = text[include_idx:].strip()
        body = body.split("</final>", 1)[0].strip()
        candidates.append((50, 10_000, body, {"method": "from_first_include", "include_index": include_idx}))

    if candidates:
        quality, _, code, meta = max(candidates, key=lambda item: (item[0], item[1], len(item[2])))
        meta = {
            **meta,
            "quality": quality,
            "code_chars": len(code),
            "has_include": "#include" in code,
            "has_main": bool(re.search(r"\bmain\s*\(", code)),
        }
        return code, meta

    code = text.strip()
    return code, {
        "method": "whole_response",
        "quality": 0,
        "code_chars": len(code),
        "has_include": "#include" in code,
        "has_main": bool(re.search(r"\bmain\s*\(", code)),
    }


_VERL_EXTRACT_CPP = None


def load_verl_extract_cpp(frontiercs_path: Path):
    global _VERL_EXTRACT_CPP
    if _VERL_EXTRACT_CPP is not None:
        return _VERL_EXTRACT_CPP
    frontiercs_path = frontiercs_path.resolve()
    spec = importlib.util.spec_from_file_location("verl_frontiercs_reward_extract_only", frontiercs_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load VERL frontiercs extractor from {frontiercs_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _VERL_EXTRACT_CPP = module.extract_cpp
    return _VERL_EXTRACT_CPP


def extract_code_verl(response_text: str, frontiercs_path: Path) -> tuple[str, dict[str, Any]]:
    extract_cpp = load_verl_extract_cpp(frontiercs_path)
    code = extract_cpp(response_text or "")
    return code, {
        "method": "verl.frontiercs.extract_cpp",
        "code_chars": len(code),
        "has_include": "#include" in code,
        "has_main": bool(re.search(r"\bmain\s*\(", code)),
    }


def discover_tasks(samples_root: Path, nodes: list[str], problems: list[str]) -> list[SampleTask]:
    tasks: list[SampleTask] = []
    for problem_id in problems:
        for node in nodes:
            problem_dir = samples_root / node / problem_id
            for sample_path in sorted(problem_dir.glob("sample_*.json")):
                m = re.search(r"sample_(\d+)\.json$", sample_path.name)
                if not m:
                    continue
                tasks.append(
                    SampleTask(
                        node=node,
                        problem_id=problem_id,
                        sample_index=int(m.group(1)),
                        sample_path=sample_path,
                    )
                )
    return tasks


def sample_result_path(results_root: Path, task: SampleTask) -> Path:
    return results_root / "samples" / task.node / task.problem_id / f"sample_{task.sample_index:02d}.json"


def case_status_counts(private_result: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in private_result.case_results:
        key = str(case.judge_result.value if hasattr(case.judge_result, "value") else case.judge_result)
        counts[key] = counts.get(key, 0) + 1
    return counts


def model_dump_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def evaluate_task(
    session: Any,
    task: SampleTask,
    results_root: Path,
    judge_version: str,
    code_language: str,
    extractor: str,
    verl_frontiercs_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    out_path = sample_result_path(results_root, task)
    if out_path.exists() and not overwrite:
        loaded = read_json(out_path)
        loaded["_loaded_from_cache"] = True
        return loaded

    sample = read_json(task.sample_path)
    response_text = (((sample.get("response") or {}).get("text")) or "")
    if extractor == "verl":
        code, extraction = extract_code_verl(response_text, verl_frontiercs_path)
    else:
        code, extraction = extract_code(response_text)

    started = time.time()
    result: dict[str, Any] = {
        "node": task.node,
        "problem_id": task.problem_id,
        "sample_index": task.sample_index,
        "source_sample_path": str(task.sample_path),
        "source_task_id": sample.get("task_id"),
        "source_model": sample.get("model"),
        "code_language": code_language,
        "judge_version": judge_version,
        "extractor": extractor,
        "code_extraction": extraction,
        "ok": False,
    }
    try:
        private_result, rank, performance = session.private_eval(
            code,
            code_language=code_language,
            judge_version=judge_version,
        )
        result.update(
            {
                "ok": True,
                "overall_judge_result": private_result.overall_judge_result.value,
                "overall_absolute_score": private_result.overall_absolute_score,
                "overall_relative_score": private_result.overall_relative_score,
                "rank": rank,
                "performance": performance,
                "case_status_counts": case_status_counts(private_result),
                "private_result": model_dump_jsonable(private_result),
            }
        )
    except Exception as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "overall_judge_result": None,
                "overall_absolute_score": None,
                "overall_relative_score": None,
                "rank": None,
                "performance": None,
            }
        )
    finally:
        # ALE-Bench sessions allow only one private_eval by default. For offline
        # evaluation of independent pre-generated samples, reset only that quota.
        try:
            from ale_bench.result import ResourceUsage

            usage = session.current_resource_usage
            session._current_resource_usage = ResourceUsage(  # noqa: SLF001
                num_case_gen=usage.num_case_gen,
                num_case_eval=usage.num_case_eval,
                num_call_public_eval=usage.num_call_public_eval,
                num_call_private_eval=0,
                execution_time_case_eval=usage.execution_time_case_eval,
            )
        except Exception:
            pass

    result["elapsed_s"] = time.time() - started
    write_json(out_path, result)
    return result


def mean_or_none(values: list[float | int | None]) -> float | None:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return None
    return float(statistics.mean(clean))


def best_result_for_problem(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    clean = [r for r in results if isinstance(r.get("performance"), (int, float))]
    if not clean:
        return None
    return max(clean, key=lambda r: (r["performance"], -(r.get("rank") or 10**18), r.get("overall_absolute_score") or 0))


def summarize(results_root: Path, nodes: list[str], problems: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sample_summary_path = results_root / "sample_results.jsonl"
    if sample_summary_path.exists():
        sample_summary_path.unlink()

    for node in nodes:
        for problem_id in problems:
            sample_results = []
            for path in sorted((results_root / "samples" / node / problem_id).glob("sample_*.json")):
                item = read_json(path)
                item.pop("private_result", None)
                sample_results.append(item)
                append_jsonl(sample_summary_path, item)
            best = best_result_for_problem(sample_results)
            performances = [r.get("performance") for r in sample_results]
            ranks = [r.get("rank") for r in sample_results]
            problem_row = {
                "node": node,
                "problem_id": problem_id,
                "num_samples": len(sample_results),
                "num_ok": sum(1 for r in sample_results if r.get("ok")),
                "avg_at_5_performance": mean_or_none(performances),
                "best_at_5_performance": best.get("performance") if best else None,
                "best_sample_index": best.get("sample_index") if best else None,
                "avg_at_5_rank": mean_or_none(ranks),
                "best_at_5_rank": best.get("rank") if best else None,
                "judge_counts": {},
            }
            for r in sample_results:
                jr = r.get("overall_judge_result") or "ERROR"
                problem_row["judge_counts"][jr] = problem_row["judge_counts"].get(jr, 0) + 1
            rows.append(problem_row)

    node_rows = []
    for node in nodes:
        node_problem_rows = [r for r in rows if r["node"] == node]
        node_rows.append(
            {
                "node": node,
                "num_problems": len(node_problem_rows),
                "avg_at_5": mean_or_none([r["avg_at_5_performance"] for r in node_problem_rows]),
                "best_at_5": mean_or_none([r["best_at_5_performance"] for r in node_problem_rows]),
                "avg_rank_at_5": mean_or_none([r["avg_at_5_rank"] for r in node_problem_rows]),
                "best_rank_at_5": mean_or_none([r["best_at_5_rank"] for r in node_problem_rows]),
                "num_ok_samples": sum(r["num_ok"] for r in node_problem_rows),
                "num_samples": sum(r["num_samples"] for r in node_problem_rows),
            }
        )

    summary = {"problem_summary": rows, "node_summary": node_rows}
    write_json(results_root / "problem_summary.json", rows)
    write_json(results_root / "node_summary.json", node_rows)
    write_json(results_root / "summary.json", summary)
    return summary


def print_node_table(node_summary: list[dict[str, Any]]) -> None:
    print("\nnode,avg@5,best@5,avg_rank@5,best_rank@5,ok/samples", flush=True)
    for row in node_summary:
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.3f}"

        print(
            ",".join(
                [
                    row["node"],
                    fmt(row["avg_at_5"]),
                    fmt(row["best_at_5"]),
                    fmt(row["avg_rank_at_5"]),
                    fmt(row["best_rank_at_5"]),
                    f"{row['num_ok_samples']}/{row['num_samples']}",
                ]
            ),
            flush=True,
        )


def main() -> int:
    args = parse_args()
    add_ale_bench_to_path(args.ale_bench_src)
    os.environ["ALE_BENCH_DATA"] = str(args.ale_bench_data.resolve())
    os.environ["ALE_BENCH_CACHE"] = str(args.ale_bench_cache.resolve())
    args.ale_bench_cache.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)

    import ale_bench

    tasks = discover_tasks(args.samples_root, args.nodes, args.problems)
    expected = len(args.nodes) * len(args.problems) * 5
    print(f"Discovered {len(tasks)} sample tasks; expected around {expected}.", flush=True)

    tasks_by_problem: dict[str, list[SampleTask]] = {problem_id: [] for problem_id in args.problems}
    for task in tasks:
        tasks_by_problem.setdefault(task.problem_id, []).append(task)

    manifest = {
        "samples_root": str(args.samples_root),
        "results_root": str(args.results_root),
        "nodes": args.nodes,
        "problems": args.problems,
        "num_workers": args.num_workers,
        "judge_version": args.judge_version,
        "code_language": args.code_language,
        "extractor": args.extractor,
        "verl_frontiercs_path": str(args.verl_frontiercs_path),
        "ale_bench_data": os.environ["ALE_BENCH_DATA"],
        "ale_bench_cache": os.environ["ALE_BENCH_CACHE"],
        "overwrite": args.overwrite,
    }
    write_json(args.results_root / "run_config.json", manifest)

    progress_path = args.results_root / "progress.jsonl"
    for problem_id in args.problems:
        problem_tasks = tasks_by_problem.get(problem_id, [])
        if not problem_tasks:
            print(f"[{problem_id}] no tasks found, skipping", flush=True)
            continue

        print(f"[{problem_id}] starting session for {len(problem_tasks)} samples", flush=True)
        session = ale_bench.start(
            problem_id,
            lite_version=True,
            num_workers=args.num_workers,
            run_visualization_server=False,
        )
        try:
            for idx, task in enumerate(problem_tasks, start=1):
                out_path = sample_result_path(args.results_root, task)
                cached = out_path.exists() and not args.overwrite
                print(
                    f"[{problem_id}] {idx:02d}/{len(problem_tasks):02d} "
                    f"{task.node} sample_{task.sample_index:02d}"
                    f"{' cached' if cached else ''}",
                    flush=True,
                )
                result = evaluate_task(
                    session=session,
                    task=task,
                    results_root=args.results_root,
                    judge_version=args.judge_version,
                    code_language=args.code_language,
                    extractor=args.extractor,
                    verl_frontiercs_path=args.verl_frontiercs_path,
                    overwrite=args.overwrite,
                )
                append_jsonl(
                    progress_path,
                    {
                        "problem_id": problem_id,
                        "node": task.node,
                        "sample_index": task.sample_index,
                        "ok": result.get("ok"),
                        "judge": result.get("overall_judge_result"),
                        "performance": result.get("performance"),
                        "rank": result.get("rank"),
                        "cached": result.get("_loaded_from_cache", False),
                    },
                )
        finally:
            session.close()

    summary = summarize(args.results_root, args.nodes, args.problems)
    print_node_table(summary["node_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
