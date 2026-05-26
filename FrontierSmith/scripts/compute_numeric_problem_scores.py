#!/usr/bin/env python3
"""
Compute score@1 and pass@1 for models, considering only problems with numeric IDs (编号).

Reads results from results/batch/algorithmic/results.csv (or --results-dir).
Outputs by_model.csv with columns: model, score_at_1, pass_at_1, num_problems.

Usage:
  python scripts/compute_numeric_problem_scores.py
  python scripts/compute_numeric_problem_scores.py --results-dir results/batch/algorithmic
  python scripts/compute_numeric_problem_scores.py --output my_by_model.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "batch" / "algorithmic"


def is_numeric_problem(problem_id: str) -> bool:
    """True if problem ID is a pure number (编号)."""
    s = str(problem_id).strip()
    return s.isdigit() if s else False


def parse_solution_path(solution: str) -> tuple[str, str, int]:
    """Extract (problem_dir, model, variant) from path like '0/qwenqwen3527b.cpp' or '0/qwenqwen3527b_1.cpp'."""
    parts = solution.split("/")
    if len(parts) >= 2:
        problem_dir = parts[0]
        filename = parts[-1]
    else:
        problem_dir = ""
        filename = parts[-1] if parts else ""
    stem = Path(filename).stem
    if "_" in stem and stem.rsplit("_", 1)[-1].isdigit():
        model = stem.rsplit("_", 1)[0]
        variant = int(stem.rsplit("_", 1)[-1])
    else:
        model = stem
        variant = 0
    return problem_dir, model, variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute score@1 and pass@1 for numeric-ID problems only")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Results directory containing results.csv")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output CSV path (default: {results-dir}/by_model_numeric.csv)")
    args = parser.parse_args()

    results_csv = args.results_dir / "results.csv"
    if not results_csv.exists():
        print(f"Error: {results_csv} not found")
        return

    if args.output is None:
        args.output = args.results_dir / "by_model.csv"

    # model -> {problem -> score} for variant 0 only (score@1)
    by_model: dict[str, dict[str, float]] = defaultdict(dict)

    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem = row.get("problem", "").strip()
            if not is_numeric_problem(problem):
                continue

            solution = row.get("solution", "")
            _problem_dir, model, variant = parse_solution_path(solution)

            if variant != 0:
                continue  # score@1 uses only variant 0

            status = row.get("status", "").strip().lower()
            score_str = row.get("score", "").strip()

            if status == "success" and score_str:
                try:
                    score = float(score_str)
                    score = max(0, min(100, score))
                except ValueError:
                    score = 0.0
            else:
                score = 0.0

            by_model[model][problem] = score

    rows = []
    for model in sorted(by_model.keys()):
        problem_scores = by_model[model]
        scores = list(problem_scores.values())
        if not scores:
            continue
        score_at_1 = sum(scores) / len(scores)
        pass_at_1 = sum(1 for s in scores if s > 0) / len(scores)
        rows.append({
            "model": model,
            "score_at_1": f"{score_at_1:.2f}",
            "pass_at_1": f"{pass_at_1:.4f}",
            "num_problems": len(scores),
        })

    if not rows:
        print("No results found for numeric-ID problems.")
        return

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "score_at_1", "pass_at_1", "num_problems"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    print(f"Models: {len(rows)}, problems filtered to numeric IDs only")
    for r in rows:
        print(f"  {r['model']}: score@1={r['score_at_1']}, pass@1={r['pass_at_1']}, n={r['num_problems']}")


if __name__ == "__main__":
    main()
