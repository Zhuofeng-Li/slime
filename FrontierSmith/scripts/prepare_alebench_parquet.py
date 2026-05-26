#!/usr/bin/env python3
"""
Prepare ALE-Bench problems as validation Parquet for VERL RL training.

Downloads problem statements from HuggingFace and formats them identically
to the Frontier-CS parquet (columns: prompt, reward_model, data_source).

Usage:
  python scripts/prepare_alebench_parquet.py
  python scripts/prepare_alebench_parquet.py --lite          # 10 lite problems (default)
  python scripts/prepare_alebench_parquet.py --no-lite       # all 40+ problems
  python scripts/prepare_alebench_parquet.py --output-dir data/alebench
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "alebench"


def build_prompt(problem_id: str, statement: str, time_limit: float, mem_limit_mb: int) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                f"You are a competitive programmer. "
                f"Solve the following optimization problem in C++17. "
                f"Output ONLY the C++ code wrapped in ```cpp and ```. No explanation.\n\n"
                f"Time limit: {time_limit} seconds\n"
                f"Memory limit: {mem_limit_mb} MB\n\n"
                f"{statement}\n\n"
                f"Generate solution code:"
            ),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--data-source",
        type=str,
        default="alebench",
        help="Value to write into the parquet data_source column.",
    )
    parser.add_argument("--lite", dest="lite", action="store_true", default=True,
                        help="Use lite problem set (10 problems, default)")
    parser.add_argument("--no-lite", dest="lite", action="store_false",
                        help="Use full problem set (40+ problems)")
    parser.add_argument("--batch-only", action="store_true", default=False,
                        help="Exclude reactive (interactive) problems")
    args = parser.parse_args()

    # Import here so HF_HOME etc. are set before import
    from ale_bench.data import list_problem_ids, load_problem

    problem_ids = list_problem_ids(lite_version=args.lite)
    print(f"Loading {len(problem_ids)} ALE-Bench problems (lite={args.lite})...")

    rows = []
    for pid in problem_ids:
        try:
            problem, _seeds, _standings, _rpm, data_root = load_problem(pid, lite_version=args.lite)
        except Exception as e:
            print(f"  SKIP {pid}: {e}")
            continue

        if args.batch_only and problem.metadata.problem_type.value == "reactive":
            print(f"  SKIP {pid}: reactive problem")
            continue

        stmt = problem.statement  # English markdown
        time_limit = problem.constraints.time_limit
        mem_limit_mb = problem.constraints.memory_limit // 1024 // 1024

        rows.append(
            {
                "prompt": build_prompt(pid, stmt, time_limit, mem_limit_mb),
                "reward_model": {"ground_truth": pid},
                "data_source": args.data_source,
            }
        )
        ptype = problem.metadata.problem_type.value
        stype = problem.metadata.score_type.value
        print(f"  OK {pid}: {ptype}, {stype}, {time_limit}s, {mem_limit_mb}MB")

    if not rows:
        print("No problems found.")
        return

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "val.parquet"

    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df)
    idx = table.schema.get_field_index("data_source")
    table = table.set_column(idx, pa.field("data_source", pa.string()), table.column(idx).cast(pa.string()))
    pq.write_table(table, out_path)
    print(f"\nSaved {len(df)} problems -> {out_path}")


if __name__ == "__main__":
    main()
