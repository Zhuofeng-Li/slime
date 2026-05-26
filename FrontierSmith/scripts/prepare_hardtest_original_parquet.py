#!/usr/bin/env python3
"""
Build train_hardtest_original.parquet for VERL GRPO on HardTests (original statements + public tests).

Expects judge problem dirs created by:
  python scripts/install_hardtest_frontier_packages.py [--package-prefix hardtest_smp_]

Each row uses data_source='hardtest_orig' and reward_model.ground_truth='<package_prefix><problem_id>'.
VERL uses Python OJF + g++ when output_judging_function.py is present, else the Frontier-CS HTTP judge.

Usage:
  python scripts/install_hardtest_frontier_packages.py
  python scripts/prepare_hardtest_original_parquet.py
  python scripts/prepare_hardtest_original_parquet.py --package-prefix hardtest_smp_ --output data/frontiercs/train_hardtest_sampled.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBLEMS_ROOT = PROJECT_ROOT / "Frontier-CS" / "algorithmic" / "problems"
DEFAULT_OUT = PROJECT_ROOT / "data" / "frontiercs"


def build_prompt(statement: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"""You are a competitive programmer. Solve the following problem in C++. Output ONLY the C++ code wrapped in ```cpp and ```. No explanation.

{statement}

Generate solution code:""",
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--package-prefix",
        type=str,
        default="hardtest_orig_",
        help="Glob Frontier-CS/algorithmic/problems/{prefix}* (e.g. hardtest_smp_)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path (default: output-dir/train_hardtest_original.parquet)",
    )
    args = parser.parse_args()

    rows = []
    pattern = f"{args.package_prefix}*"
    for d in sorted(PROBLEMS_ROOT.glob(pattern)):
        if not d.is_dir():
            continue
        stmt_path = d / "statement.txt"
        cfg_path = d / "config.yaml"
        td = d / "testdata"
        if not stmt_path.is_file() or not cfg_path.is_file() or not td.is_dir():
            continue
        statement = stmt_path.read_text(encoding="utf-8")
        rows.append(
            {
                "prompt": build_prompt(statement),
                "reward_model": {"ground_truth": d.name},
                "data_source": "frontiercs",
            }
        )

    if not rows:
        print(f"No matching problem dirs ({pattern}) under {PROBLEMS_ROOT}/.")
        print("Run: python scripts/install_hardtest_frontier_packages.py [--package-prefix ...]")
        return

    df = pd.DataFrame(rows)
    out_path = args.output if args.output is not None else (args.output_dir / "train_hardtest_original.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(rows)} problems -> {out_path}")


if __name__ == "__main__":
    main()
