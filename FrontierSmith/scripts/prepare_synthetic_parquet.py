#!/usr/bin/env python3
"""Build a Parquet of the 10 supplementary synthetic problems for VERL training.

Each problem directory under ``Frontier-CS/algorithmic/problems/`` contributes a
single row.  ``data_source`` is set to ``frontiercs`` and ``ground_truth`` is the
problem directory name, so the existing Frontier-CS judge handles them
unchanged.

Usage:
    python scripts/prepare_synthetic_parquet.py
    python scripts/prepare_synthetic_parquet.py --output-dir data/frontiercs --output-name train_synthetic.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBLEMS_DIR = PROJECT_ROOT / "Frontier-CS" / "algorithmic" / "problems"

SYNTHETIC_PROBLEM_IDS = [
    "synthetic_v5_very_hard_codeforces_633_g",
    "synthetic_v5_very_hard_codeforces_209_c",
    "synthetic_v5_hard_luogu_p2495",
    "synthetic_v5_very_hard_codeforces_1046_g",
    "synthetic_v5_hard_luogu_p4264",
    "synthetic_v5_very_hard_atcoder_arc177_f",
    "synthetic_v5_very_hard_luogu_p6610",
    "synthetic_v5_very_hard_codeforces_587_f",
    "synthetic_v5_hard_luogu_p2533",
    "synthetic_v5_very_hard_codeforces_1182_f",
]


def build_prompt(statement: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "You are a competitive programmer. Solve the following problem in C++. "
                "Output ONLY the C++ code wrapped in ```cpp and ```. No explanation.\n\n"
                f"{statement}\n\nGenerate solution code:"
            ),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "frontiercs")
    parser.add_argument("--output-name", type=str, default="train_synthetic.parquet")
    args = parser.parse_args()

    rows: list[dict] = []
    missing: list[str] = []
    for pid in SYNTHETIC_PROBLEM_IDS:
        stmt = PROBLEMS_DIR / pid / "statement.txt"
        if not stmt.is_file():
            missing.append(pid)
            continue
        rows.append(
            {
                "prompt": build_prompt(stmt.read_text(encoding="utf-8")),
                "reward_model": {"ground_truth": pid},
                "data_source": "frontiercs",
            }
        )

    if missing:
        print(f"Skipped {len(missing)} problems without statement.txt:")
        for pid in missing:
            print(f"  - {pid}")
    if not rows:
        raise SystemExit("No synthetic problems found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"Saved {len(rows)} problems -> {out_path}")


if __name__ == "__main__":
    main()
