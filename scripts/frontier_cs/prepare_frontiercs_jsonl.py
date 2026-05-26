#!/usr/bin/env python3
"""
Convert FrontierSmith problems to SLIME JSONL format.

Reads problem statements from FrontierSmith/Frontier-CS/algorithmic/problems/
and writes one JSONL per line:
  {"prompt": [{"role": "user", "content": "..."}], "label": "<problem_id>"}

Label is the directory name (e.g. "frontiersmith_1", or numeric "123"),
which is passed to the judge as problem_id.

Usage:
  uv run python scripts/prepare_frontiercs_jsonl.py
  uv run python scripts/prepare_frontiercs_jsonl.py --output-dir /root/data/frontiercs --val-ratio 0.1
  uv run python scripts/prepare_frontiercs_jsonl.py --full-for-both   # all problems as both train & val
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PROBLEMS_DIR = REPO_ROOT / "FrontierSmith" / "Frontier-CS" / "algorithmic" / "problems"
DEFAULT_OUT = REPO_ROOT / "data" / "frontiercs"

SYSTEM_PROMPT = (
    "You are a competitive programmer. Solve the following problem in C++. "
    "Output ONLY the C++ code wrapped in ```cpp and ```. No explanation."
)


def build_prompt(statement: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"{SYSTEM_PROMPT}\n\n{statement}\n\nGenerate solution code:",
        }
    ]


def load_problems() -> list[dict]:
    rows = []
    for pid_dir in sorted(PROBLEMS_DIR.iterdir()):
        if not pid_dir.is_dir():
            continue
        stmt_file = pid_dir / "statement.txt"
        if not stmt_file.exists():
            continue
        statement = stmt_file.read_text(encoding="utf-8").strip()
        rows.append(
            {
                "prompt": build_prompt(statement),
                "label": pid_dir.name,
            }
        )
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} samples -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-ratio", type=float, default=0.0,
                        help="Fraction reserved for validation (0 = no separate val file)")
    parser.add_argument("--full-for-both", action="store_true",
                        help="Write all problems as both train.jsonl and val.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_problems()
    if not rows:
        print(f"No problems found under {PROBLEMS_DIR}")
        return

    print(f"Loaded {len(rows)} problems from {PROBLEMS_DIR}")

    if args.full_for_both:
        write_jsonl(rows, args.output_dir / "train.jsonl")
        write_jsonl(rows, args.output_dir / "val.jsonl")
        return

    if args.val_ratio > 0:
        rng = random.Random(args.seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_ratio))
        write_jsonl(shuffled[n_val:], args.output_dir / "train.jsonl")
        write_jsonl(shuffled[:n_val], args.output_dir / "val.jsonl")
    else:
        write_jsonl(rows, args.output_dir / "train.jsonl")


if __name__ == "__main__":
    main()
