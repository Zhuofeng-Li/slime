#!/usr/bin/env python3
"""
Randomly sample N problems from all local HardTest tiers (not the synthetic-pipeline 800).

Scans:
  data/problems/hardtest_medium_hard/<pid>/statement.txt
  data/problems/hardtest_hard/<pid>/statement.txt
  data/problems/hardtest_very_hard/<pid>/statement.txt

Writes a manifest compatible with install_hardtest_frontier_packages.py (--filtered-json).

Usage:
  python scripts/sample_hardtest_problems.py --n 800 --seed 42
  python scripts/sample_hardtest_problems.py --n 800 --seed 42 -o results/hardtest_sampled_800.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "problems"
DEFAULT_OUT = PROJECT_ROOT / "results" / "hardtest_sampled_800.json"

HARDTEST_TIERS = ("medium_hard", "hard", "very_hard")


def list_all_problems() -> list[dict]:
    out: list[dict] = []
    for tier in HARDTEST_TIERS:
        tier_dir = DATA_DIR / f"hardtest_{tier}"
        if not tier_dir.is_dir():
            continue
        for p in sorted(tier_dir.iterdir()):
            if not p.is_dir():
                continue
            if not (p / "statement.txt").is_file():
                continue
            problem_id = f"{tier}_{p.name}"
            out.append(
                {
                    "tier": tier,
                    "problem_id": problem_id,
                    "folder_name": p.name,
                    "source_path": str(p / "statement.txt"),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=800, help="Number of problems to sample")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible split)")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    pool = list_all_problems()
    if len(pool) < args.n:
        print(f"Pool has only {len(pool)} problems with statement.txt; requested {args.n}.")
        return

    rng = random.Random(args.seed)
    sampled = rng.sample(pool, args.n)

    out = {
        "source": "random_sample_local_hardtest",
        "seed": args.seed,
        "n_requested": args.n,
        "pool_size": len(pool),
        "valid_count": len(sampled),
        "valid_problems": [{"problem_id": s["problem_id"], "tier": s["tier"], "folder_name": s["folder_name"]} for s in sampled],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sampled {len(sampled)} / pool {len(pool)} -> {args.output}")


if __name__ == "__main__":
    main()
