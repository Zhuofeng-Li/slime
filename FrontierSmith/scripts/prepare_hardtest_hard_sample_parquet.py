#!/usr/bin/env python3
"""
Random-sample N problems from local data/problems/hardtest_hard and write:
  1. A manifest JSON compatible with install_hardtest_frontier_packages.py.
  2. A training parquet for VERL with data_source='hardtest_orig'.

The parquet uses data_source="frontiercs" so it goes through the Frontier-CS HTTP
judge on port 8082. reward_model.ground_truth references package dirs that
install_hardtest_frontier_packages.py creates with --package-prefix hardtest_smp_,
so eval requires running the installer first (see comments at bottom of file).

Usage:
  python scripts/prepare_hardtest_hard_sample_parquet.py
  python scripts/prepare_hardtest_hard_sample_parquet.py --n 200 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "problems"
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "frontiercs" / "train_hardtest_hard_200.parquet"
DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "hardtest_hard_sampled_200.json"

PACKAGE_PREFIX = "hardtest_smp_"
TIER = "hard"


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
    parser.add_argument("--n", type=int, default=200, help="Number of problems to sample")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_PARQUET, help="Output parquet path")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Output manifest JSON")
    args = parser.parse_args()

    tier_dir = DATA_DIR / f"hardtest_{TIER}"
    if not tier_dir.is_dir():
        raise SystemExit(f"Missing {tier_dir}")

    pool: list[dict] = []
    for p in sorted(tier_dir.iterdir()):
        if not p.is_dir():
            continue
        stmt = p / "statement.txt"
        if not stmt.is_file():
            continue
        text = stmt.read_text(encoding="utf-8")
        if not text.strip():
            continue
        pool.append({"folder_name": p.name, "statement_path": str(stmt), "statement": text})

    if len(pool) < args.n:
        raise SystemExit(f"Pool has only {len(pool)} valid problems; requested {args.n}")

    rng = random.Random(args.seed)
    sampled = rng.sample(pool, args.n)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "random_sample_local_hardtest_hard",
        "seed": args.seed,
        "n_requested": args.n,
        "pool_size": len(pool),
        "valid_count": len(sampled),
        "valid_problems": [
            {"problem_id": f"{TIER}_{s['folder_name']}", "tier": TIER, "folder_name": s["folder_name"]}
            for s in sampled
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote manifest: {args.manifest}")

    rows = [
        {
            "prompt": build_prompt(s["statement"]),
            "reward_model": {"ground_truth": f"{PACKAGE_PREFIX}{TIER}_{s['folder_name']}"},
            "data_source": "frontiercs",
        }
        for s in sampled
    ]
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote parquet : {args.output}  ({len(df)} rows)")
    print()
    print("Next steps to make these problems judgeable:")
    print(f"  python scripts/install_hardtest_frontier_packages.py \\")
    print(f"      --filtered-json {args.manifest} \\")
    print(f"      --package-prefix {PACKAGE_PREFIX}")


if __name__ == "__main__":
    main()
