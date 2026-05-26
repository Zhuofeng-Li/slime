#!/usr/bin/env python3
"""
Prepare two 400-problem parquets for the two-phase curriculum training script.

1. train_hardtest_400.parquet  — random 400 from installed hardtest_smp_* packages
2. train_synthetic_400.parquet — random 400 from the full synthetic set (train_synthetic.parquet)

Both outputs land in data/frontiercs/.

Usage:
  python scripts/prepare_mixed_400_parquets.py [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "frontiercs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hardtest-parquet",
        type=Path,
        default=DATA_DIR / "train_hardtest_sampled.parquet",
        help="Source parquet for hardtest problems",
    )
    parser.add_argument(
        "--synthetic-parquet",
        type=Path,
        default=DATA_DIR / "train_synthetic.parquet",
        help="Source parquet for synthetic problems",
    )
    parser.add_argument("--n", type=int, default=400, help="Number of problems per split")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    for name, src in [("hardtest", args.hardtest_parquet), ("synthetic", args.synthetic_parquet)]:
        if not src.is_file():
            print(f"Missing {src}")
            print(f"  hardtest: run sample + install + prepare_hardtest_original_parquet")
            print(f"  synthetic: run filter + prepare_synthetic_parquet")
            return

        df = pd.read_parquet(src)
        if len(df) < args.n:
            print(f"WARNING: {name} source has only {len(df)} rows, using all of them (requested {args.n})")
            sampled = df
        else:
            sampled = df.sample(n=args.n, random_state=args.seed)

        out = args.output_dir / f"train_{name}_400.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        sampled.to_parquet(out, index=False)
        print(f"[{name}] Saved {len(sampled)} problems -> {out}")


if __name__ == "__main__":
    main()
