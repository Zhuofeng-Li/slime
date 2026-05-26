"""
Download HardTests problems dataset from HuggingFace and save each
problem's question_content as data/problems/hardtest/<pid>/statement.txt.

Usage:
    python scripts/download_hardtest.py [--limit N]

Options:
    --limit N   Only download the first N problems (for testing). Default: all.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "problems" / "hardtest"


def main():
    parser = argparse.ArgumentParser(description="Download HardTests problems")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max problems to download (0 = all)")
    args = parser.parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────
    print("Loading sigcp/hardtests_problems from HuggingFace ...")
    from datasets import load_dataset

    ds = load_dataset("sigcp/hardtests_problems", split="train")
    total = len(ds)
    n = args.limit if args.limit > 0 else total
    print(f"Dataset has {total} problems. Will process {n}.")

    # ── Write statement files ────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    for i in range(n):
        row = ds[i]
        pid = row["pid"]
        content = row.get("question_content") or ""

        if not content.strip():
            skipped += 1
            continue

        # Sanitize pid for filesystem (replace / with _)
        safe_pid = pid.replace("/", "_").replace("\\", "_")
        prob_dir = OUTPUT_DIR / safe_pid
        prob_dir.mkdir(parents=True, exist_ok=True)

        stmt_path = prob_dir / "statement.txt"
        stmt_path.write_text(content, encoding="utf-8")
        written += 1

        if written % 5000 == 0:
            print(f"  [{written}/{n}] written ...")

    print(f"\nDone. Written: {written}, Skipped (empty): {skipped}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
