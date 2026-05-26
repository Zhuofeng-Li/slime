"""
Split hardtests_problems by difficulty_ratings into subdatasets.

Each problem's first difficulty_ratings entry is used. Levels are normalized:
  - "very hard" / "very_hard" -> very_hard
  - "unknown" / "unknown_difficulty" -> unknown

Output: data/problems/hardtest_<level>/<pid>/statement.txt

Usage:
    python scripts/split_hardtest_by_difficulty.py [--limit N]

Options:
    --limit N   Only process first N problems (for testing). Default: all.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "problems"

# Normalize level names for directory names
LEVEL_ALIASES = {
    "very hard": "very_hard",
    "very_hard": "very_hard",
    "unknown": "unknown",
    "unknown_difficulty": "unknown",
}


def normalize_level(raw: str) -> str:
    """Return normalized level for use as subdir name."""
    if not raw:
        return "unknown"
    return LEVEL_ALIASES.get(raw.strip().lower(), raw.strip().lower().replace(" ", "_"))


def main():
    parser = argparse.ArgumentParser(description="Split HardTests by difficulty level")
    parser.add_argument("--limit", type=int, default=0, help="Max problems to process (0 = all)")
    args = parser.parse_args()

    print("Loading sigcp/hardtests_problems from HuggingFace ...")
    from datasets import load_dataset

    ds = load_dataset("sigcp/hardtests_problems", split="train")
    total = len(ds)
    n = args.limit if args.limit > 0 else total
    print(f"Dataset has {total} problems. Will process {n}.")

    # level -> count, level -> list of pids
    level_counts = {}
    level_pids = {}

    for i in range(n):
        row = ds[i]
        pid = row.get("pid", "")
        content = row.get("question_content") or ""

        if not content.strip():
            continue

        ratings = row.get("difficulty_ratings") or []
        if ratings:
            first = ratings[0] if isinstance(ratings[0], dict) else {}
            raw_level = first.get("level", "unknown") or "unknown"
        else:
            raw_level = "unknown"

        level = normalize_level(raw_level)
        level_counts[level] = level_counts.get(level, 0) + 1
        if level not in level_pids:
            level_pids[level] = []
        level_pids[level].append((pid, content))

    # Write files
    print("\n--- Count by difficulty level ---")
    for level in sorted(level_counts.keys()):
        print(f"  {level}: {level_counts[level]}")

    print("\nWriting statement files ...")
    for level, items in level_pids.items():
        out_dir = BASE_DIR / f"hardtest_{level}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for pid, content in items:
            safe_pid = pid.replace("/", "_").replace("\\", "_")
            prob_dir = out_dir / safe_pid
            prob_dir.mkdir(parents=True, exist_ok=True)
            (prob_dir / "statement.txt").write_text(content, encoding="utf-8")

    print(f"\nDone. Subdatasets written to {BASE_DIR}/hardtest_<level>/")
    print("\nSummary:")
    for level in sorted(level_counts.keys()):
        print(f"  hardtest_{level}: {level_counts[level]} problems")


if __name__ == "__main__":
    main()
