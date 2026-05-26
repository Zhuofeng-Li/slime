#!/usr/bin/env python3
"""
Inspect response content from GRPO training when response_length spikes occur.

Use when trainer.dump_long_responses_dir is set during training. When a spike is detected,
samples are saved as spike_step{N}.jsonl. This script helps analyze them.

Usage:
  python scripts/inspect_response_length_spikes.py outputs/dump_long_responses
  python scripts/inspect_response_length_spikes.py outputs/dump_long_responses --limit 3
  python scripts/inspect_response_length_spikes.py outputs/dump_long_responses --check-repetition
"""

import argparse
import json
from pathlib import Path


def detect_repetition(text: str) -> dict:
    """Detect if text has repetition (model repeating same content).
    Returns dict with: has_repetition, unique_ngram_ratio, dup_line_ratio, max_repeat_span.
    """
    if not text or len(text) < 100:
        return {"has_repetition": False, "unique_ngram_ratio": 1.0, "dup_line_ratio": 0, "max_repeat_span": 0}

    words = text.split()
    lines = [L.strip() for L in text.split("\n") if L.strip()]

    # 1. Unique n-gram ratio (low = repetitive)
    n = 12
    step = max(1, n // 2)
    ngrams = [tuple(words[i : i + n]) for i in range(0, len(words) - n + 1, step)]
    unique_ratio = len(set(ngrams)) / len(ngrams) if len(ngrams) >= 10 else 1.0

    # 2. Consecutive duplicate lines ratio
    dup_count = 0
    prev = None
    for L in lines:
        if L == prev:
            dup_count += 1
        prev = L
    dup_line_ratio = dup_count / len(lines) if lines else 0

    # 3. Longest back-to-back repeated phrase (words)
    max_span = 0
    for span in [30, 80, 150, 300]:
        if span > len(words) // 2:
            break
        for i in range(0, min(5000, len(words) - 2 * span), max(1, span // 2)):
            if tuple(words[i : i + span]) == tuple(words[i + span : i + 2 * span]):
                max_span = max(max_span, span)
                break

    has_rep = unique_ratio < 0.4 or dup_line_ratio > 0.15 or max_span >= 30
    return {
        "has_repetition": has_rep,
        "unique_ngram_ratio": round(unique_ratio, 3),
        "repetition_score": round(1 - unique_ratio, 3),  # high = more repetitive
        "dup_line_ratio": round(dup_line_ratio, 3),
        "max_repeat_span": max_span,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dumped long responses from GRPO training")
    parser.add_argument("dump_dir", type=Path, help="Directory with spike_step*.jsonl files")
    parser.add_argument("--limit", type=int, default=0, help="Max files to show (0=all)")
    parser.add_argument("--show-full-output", action="store_true", help="Print full output instead of preview")
    parser.add_argument(
        "--check-repetition",
        action="store_true",
        help="Analyze each sample for repetition (model repeating same content)",
    )
    args = parser.parse_args()

    dump_dir = args.dump_dir
    if not dump_dir.exists():
        print(f"Directory not found: {dump_dir}")
        print("\nTo capture spike data, run training with:")
        print("  trainer.dump_long_responses_dir=outputs/dump_long_responses \\")
        print("  trainer.dump_long_responses_threshold=20000 \\")
        print("  ... other args ...")
        return

    files = sorted(dump_dir.glob("spike_step*.jsonl"))
    if not files:
        print(f"No spike_step*.jsonl files in {dump_dir}")
        return

    if args.limit > 0:
        files = files[: args.limit]

    for fpath in files:
        print(f"\n{'='*60}\nFile: {fpath.name}\n{'='*60}")
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                print(f"\n--- Sample (step={entry.get('step')}, idx={entry.get('index')}) ---")
                print(f"Response length: {entry.get('response_length')} tokens")
                print(f"Score: {entry.get('score')}")
                print(f"Problem ID: {entry.get('ground_truth')}")

                # Repetition check
                output_text = entry.get("output_full") or entry.get("output_preview", "")
                if args.check_repetition and output_text:
                    rep = detect_repetition(output_text)
                    rep_flag = "⚠️ YES" if rep["has_repetition"] else "✓ No"
                    print(f"\nRepetition: {rep_flag}")
                    print(
                        f"  unique_ngram_ratio={rep['unique_ngram_ratio']} "
                        f"(low=repetitive), dup_line_ratio={rep['dup_line_ratio']}, "
                        f"max_repeat_span={rep['max_repeat_span']} words"
                    )
                    if rep["has_repetition"]:
                        print("  → Model likely stuck in repeat loop for this spike.")
                elif entry.get("repetition"):
                    rep = entry["repetition"]
                    rep_flag = "⚠️ YES" if rep.get("has_repetition") else "✓ No"
                    print(f"\nRepetition: {rep_flag} (from dump)")
                    print(f"  {rep}")

                print(f"\nInput (preview):\n{entry.get('input_preview', '')[:800]}...")
                if args.show_full_output:
                    print(f"\nFull output:\n{entry.get('output_full', '')}")
                else:
                    print(f"\nOutput (preview):\n{entry.get('output_preview', '')}")
                print()


if __name__ == "__main__":
    main()
