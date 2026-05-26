#!/usr/bin/env python3
"""Parse Frontier-CS validation metrics from VERL trainer log output.

Usage:
  python scripts/parse_frontiercs_val_metrics.py <log_file>
  cat log.txt | python scripts/parse_frontiercs_val_metrics.py -

Reads stdin or file, extracts val-core/frontiercs/reward/* metrics, prints as CSV row.
"""

import re
import sys


def parse_metrics(text: str) -> dict[str, str]:
    """Extract mean@1, mean@5, best@5/mean from log text."""
    result = {}
    patterns = [
        ("score_at_1", r"['\"]?val-core/frontiercs/reward/mean@1['\"]?\s*:\s*([0-9.]+)"),
        ("avg_score_at_5", r"['\"]?val-core/frontiercs/reward/mean@5['\"]?\s*:\s*([0-9.]+)"),
        ("best_score_at_5", r"['\"]?val-core/frontiercs/reward/best@5/mean['\"]?\s*:\s*([0-9.]+)"),
    ]
    for key, pat in patterns:
        m = re.search(pat, text)
        result[key] = m.group(1) if m else ""
    return result


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: parse_frontiercs_val_metrics.py <log_file>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    if path == "-":
        text = sys.stdin.read()
    else:
        with open(path) as f:
            text = f.read()
    metrics = parse_metrics(text)
    print(f"{metrics.get('score_at_1', '')},{metrics.get('avg_score_at_5', '')},{metrics.get('best_score_at_5', '')}")


if __name__ == "__main__":
    main()
