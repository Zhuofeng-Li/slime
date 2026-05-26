#!/usr/bin/env python3
"""
Run Qwen3.5-27B evaluation on Frontier-CS algorithmic dataset.

Prerequisites:
  - vLLM serving Qwen3.5-27B (default: http://localhost:8000/v1)
  - Frontier-CS judge on port 8082 (cd Frontier-CS/algorithmic && ./run_judge.sh)

Usage:
  python scripts/run_qwen_frontiercs.py
  python scripts/run_qwen_frontiercs.py --vllm-url http://192.168.1.100:8000/v1
  python scripts/run_qwen_frontiercs.py --skip-generate   # Only run evaluation (use existing solutions)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FRONTIER_ALGO = PROJECT_ROOT / "Frontier-CS" / "algorithmic"
SCRIPTS_DIR = FRONTIER_ALGO / "scripts"
SOLUTIONS_DIR = FRONTIER_ALGO / "solutions"
RESULTS_DIR = FRONTIER_ALGO / "results" / "batch"

# Model prefix for solution filenames (from get_model_prefix)
def get_model_prefix(model: str) -> str:
    """Sanitize model name to file prefix (matches frontier_cs.models.get_model_prefix)."""
    import re
    if "/" in model:
        model = model.split("/", 1)[1]
    return re.sub(r"[^a-zA-Z0-9]+", "", model.lower()) or "qwen3527b"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5-27B on Frontier-CS")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible API base URL")
    parser.add_argument("--vllm-model", default="Qwen/Qwen3.5-27B",
                        help="Model name in vLLM (match --served-model-name or HF path, default: Qwen/Qwen3.5-27B)")
    parser.add_argument("--judge-url", default="http://localhost:8082",
                        help="Frontier-CS judge URL")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Parallel generation workers (default: 8)")
    parser.add_argument("--variants", type=int, default=1,
                        help="Number of solution variants per problem (default: 1)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip generation, only run evaluation")

    args = parser.parse_args()

    # Use vllm/ prefix so actual_model sent to API = full name (e.g. Qwen/Qwen3.5-27B)
    model_name = f"vllm/{args.vllm_model}"
    model_prefix = get_model_prefix(model_name)

    if not args.skip_generate:
        print("=" * 60)
        print("Step 1: Generate solutions with Qwen3.5-27B via vLLM")
        print("=" * 60)
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "generate_solutions.py"),
            "--model", model_name,
            "--base-url", args.vllm_url,
            "--judge-url", args.judge_url,
            "--concurrency", str(args.concurrency),
            "--indices", str(args.variants),
        ]
        print(f"Running: {' '.join(cmd)}")
        ret = subprocess.run(cmd, cwd=str(FRONTIER_ALGO))
        if ret.returncode != 0:
            print("Generation failed. Exiting.")
            sys.exit(ret.returncode)

    print("\n" + "=" * 60)
    print("Step 2: Evaluate solutions on judge")
    print("=" * 60)

    # frontier batch algorithmic --model qwen3527b --judge-url http://localhost:8082 --workers 8
    frontier_cmd = [
        "frontier", "batch", "algorithmic",
        "--model", model_prefix,
        "--judge-url", args.judge_url,
        "--workers", str(args.concurrency),
    ]
    print(f"Running: {' '.join(frontier_cmd)}")
    # frontier batch uses Frontier-CS package; run from project root
    ret = subprocess.run(frontier_cmd, cwd=str(PROJECT_ROOT))
    if ret.returncode != 0:
        print("Evaluation failed. Exiting.")
        sys.exit(ret.returncode)

    print("\n" + "=" * 60)
    print("Done. Results in:", PROJECT_ROOT / "Frontier-CS" / "algorithmic" / "results" / "batch")
    print("=" * 60)


if __name__ == "__main__":
    main()
