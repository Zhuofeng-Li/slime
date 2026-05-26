#!/usr/bin/env python3
"""
Evaluate Frontier-CS 172 problems via vLLM OpenAI-compatible API.

Calls vLLM server for generation, Frontier-CS judge for scoring.
Computes score@1, avg_score@5, best_score@5.

Usage:
  # With vLLM server already running on localhost:8000
  python scripts/eval_frontiercs_via_vllm.py
  python scripts/eval_frontiercs_via_vllm.py --base-url http://localhost:8000/v1
  python scripts/eval_frontiercs_via_vllm.py --data data/frontiercs/full.parquet --n-samples 5
  python scripts/eval_frontiercs_via_vllm.py --concurrency 64  # 64 concurrent vLLM requests (default)
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
# Add verl package to path (verl/verl/ is the package when run from project root)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "verl"))

from verl.utils.reward_score.frontiercs import compute_score


def prompt_to_messages(prompt: list[dict]) -> list[dict]:
    """Convert parquet prompt format to OpenAI messages."""
    return [{"role": m["role"], "content": m["content"]} for m in prompt]


def generate_one(client, messages: list[dict], temperature: float = 0.7, max_tokens: int = 16384) -> str:
    """Generate one completion via OpenAI client."""
    resp = client.chat.completions.create(
        model="",  # vLLM ignores model when serving single model
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _eval_one(
    client,
    prob_idx: int,
    sample_idx: int,
    messages: list[dict],
    uid: str,
    temperature: float,
    max_tokens: int,
    judge_url: str,
) -> tuple[int, int, float]:
    """Single task: generate + score. Returns (prob_idx, sample_idx, score)."""
    try:
        text = generate_one(client, messages, temperature=temperature, max_tokens=max_tokens)
        score = compute_score("frontiercs", text, uid, judge_url=judge_url)
        return (prob_idx, sample_idx, score)
    except Exception as e:
        print(f"\n  Problem {uid} sample {sample_idx + 1} error: {e}")
        return (prob_idx, sample_idx, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "frontiercs" / "full.parquet")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--judge-url", default="http://localhost:8082")
    parser.add_argument("--n-samples", type=int, default=1, help="Samples per problem (1 for score@1, 5 for avg/best@5)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent vLLM requests (64 may cause !!! output on some setups)")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--print-csv-row", action="store_true", help="Print step,score_at_1,avg_score_at_5,best_score_at_5 for scripting")
    parser.add_argument("--print-progress", action="store_true", help="Print EVAL_PROGRESS after each problem completes (for checkpoint script)")
    parser.add_argument("--progress-prefix", type=str, default="", help="Prefix for EVAL_PROGRESS lines (e.g. 'Step 10:')")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Data not found: {args.data}")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="dummy")

    df = pd.read_parquet(args.data)
    problems = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, str):
            import ast
            prompt = ast.literal_eval(prompt) if prompt.startswith("[") else [{"role": "user", "content": prompt}]
        rm = row.get("reward_model", {})
        gt = rm.get("ground_truth", "") if isinstance(rm, dict) else row.get("ground_truth", "")
        problems.append({"prompt": prompt, "ground_truth": str(gt)})

    print(f"Loaded {len(problems)} problems from {args.data}")
    if not problems:
        print("ERROR: No problems loaded. Check parquet format.", file=sys.stderr)
        if args.print_csv_row:
            print("EVAL_CSV_ROW:NA,NA,NA", flush=True)
        sys.exit(1)
    print(f"Generating {args.n_samples} sample(s) per problem via {args.base_url} (concurrency={args.concurrency})")

    # Build tasks: (prob_idx, sample_idx, messages, uid)
    tasks = [
        (
            i,
            s,
            prompt_to_messages(prob["prompt"]),
            prob["ground_truth"],
        )
        for i, prob in enumerate(problems)
        for s in range(args.n_samples)
    ]

    # Run with ThreadPoolExecutor, progress bar
    all_scores = {prob["ground_truth"]: [None] * args.n_samples for prob in problems}
    n_problems = len(problems)

    def _aggregate(scores_dict: dict) -> tuple[float | None, float | None, float | None, int]:
        """Compute score@1, avg@5, best@5, n_complete. avg/best only for problems with all 5 samples."""
        s1_vals = [s[0] for s in scores_dict.values() if s[0] is not None]
        complete = [s for s in scores_dict.values() if all(x is not None for x in s)]
        score_at_1 = sum(s1_vals) / len(s1_vals) if s1_vals else None
        avg_at_5 = sum(sum(s) / len(s) for s in complete) / len(complete) if complete else None
        best_at_5 = sum(max(s) for s in complete) / len(complete) if complete else None
        return score_at_1, avg_at_5, best_at_5, len(complete)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {
            ex.submit(
                _eval_one,
                client,
                i,
                s,
                messages,
                uid,
                args.temperature,
                args.max_tokens,
                args.judge_url,
            ): (i, s)
            for i, s, messages, uid in tasks
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Eval", unit="req")
        for f in pbar:
            prob_idx, sample_idx, score = f.result()
            uid = problems[prob_idx]["ground_truth"]
            all_scores[uid][sample_idx] = score

            # When this problem completes (all n_samples done), output its scores + running totals
            if args.print_progress and args.n_samples >= 5:
                prob_scores = all_scores[uid]
                if all(x is not None for x in prob_scores):
                    s1, a5, b5, n_complete = _aggregate(all_scores)
                    ps1 = prob_scores[0]
                    pavg = sum(prob_scores) / len(prob_scores)
                    pbest = max(prob_scores)
                    s1_str = f"{s1:.4f}" if s1 is not None else "-"
                    a5_str = f"{a5:.4f}" if a5 is not None else "-"
                    b5_str = f"{b5:.4f}" if b5 is not None else "-"
                    prefix = args.progress_prefix
                    print(
                        f"{prefix}EVAL_PROGRESS: problem={uid} score@1={ps1:.4f} avg@5={pavg:.4f} best@5={pbest:.4f} | "
                        f"running: {n_complete}/{n_problems} score@1={s1_str} avg@5={a5_str} best@5={b5_str}",
                        flush=True,
                    )

    # Aggregate: score@1 = first sample; avg@5 = mean of 5; best@5 = max of 5 per problem
    score_at_1 = sum(s[0] for s in all_scores.values()) / len(all_scores)
    avg_score_at_5 = best_score_at_5 = None
    if args.n_samples >= 5:
        avg_score_at_5 = sum(sum(s) / len(s) for s in all_scores.values()) / len(all_scores)
        best_score_at_5 = sum(max(s) for s in all_scores.values()) / len(all_scores)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  score@1:        {score_at_1:.2f}")
    if avg_score_at_5 is not None:
        print(f"  avg_score@5:   {avg_score_at_5:.2f}")
    if best_score_at_5 is not None:
        print(f"  best_score@5:  {best_score_at_5:.2f}")
    print("=" * 50)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w") as f:
            f.write("metric,value\n")
            f.write(f"score_at_1,{score_at_1:.4f}\n")
            if avg_score_at_5 is not None:
                f.write(f"avg_score_at_5,{avg_score_at_5:.4f}\n")
            if best_score_at_5 is not None:
                f.write(f"best_score_at_5,{best_score_at_5:.4f}\n")
        print(f"Saved to {args.output_csv}")

    if args.print_csv_row:
        a5 = f"{avg_score_at_5:.4f}" if avg_score_at_5 is not None else "NA"
        b5 = f"{best_score_at_5:.4f}" if best_score_at_5 is not None else "NA"
        print(f"EVAL_CSV_ROW:{score_at_1:.4f},{a5},{b5}")


if __name__ == "__main__":
    main()
