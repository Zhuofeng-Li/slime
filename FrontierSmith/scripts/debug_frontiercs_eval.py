#!/usr/bin/env python3
"""
Debug Frontier-CS eval: check judge, extract_cpp, and a few sample scores.

Usage:
  python scripts/debug_frontiercs_eval.py --base-url http://localhost:8010/v1 --judge-url http://localhost:8082
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "verl"))

import pandas as pd
import requests

# Direct import to avoid pulling in ray
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "frontiercs",
    PROJECT_ROOT / "verl" / "verl" / "utils" / "reward_score" / "frontiercs.py",
)
_frontiercs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_frontiercs)
extract_cpp = _frontiercs.extract_cpp
compute_score = _frontiercs.compute_score


def submit_and_get_full_result(judge_url: str, code: str, problem_id: str) -> dict | None:
    """Submit code to judge and return full result (for debugging)."""
    import time
    url = judge_url.rstrip("/")
    try:
        r = requests.post(
            f"{url}/submit",
            files={"code": ("sol.cpp", code)},
            data={"pid": problem_id, "lang": "cpp"},
            timeout=30,
        )
        r.raise_for_status()
        sid = r.json().get("sid")
        if not sid:
            return None
        for _ in range(150):  # poll up to ~5 min
            r2 = requests.get(f"{url}/result/{sid}", timeout=10)
            if r2.status_code == 404:
                time.sleep(2)
                continue
            r2.raise_for_status()
            res = r2.json()
            if res.get("status") in ("done", "error"):
                return res
            time.sleep(2)
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "frontiercs" / "full.parquet")
    parser.add_argument("--base-url", default="http://localhost:8010/v1")
    parser.add_argument("--judge-url", default="http://localhost:8082")
    parser.add_argument("--n-check", type=int, default=3, help="Number of problems to test")
    args = parser.parse_args()

    # 1. Check judge
    print("=" * 60)
    print("1. Judge connectivity")
    print("=" * 60)
    try:
        r = requests.get(f"{args.judge_url.rstrip('/')}/health", timeout=5)
        print(f"  GET {args.judge_url}/health -> {r.status_code}")
    except Exception as e:
        print(f"  FAIL: {e}")
        print("  -> Judge not reachable. Start: cd Frontier-CS/algorithmic && ./run_judge.sh")
    try:
        r = requests.get(f"{args.judge_url.rstrip('/')}/problems", timeout=5)
        data = r.json() if r.ok else {}
        probs = data.get("problems", data) if isinstance(data, dict) else data
        probs = probs if isinstance(probs, list) else []
        print(f"  GET {args.judge_url}/problems -> {len(probs)} problems")
        if probs:
            pids = [p.get("id", p) if isinstance(p, dict) else p for p in probs[:5]]
            print(f"  Sample pids: {pids}")
    except Exception as e:
        print(f"  FAIL: {e}")

    # 2. Load data and get prompts
    print("\n" + "=" * 60)
    print("2. Data & generation")
    print("=" * 60)
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

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="dummy")

    def prompt_to_messages(prompt):
        return [{"role": m["role"], "content": m["content"]} for m in prompt]

    for i, prob in enumerate(problems[: args.n_check]):
        uid = prob["ground_truth"]
        messages = prompt_to_messages(prob["prompt"])
        print(f"\n  Problem {uid}:")
        try:
            resp = client.chat.completions.create(
                model="",
                messages=messages,
                temperature=0.7,
                max_tokens=16384,
            )
            text = resp.choices[0].message.content or ""
            code = extract_cpp(text)
            print(f"    Response length: {len(text)} chars")
            has_block = "```cpp" in text or "```c++" in text
            print(f"    Has ```cpp block: {'yes' if has_block else 'no'}")
            print(f"    Extracted code length: {len(code)} chars")
            if not code:
                print(f"    -> extract_cpp returned empty -> score=0")
                print(f"    First 250 chars of raw response:\n{repr(text[:250])}")
            else:
                if not has_block:
                    print(f"    -> Used fallback (no ``` block); first 150 chars of extracted:\n{repr(code[:150])}")
                    print(f"    First 250 chars of raw response:\n{repr(text[:250])}")
                score = compute_score("frontiercs", text, uid, judge_url=args.judge_url)
                print(f"    Judge score: {score}")
                if score == 0 and code:
                    print(f"    -> Code extracted but judge returned 0 (compile/runtime error or WA)")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("  - If extract_cpp is empty: model may not output ```cpp ... ```")
    print("  - If judge score=0 with code: check judge logs, code may be wrong")
    print("  - Judge score range: 0-100 (percentage)")


if __name__ == "__main__":
    main()
