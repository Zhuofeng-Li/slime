#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configurable settings
# =========================

SERVER_URLS=(
  "http://SERVER_1:8000/v1"  # baseline
  "http://SERVER_2:8000/v1"  # train on Frontier-CS
  "http://SERVER_3:8000/v1"  # train on synthetic
  "http://SERVER_4:8000/v1"  # train on HardTest
)

SERVER_LABELS=(
  "baseline"
  "train_on_frontiercs"
  "train_on_synthetic"
  "train_on_hardtest"
)

MODEL_NAME="Qwen3.5-27B"
EXPECTED_SERVER_COUNT=4

SAMPLES_PER_PROBLEM=5
NUM_PROBLEMS=10
CONCURRENCY=8

MAX_TOKENS=40192
TEMPERATURE=0.7
TOP_P=0.95
PRESENCE_PENALTY=0.0
FREQUENCY_PENALTY=0.0
STOP_JSON="[]"
EXTRA_REQUEST_JSON="{}"

REQUEST_TIMEOUT_SECONDS=7200
MAX_RETRIES=3
RETRY_BASE_SLEEP_SECONDS=5
RETRY_MAX_SLEEP_SECONDS=60

CODE_LANGUAGE_NAME="C++17"
CODE_FENCE_LANGUAGE="cpp"
PROMPT_LANGUAGE="en"
INCLUDE_EXAMPLES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# This script intentionally reads pre-materialized ALE-Bench-Lite JSON and
# never starts ALE-Bench sessions, judges, or Docker.
ALE_BENCH_JSON_PATH="${ALE_BENCH_JSON_PATH:-${REPO_ROOT}/data/alebench/lite_problems.json}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/alebench_lite_qwen35_27b_oneshot}"
SUMMARY_JSONL_NAME="summary.jsonl"
ERROR_LOG_NAME="errors.jsonl"

API_KEY="${API_KEY:-dummy}"

# Leave empty to use the first NUM_PROBLEMS entries from problem_ids_lite.txt.
# Example:
# PROBLEM_IDS=(ahc027 ahc039)
PROBLEM_IDS=()

# =========================
# CLI
# =========================

OVERWRITE=0

usage() {
  cat <<EOF
Usage: $0 [--overwrite] [--output-root PATH] [--json-path PATH] [--help]

Generates one-shot ALE-Bench-Lite responses from ${MODEL_NAME} served by four
OpenAI-compatible vLLM servers.

Outputs:
  \$OUTPUT_ROOT/samples/<server_label>/<problem_id>/sample_YY.json
  \$OUTPUT_ROOT/${SUMMARY_JSONL_NAME}
  \$OUTPUT_ROOT/${ERROR_LOG_NAME}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --output-root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --output-root" >&2
        exit 2
      fi
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --json-path)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --json-path" >&2
        exit 2
      fi
      ALE_BENCH_JSON_PATH="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#SERVER_URLS[@]} -ne "${EXPECTED_SERVER_COUNT}" ]]; then
  echo "Expected ${EXPECTED_SERVER_COUNT} SERVER_URLS, got ${#SERVER_URLS[@]}" >&2
  exit 2
fi

if [[ ${#SERVER_LABELS[@]} -ne "${EXPECTED_SERVER_COUNT}" ]]; then
  echo "Expected ${EXPECTED_SERVER_COUNT} SERVER_LABELS, got ${#SERVER_LABELS[@]}" >&2
  exit 2
fi

PROBLEM_IDS_CSV="$(IFS=,; echo "${PROBLEM_IDS[*]}")"
SERVER_LABELS_CSV="$(IFS=,; echo "${SERVER_LABELS[*]}")"

export MODEL_NAME
export SAMPLES_PER_PROBLEM
export NUM_PROBLEMS
export CONCURRENCY
export MAX_TOKENS
export TEMPERATURE
export TOP_P
export PRESENCE_PENALTY
export FREQUENCY_PENALTY
export STOP_JSON
export EXTRA_REQUEST_JSON
export REQUEST_TIMEOUT_SECONDS
export MAX_RETRIES
export RETRY_BASE_SLEEP_SECONDS
export RETRY_MAX_SLEEP_SECONDS
export CODE_LANGUAGE_NAME
export CODE_FENCE_LANGUAGE
export PROMPT_LANGUAGE
export INCLUDE_EXAMPLES
export ALE_BENCH_JSON_PATH
export OUTPUT_ROOT
export SUMMARY_JSONL_NAME
export ERROR_LOG_NAME
export API_KEY
export PROBLEM_IDS_CSV
export SERVER_LABELS_CSV

python3 - "$OVERWRITE" "${SERVER_URLS[@]}" <<'PY'
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import random
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

overwrite = sys.argv[1] == "1"
server_urls = [s.rstrip("/") for s in sys.argv[2:]]
server_labels = [s.strip() for s in os.environ["SERVER_LABELS_CSV"].split(",") if s.strip()]

model_name = os.environ["MODEL_NAME"]
samples_per_problem = int(os.environ["SAMPLES_PER_PROBLEM"])
num_problems = int(os.environ["NUM_PROBLEMS"])
concurrency = int(os.environ["CONCURRENCY"])

max_tokens = int(os.environ["MAX_TOKENS"])
temperature = float(os.environ["TEMPERATURE"])
top_p = float(os.environ["TOP_P"])
presence_penalty = float(os.environ["PRESENCE_PENALTY"])
frequency_penalty = float(os.environ["FREQUENCY_PENALTY"])
stop_json = json.loads(os.environ["STOP_JSON"])
extra_request_json = json.loads(os.environ["EXTRA_REQUEST_JSON"])

request_timeout = float(os.environ["REQUEST_TIMEOUT_SECONDS"])
max_retries = int(os.environ["MAX_RETRIES"])
retry_base_sleep = float(os.environ["RETRY_BASE_SLEEP_SECONDS"])
retry_max_sleep = float(os.environ["RETRY_MAX_SLEEP_SECONDS"])

code_language_name = os.environ["CODE_LANGUAGE_NAME"]
code_fence_language = os.environ["CODE_FENCE_LANGUAGE"]
prompt_language = os.environ["PROMPT_LANGUAGE"]
include_examples = os.environ["INCLUDE_EXAMPLES"] not in {"0", "false", "False", "no", "NO"}

data_path = Path(os.environ["ALE_BENCH_JSON_PATH"]).expanduser().resolve()
output_root = Path(os.environ["OUTPUT_ROOT"]).expanduser().resolve()
summary_jsonl = output_root / os.environ["SUMMARY_JSONL_NAME"]
error_log = output_root / os.environ["ERROR_LOG_NAME"]
api_key = os.environ.get("API_KEY", "")
problem_ids_csv = os.environ.get("PROBLEM_IDS_CSV", "").strip()

samples_root = output_root / "samples"
prompts_root = output_root / "prompts"
metadata_root = output_root / "metadata"

log_lock = threading.Lock()
print_lock = threading.Lock()


def slugify_label(label: str) -> str:
    chars = []
    for ch in label.strip().lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {"-", "_", ".", " "}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "server"


def utc_now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def append_error(payload: dict[str, Any]) -> None:
    payload = {"logged_at": utc_now(), **payload}
    with log_lock:
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with error_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_problems_from_json() -> list[dict[str, Any]]:
    if not data_path.is_file():
        raise FileNotFoundError(
            f"ALE_BENCH_JSON_PATH does not exist: {data_path}. Build it from ALE-Bench-Lite zips first."
        )

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    raw_problems = payload.get("problems")
    if not isinstance(raw_problems, list):
        raise TypeError(f"Invalid ALE-Bench JSON format: expected top-level 'problems' list in {data_path}")

    by_id: dict[str, dict[str, Any]] = {}
    for raw_problem in raw_problems:
        if not isinstance(raw_problem, dict):
            raise TypeError(f"Invalid problem entry in {data_path}: {type(raw_problem).__name__}")
        problem_id = raw_problem.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError(f"Invalid problem_id in {data_path}: {problem_id!r}")
        by_id[problem_id] = raw_problem

    if problem_ids_csv:
        ids = [x.strip() for x in problem_ids_csv.split(",") if x.strip()]
    else:
        raw_ids = payload.get("problem_ids")
        if isinstance(raw_ids, list) and raw_ids:
            ids = [str(x) for x in raw_ids][:num_problems]
        else:
            ids = list(by_id.keys())[:num_problems]

    if len(ids) != num_problems:
        raise ValueError(f"Expected {num_problems} ALE-Bench-Lite problem ids, got {len(ids)}: {ids}")
    missing_ids = [problem_id for problem_id in ids if problem_id not in by_id]
    if missing_ids:
        raise ValueError(f"Problem ids not found in {data_path}: {missing_ids}")

    problems = []
    for problem_id in ids:
        raw = by_id[problem_id]
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        constraints = raw.get("constraints") if isinstance(raw.get("constraints"), dict) else {}
        statement = (
            raw.get(f"statement_{prompt_language}")
            or raw.get("statement_en")
            or raw.get("statement")
            or ""
        )
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"Problem {problem_id} has no statement text in {data_path}")

        memory_limit = raw.get("memory_limit", constraints.get("memory_limit"))
        memory_limit_mb = raw.get("memory_limit_mb")
        if memory_limit_mb is None and isinstance(memory_limit, int):
            memory_limit_mb = memory_limit // 1024 // 1024

        problems.append(
            {
                "problem_id": problem_id,
                "title": raw.get("title") or metadata.get("title") or problem_id,
                "problem_type": raw.get("problem_type") or metadata.get("problem_type"),
                "score_type": raw.get("score_type") or metadata.get("score_type"),
                "relative_score_type": raw.get("relative_score_type") or metadata.get("relative_score_type"),
                "time_limit": raw.get("time_limit", constraints.get("time_limit")),
                "memory_limit": memory_limit,
                "memory_limit_mb": memory_limit_mb,
                "statement": statement,
                "statement_sha256": sha256_text(statement),
                "example_input": raw.get("example_input") or "",
                "example_output": raw.get("example_output") or "",
                "metadata": metadata,
                "constraints": constraints,
                "source_json": str(data_path),
            }
        )
    return problems


def build_prompt(problem: dict[str, Any]) -> str:
    lines = [
        "You are a competitive programmer solving an ALE-Bench / AtCoder Heuristic Contest optimization task.",
        f"Write a complete {code_language_name} solution.",
        f"Output ONLY one {code_language_name} source file wrapped in a single ```{code_fence_language} code block.",
        "Do not include explanations, markdown outside the code block, or multiple alternatives.",
        "The program must read from standard input and write to standard output.",
        "",
        "[Problem metadata]",
        f"Problem ID: {problem['problem_id']}",
        f"Title: {problem.get('title') or problem['problem_id']}",
        f"Problem type: {problem.get('problem_type')}",
        f"Score type: {problem.get('score_type')}",
        f"Time limit: {problem.get('time_limit')} seconds",
        f"Memory limit: {problem.get('memory_limit_mb')} MB",
        "",
        "[Problem statement]",
        problem["statement"].rstrip(),
    ]

    if include_examples and (problem.get("example_input") or problem.get("example_output")):
        lines.extend(
            [
                "",
                "[Example input]",
                (problem.get("example_input") or "").rstrip(),
                "",
                "[Example output]",
                (problem.get("example_output") or "").rstrip(),
            ]
        )

    lines.extend(
        [
            "",
            f"Generate the final {code_language_name} solution now.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_payload(messages: list[dict[str, str]], task: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }
    if stop_json:
        payload["stop"] = stop_json
    payload.update(extra_request_json)
    return payload


def post_chat_completion(base_url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int, str]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:
            raw_body = resp.read().decode("utf-8", errors="replace")
            status = int(resp.status)
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
        try:
            parsed = json.loads(raw_body)
        except Exception:
            parsed = {"raw_body": raw_body}
        raise RuntimeError(f"HTTP {status}: {json.dumps(parsed, ensure_ascii=False)[:2000]}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URL error: {exc}") from None
    except socket.timeout:
        raise RuntimeError(f"request timed out after {request_timeout} seconds") from None

    try:
        parsed = json.loads(raw_body)
    except Exception as exc:
        raise RuntimeError(f"Could not parse JSON response: {exc}; body={raw_body[:2000]}") from None

    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected non-object JSON response: {type(parsed).__name__}")
    return parsed, status, endpoint


def extract_response_text(raw_api_output: dict[str, Any]) -> tuple[str, str | None]:
    choices = raw_api_output.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("API response did not contain choices")

    choice0 = choices[0]
    finish_reason = choice0.get("finish_reason") if isinstance(choice0, dict) else None
    text = ""

    if isinstance(choice0, dict):
        message = choice0.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                text = "".join(parts)
        elif isinstance(choice0.get("text"), str):
            text = choice0["text"]

    return text, finish_reason


def task_output_path(task: dict[str, Any]) -> Path:
    return (
        samples_root
        / task["server_label"]
        / task["problem_id"]
        / f"sample_{task['sample_index']:02d}.json"
    )


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    out_path = task_output_path(task)
    if out_path.exists() and not overwrite:
        existing = read_json_if_exists(out_path)
        return {
            "run_status": "skipped_existing",
            "task_id": task["task_id"],
            "output_path": str(out_path),
            "ok": bool(existing.get("ok")) if isinstance(existing, dict) else False,
        }

    prompt_text = task["prompt_text"]
    messages = [{"role": "user", "content": prompt_text}]
    request_payload = build_payload(messages, task)

    attempts: list[dict[str, Any]] = []
    started_at = utc_now()
    total_start = time.perf_counter()

    for attempt_index in range(max_retries + 1):
        attempt_started_at = utc_now()
        attempt_start = time.perf_counter()
        try:
            raw_api_output, http_status, api_endpoint = post_chat_completion(task["server_url"], request_payload)
            latency_s = time.perf_counter() - attempt_start
            response_text, finish_reason = extract_response_text(raw_api_output)
            usage = raw_api_output.get("usage") if isinstance(raw_api_output, dict) else None

            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "started_at": attempt_started_at,
                    "completed_at": utc_now(),
                    "latency_s": latency_s,
                    "http_status": http_status,
                    "ok": True,
                    "error": None,
                }
            )

            record = {
                "ok": True,
                "status": "success",
                "task_id": task["task_id"],
                "server_index": task["server_index"],
                "server_label": task["server_label"],
                "server_url": task["server_url"],
                "api_endpoint": api_endpoint,
                "problem_index": task["problem_index"],
                "problem_id": task["problem_id"],
                "sample_index": task["sample_index"],
                "model": model_name,
                "started_at": started_at,
                "completed_at": utc_now(),
                "latency_s": time.perf_counter() - total_start,
                "attempts": attempts,
                "generation_config": {
                    "model": model_name,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                    "stop": stop_json,
                    "extra_request": extra_request_json,
                    "request_timeout_seconds": request_timeout,
                    "max_retries": max_retries,
                },
                "prompt": {
                    "language": prompt_language,
                    "messages": messages,
                    "text": prompt_text,
                    "sha256": sha256_text(prompt_text),
                },
                "problem": task["problem_metadata"],
                "request_payload": request_payload,
                "response": {
                    "text": response_text,
                    "sha256": sha256_text(response_text),
                    "finish_reason": finish_reason,
                    "usage": usage,
                },
                "raw_api_output": raw_api_output,
                "output_path": str(out_path),
            }
            write_json_atomic(out_path, record)
            return {
                "run_status": "generated",
                "task_id": task["task_id"],
                "output_path": str(out_path),
                "ok": True,
            }

        except Exception as exc:
            latency_s = time.perf_counter() - attempt_start
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "started_at": attempt_started_at,
                    "completed_at": utc_now(),
                    "latency_s": latency_s,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

            if attempt_index < max_retries:
                sleep_s = min(retry_max_sleep, retry_base_sleep * (2**attempt_index))
                sleep_s += random.uniform(0.0, min(1.0, sleep_s * 0.1))
                time.sleep(sleep_s)

    error_record = {
        "ok": False,
        "status": "failed",
        "task_id": task["task_id"],
        "server_index": task["server_index"],
        "server_label": task["server_label"],
        "server_url": task["server_url"],
        "problem_index": task["problem_index"],
        "problem_id": task["problem_id"],
        "sample_index": task["sample_index"],
        "model": model_name,
        "started_at": started_at,
        "completed_at": utc_now(),
        "latency_s": time.perf_counter() - total_start,
        "attempts": attempts,
        "generation_config": {
            "model": model_name,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stop": stop_json,
            "extra_request": extra_request_json,
            "request_timeout_seconds": request_timeout,
            "max_retries": max_retries,
        },
        "prompt": {
            "language": prompt_language,
            "messages": [{"role": "user", "content": task["prompt_text"]}],
            "text": task["prompt_text"],
            "sha256": sha256_text(task["prompt_text"]),
        },
        "problem": task["problem_metadata"],
        "request_payload": request_payload,
        "response": None,
        "raw_api_output": None,
        "output_path": str(out_path),
    }
    write_json_atomic(out_path, error_record)
    append_error(
        {
            "event": "request_failed",
            "task_id": task["task_id"],
            "server_index": task["server_index"],
            "server_label": task["server_label"],
            "server_url": task["server_url"],
            "problem_id": task["problem_id"],
            "sample_index": task["sample_index"],
            "output_path": str(out_path),
            "attempts": attempts,
        }
    )
    return {
        "run_status": "failed",
        "task_id": task["task_id"],
        "output_path": str(out_path),
        "ok": False,
    }


def regenerate_summary() -> dict[str, int]:
    files = sorted(samples_root.glob("*/*/sample_*.json"))
    tmp = summary_jsonl.with_name(summary_jsonl.name + ".tmp")
    counts = {"total_records": 0, "success": 0, "failed": 0}

    summary_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as out:
        for path in files:
            try:
                with path.open("r", encoding="utf-8") as f:
                    record = json.load(f)
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["total_records"] += 1
                if record.get("ok"):
                    counts["success"] += 1
                else:
                    counts["failed"] += 1
            except Exception as exc:
                append_error(
                    {
                        "event": "summary_read_failed",
                        "path": str(path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    tmp.replace(summary_jsonl)
    return counts


def main() -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    samples_root.mkdir(parents=True, exist_ok=True)
    prompts_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    if overwrite:
        error_log.write_text("", encoding="utf-8")

    problems = load_problems_from_json()
    problem_ids = [problem["problem_id"] for problem in problems]
    prompts_by_problem: dict[str, str] = {}
    if len(server_labels) != len(server_urls):
        raise ValueError(f"Expected {len(server_urls)} server labels, got {len(server_labels)}: {server_labels}")
    server_label_slugs = [slugify_label(label) for label in server_labels]
    if len(set(server_label_slugs)) != len(server_label_slugs):
        raise ValueError(f"Server labels must be unique after slugification: {server_label_slugs}")

    for problem in problems:
        prompt_text = build_prompt(problem)
        prompts_by_problem[problem["problem_id"]] = prompt_text
        write_json_atomic(
            prompts_root / f"{problem['problem_id']}.json",
            {
                "problem_id": problem["problem_id"],
                "prompt_language": prompt_language,
                "messages": [{"role": "user", "content": prompt_text}],
                "prompt_text": prompt_text,
                "prompt_sha256": sha256_text(prompt_text),
                "problem": {k: v for k, v in problem.items() if k not in {"statement", "example_input", "example_output"}},
            },
        )

    generation_config = {
        "server_urls": server_urls,
        "server_labels": server_label_slugs,
        "server_label_inputs": server_labels,
        "model_name": model_name,
        "samples_per_problem": samples_per_problem,
        "num_problems": num_problems,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "stop": stop_json,
        "extra_request": extra_request_json,
        "request_timeout_seconds": request_timeout,
        "max_retries": max_retries,
        "retry_base_sleep_seconds": retry_base_sleep,
        "retry_max_sleep_seconds": retry_max_sleep,
        "code_language_name": code_language_name,
        "code_fence_language": code_fence_language,
        "prompt_language": prompt_language,
        "include_examples": include_examples,
        "ale_bench_json_path": str(data_path),
        "output_root": str(output_root),
        "overwrite": overwrite,
    }

    run_config = {
        "created_at": utc_now(),
        "host": socket.gethostname(),
        "expected_generations": len(server_urls) * len(problems) * samples_per_problem,
        "generation_config": generation_config,
        "problem_ids": problem_ids,
        "problems": [
            {k: v for k, v in p.items() if k not in {"statement", "example_input", "example_output"}}
            for p in problems
        ],
    }
    write_json_atomic(metadata_root / "run_config.json", run_config)

    tasks: list[dict[str, Any]] = []
    for server_index, server_url in enumerate(server_urls):
        server_label = server_label_slugs[server_index]
        for problem_index, problem in enumerate(problems):
            problem_id = problem["problem_id"]
            prompt_text = prompts_by_problem[problem_id]
            problem_metadata = {k: v for k, v in problem.items() if k not in {"statement", "example_input", "example_output"}}
            for sample_index in range(samples_per_problem):
                tasks.append(
                    {
                        "task_id": f"{server_label}__{problem_id}__sample_{sample_index:02d}",
                        "server_index": server_index,
                        "server_label": server_label,
                        "server_url": server_url,
                        "problem_index": problem_index,
                        "problem_id": problem_id,
                        "sample_index": sample_index,
                        "prompt_text": prompt_text,
                        "problem_metadata": problem_metadata,
                    }
                )

    expected = len(server_urls) * num_problems * samples_per_problem
    if len(tasks) != expected:
        raise RuntimeError(f"Internal generation count mismatch: expected {expected}, got {len(tasks)}")

    print(f"Output root: {output_root}", file=sys.stderr)
    print(f"ALE-Bench JSON: {data_path}", file=sys.stderr)
    print(f"Tasks: {len(tasks)} ({len(server_urls)} servers x {len(problems)} problems x {samples_per_problem} samples)", file=sys.stderr)
    print(f"Concurrency: {concurrency}", file=sys.stderr)
    print(f"Overwrite: {overwrite}", file=sys.stderr)

    counts = {"generated": 0, "skipped_existing": 0, "failed": 0}
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {executor.submit(run_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                counts["failed"] += 1
                append_error(
                    {
                        "event": "worker_crashed",
                        "task_id": task["task_id"],
                        "server_index": task["server_index"],
                        "server_label": task["server_label"],
                        "server_url": task["server_url"],
                        "problem_id": task["problem_id"],
                        "sample_index": task["sample_index"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                status = "worker_crashed"
                ok = False
            else:
                status = result["run_status"]
                ok = bool(result["ok"])
                counts[status] = counts.get(status, 0) + 1

            with print_lock:
                print(
                    f"[{completed:03d}/{len(tasks):03d}] {status} ok={ok} {task['task_id']}",
                    file=sys.stderr,
                    flush=True,
                )

    summary_counts = regenerate_summary()
    write_json_atomic(
        metadata_root / "final_status.json",
        {
            "completed_at": utc_now(),
            "run_counts": counts,
            "summary_counts": summary_counts,
            "summary_jsonl": str(summary_jsonl),
            "error_log": str(error_log),
        },
    )

    print(f"Summary JSONL: {summary_jsonl}", file=sys.stderr)
    print(f"Error log: {error_log}", file=sys.stderr)
    print(f"Run counts: {counts}", file=sys.stderr)
    print(f"Summary counts: {summary_counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
PY
