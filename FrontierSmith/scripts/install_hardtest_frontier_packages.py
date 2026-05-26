#!/usr/bin/env python3
"""
Install Frontier-CS-style judge packages for HardTests.

Manifest: JSON with valid_problems[].problem_id (e.g. synthetic filter or scripts/sample_hardtest_problems.py).

Primary test data: sigcp/hardtests_tests — decode `test_cases` (pickle+zlib+base64 per dataset card)
and write testdata/*.in, *.ans. See https://huggingface.co/datasets/sigcp/hardtests_tests

Statements and limits: local data/problems/hardtest_<tier>/<pid>/statement.txt when present, else
sigcp/hardtests_problems.

Judging:
  - If test_cases_kit.output_judging_function is non-trivial: write output_judging_function.py;
    VERL uses data_source hardtest_orig and runs g++ + Python OJF locally (see
    verl/utils/reward_score/hardtest_orig.py).
  - Else: generic testlib token chk.cc; VERL falls back to the Frontier-CS HTTP judge (same as before).

Fallback when a pid is missing from hardtests_tests: use public_test_cases from hardtests_problems.

Use --package-prefix hardtest_smp_ for random-sampled runs so dirs do not clash with hardtest_orig_*.

Usage:
  python scripts/install_hardtest_frontier_packages.py
  python scripts/install_hardtest_frontier_packages.py --filtered-json results/hardtest_sampled_800.json --package-prefix hardtest_smp_
  python scripts/install_hardtest_frontier_packages.py --dry-run
  python scripts/install_hardtest_frontier_packages.py --max-tests 64
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import re
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILTERED_JSON = PROJECT_ROOT / "results" / "synthetic_pipeline_filtered_all.json"
PROBLEMS_ROOT = PROJECT_ROOT / "Frontier-CS" / "algorithmic" / "problems"
DATA_PROBLEMS = PROJECT_ROOT / "data" / "problems"

TIER_PREFIXES = ("very_hard_", "medium_hard_", "hard_")

OJF_MIN_LEN = 50

CHK_CC = r"""#include "testlib.h"
#include <string>
using namespace std;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);
    while (!ans.seekEof()) {
        string j = ans.readToken();
        string p = ouf.readToken();
        if (j != p) {
            quitf(_wa, "Expected '%s', found '%s'", j.c_str(), p.c_str());
        }
    }
    if (!ouf.seekEof()) {
        quitf(_wa, "Extra tokens in participant output");
    }
    quitf(_ok, "ok");
    return 0;
}
"""


def decode_testcases(encoded: str) -> list[dict]:
    """Decode sigcp/hardtests_tests test_cases field (dataset card)."""
    return json.loads(
        pickle.loads(
            zlib.decompress(base64.b64decode(encoded.encode("utf-8")))
        )
    )


def hf_pid(problem_id: str) -> str | None:
    for prefix in TIER_PREFIXES:
        if problem_id.startswith(prefix):
            return problem_id[len(prefix) :]
    return None


def tier_from_problem_id(problem_id: str) -> str | None:
    for prefix in TIER_PREFIXES:
        if problem_id.startswith(prefix):
            return prefix.rstrip("_")
    return None


def local_statement_path(problem_id: str) -> Path | None:
    tier = tier_from_problem_id(problem_id)
    base = hf_pid(problem_id)
    if not tier or not base:
        return None
    p = DATA_PROBLEMS / f"hardtest_{tier}" / base / "statement.txt"
    return p if p.is_file() else None


def parse_limits(row: dict) -> tuple[str, str]:
    tl = str(row.get("time_limit") or "2 s")
    ml = str(row.get("memory_limit") or "256 MB")
    tnums = re.findall(r"(\d+(?:\.\d+)?)", tl)
    sec = 2.0
    if tnums:
        v = float(tnums[0])
        low = tl.lower()
        if "ms" in low:
            sec = v / 1000.0
        elif v > 200:
            sec = v / 1000.0
        else:
            sec = v
    sec = max(1.0, min(sec, 60.0))
    mnums = re.findall(r"(\d+(?:\.\d+)?)", ml)
    mb = int(float(mnums[0])) if mnums else 256
    mb = max(32, min(mb, 2048))
    return f"{int(sec)}s", f"{mb}m"


def list_train_parquet_shards() -> list[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files("sigcp/hardtests_tests", repo_type="dataset")
    shards = sorted(f for f in files if f.startswith("data/train-") and f.endswith(".parquet"))
    return shards


def load_tests_rows(needed: set[str]) -> dict[str, dict]:
    """pid -> {test_cases: str, test_cases_kit: dict|None} from parquet shards (partial scan)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    found: dict[str, dict] = {}
    if not needed:
        return found
    for rel in list_train_parquet_shards():
        if len(found) >= len(needed):
            break
        path = hf_hub_download("sigcp/hardtests_tests", rel, repo_type="dataset")
        table = pq.read_table(path, columns=["pid", "test_cases", "test_cases_kit"])
        pcol = table.column("pid").to_pylist()
        for idx, pid in enumerate(pcol):
            if pid not in needed or pid in found:
                continue
            found[pid] = {
                "test_cases": table.column("test_cases")[idx].as_py(),
                "test_cases_kit": table.column("test_cases_kit")[idx].as_py(),
            }
    return found


def write_package(
    dest: Path,
    statement: str,
    cases: list[dict],
    time_s: str,
    mem_m: str,
    ojf_source: str | None,
    max_tests: int,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "statement.txt").write_text(statement, encoding="utf-8")

    td = dest / "testdata"
    if td.exists():
        for p in td.glob("*"):
            p.unlink()
    td.mkdir(parents=True, exist_ok=True)

    if max_tests > 0:
        cases = cases[:max_tests]

    for i, tc in enumerate(cases, start=1):
        inp = tc.get("input")
        out = tc.get("output")
        if inp is None or out is None:
            continue
        inp = str(inp).replace("\r\n", "\n")
        out = str(out).replace("\r\n", "\n")
        (td / f"{i}.in").write_text(inp, encoding="utf-8")
        (td / f"{i}.ans").write_text(out, encoding="utf-8")

    n = len(list(td.glob("*.in")))
    use_ojf = ojf_source and len(ojf_source.strip()) >= OJF_MIN_LEN
    if use_ojf:
        (dest / "output_judging_function.py").write_text(ojf_source.strip() + "\n", encoding="utf-8")
    else:
        (dest / "output_judging_function.py").unlink(missing_ok=True)
    (dest / "chk.cc").write_text(CHK_CC, encoding="utf-8")

    cfg_text = (
        "\n".join(
            [
                "type: default",
                "checker: chk.cc",
                "",
                "# Time and memory limits for the contestant solution",
                f"time: {time_s}",
                f"memory: {mem_m}",
                "",
                "subtasks:",
                "- score: 100",
                f"  n_cases: {n}",
                "",
            ]
        )
    )
    (dest / "config.yaml").write_text(cfg_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-json", type=Path, default=FILTERED_JSON)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-missing", action="store_true", help="Less per-problem logging")
    parser.add_argument("--limit", type=int, default=0, help="Install at most N problems (0 = all)")
    parser.add_argument(
        "--max-tests",
        type=int,
        default=0,
        help="Cap decoded test cases per problem (0 = no cap)",
    )
    parser.add_argument(
        "--package-prefix",
        type=str,
        default="hardtest_orig_",
        help="Problem directory name prefix under Frontier-CS/algorithmic/problems/ (e.g. hardtest_smp_)",
    )
    args = parser.parse_args()

    if not args.filtered_json.is_file():
        print(
            f"Missing {args.filtered_json} — run: python scripts/filter_synthetic_pipeline.py --top-k 800\n"
            f"  or: python scripts/sample_hardtest_problems.py --n 800 --seed 42"
        )
        return

    data = json.loads(args.filtered_json.read_text(encoding="utf-8"))
    valid = data.get("valid_problems", [])
    ids = [v["problem_id"] for v in valid if v.get("problem_id")]
    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    need_hf = {hf_pid(pid) for pid in ids if hf_pid(pid)}

    print("Loading sigcp/hardtests_problems (statements / limits / fallback) ...")
    from datasets import load_dataset

    ds_prob = load_dataset("sigcp/hardtests_problems", split="train")
    rows_prob: dict[str, dict] = {}
    for i in range(len(ds_prob)):
        pid = ds_prob[i]["pid"]
        if pid in need_hf:
            rows_prob[pid] = ds_prob[i]
            if len(rows_prob) == len(need_hf):
                break

    print("Scanning sigcp/hardtests_tests parquet shards for test_cases ...")
    rows_tests = load_tests_rows(need_hf)

    installed = 0
    skipped_no_tests = 0
    skipped_other = 0
    from_tests = 0
    from_public = 0

    for full_id in ids:
        hpid = hf_pid(full_id)
        if not hpid:
            skipped_other += 1
            continue

        prob_row = rows_prob.get(hpid)
        if not prob_row:
            skipped_other += 1
            if not args.skip_missing:
                print(f"  SKIP (no hardtests_problems row): {full_id}")
            continue

        loc = local_statement_path(full_id)
        if loc:
            statement = loc.read_text(encoding="utf-8")
        else:
            statement = prob_row.get("question_content") or ""
        if not statement.strip():
            skipped_other += 1
            print(f"  SKIP (empty statement): {full_id}")
            continue

        cases: list[dict] = []
        ojf_source: str | None = None
        used_hf_tests = False
        trow = rows_tests.get(hpid)
        if trow and trow.get("test_cases"):
            try:
                cases = decode_testcases(trow["test_cases"])
                used_hf_tests = bool(cases)
            except Exception as e:
                if not args.skip_missing:
                    print(f"  WARN decode test_cases failed {full_id}: {e}")
                cases = []
            if used_hf_tests and isinstance(trow.get("test_cases_kit"), dict):
                kit = trow["test_cases_kit"]
                raw = kit.get("output_judging_function")
                if isinstance(raw, str) and len(raw.strip()) >= OJF_MIN_LEN:
                    ojf_source = raw

        if not cases:
            pub = prob_row.get("public_test_cases") or []
            if pub:
                cases = [
                    {"input": p.get("input", ""), "output": p.get("output", "")}
                    for p in pub
                    if p is not None
                ]
                ojf_source = None
                from_public += 1
            else:
                skipped_no_tests += 1
                if not args.skip_missing:
                    print(f"  SKIP (no tests): {full_id}")
                continue
        elif used_hf_tests:
            from_tests += 1

        time_s, mem_m = parse_limits(prob_row)
        dest = PROBLEMS_ROOT / f"{args.package_prefix}{full_id}"
        n_show = len(cases) if args.max_tests <= 0 else min(len(cases), args.max_tests)
        mode = "OJF" if (ojf_source and len(ojf_source.strip()) >= OJF_MIN_LEN) else "token"

        if args.dry_run:
            print(f"  WOULD INSTALL {dest.name}  ({n_show} tests, {time_s}, {mem_m}, {mode})")
            installed += 1
            continue

        write_package(dest, statement, cases, time_s, mem_m, ojf_source, args.max_tests)
        installed += 1

    print(
        f"Done. Installed (or dry-run): {installed}; "
        f"primary hardtests_tests: {from_tests}; "
        f"fallback public_test_cases only: {from_public}; "
        f"skipped no tests: {skipped_no_tests}; "
        f"skipped other: {skipped_other}"
    )


if __name__ == "__main__":
    main()
