# FrontierCS Qwen3.5-27B Runbook

This README is only for the Qwen3.5-27B FrontierCS setup.

Run commands from the repo root:

```bash
cd /fsx-alignment/home/zhuofeng/slime
mkdir -p logs/frontiercs
```

## Data Source

Raw FrontierCS problems are under:

```text
FrontierSmith/Frontier-CS/algorithmic/problems/problems
```

`launch.sh` prepares `data/frontiercs/{train,val}.jsonl` from those problems unless `--skip-data` is passed.
**Currently, the prepared FrontierCS dataset only includes problem `0`.**

## One-Click 27B Launch

Use `scripts/frontier_cs/launch.sh` with `--model qwen3.5-27B`. This selects:

```text
scripts/frontier_cs/run-frontiercs-qwen3.5-27B-debug.sh
```

First run, or when data / judge / model state is uncertain:

```bash
RUN_NAME="Qwen3.5-27B-frontiercs-debug-$(date +%Y%m%d-%H%M%S)"
LOG="logs/frontiercs/${RUN_NAME}.log"

env RUN_NAME="${RUN_NAME}" \
  bash scripts/frontier_cs/launch.sh \
    --base-dir /fsx-alignment/home/zhuofeng/slime \
    --num-gpus 8 \
    --model qwen3.5-27B \
  2>&1 | tee "${LOG}"
```

If data and judge are already ready:

```bash
RUN_NAME="Qwen3.5-27B-frontiercs-debug-$(date +%Y%m%d-%H%M%S)"
LOG="logs/frontiercs/${RUN_NAME}.log"

env RUN_NAME="${RUN_NAME}" \
  bash scripts/frontier_cs/launch.sh \
    --base-dir /fsx-alignment/home/zhuofeng/slime \
    --num-gpus 8 \
    --model qwen3.5-27B \
    --skip-data \
    --skip-judge \
  2>&1 | tee "${LOG}"
```

## 27B Debug Run With Reward Filter

This keeps the 27B sampling defaults, keeps max response length unchanged, uses `n=8`, and enables the existing non-zero/std reward filter through env passthrough:

```bash
RUN_NAME="Qwen3.5-27B-frontiercs-debug-thinking-n8-nonzero-std-filter-memfrac075-$(date +%Y%m%d-%H%M%S)"
LOG="logs/frontiercs/${RUN_NAME}.log"

env RUN_NAME="${RUN_NAME}" \
  NUM_ROLLOUT=10 \
  N_SAMPLES_PER_PROMPT=8 \
  GLOBAL_BATCH_SIZE=8 \
  SGLANG_MEM_FRACTION_STATIC=0.75 \
  DYNAMIC_SAMPLING_FILTER_PATH=slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
  bash scripts/frontier_cs/launch.sh \
    --base-dir /fsx-alignment/home/zhuofeng/slime \
    --num-gpus 8 \
    --model qwen3.5-27B \
    --skip-data \
    --skip-judge \
  2>&1 | tee "${LOG}"
```

## Important Defaults

- `ROLLOUT_MAX_RESPONSE_LEN` / max response must stay `81920` unless explicitly requested otherwise.
- Do not change sampling temperature or related sampling params unless explicitly requested.
- For parameter changes, use env passthrough and put the change in `RUN_NAME`.
- Save rollout dumps when debugging reward behavior; the 27B debug script saves rollout data.
- Use `tee` and keep logs under `logs/frontiercs/`.

## Monitoring

Find the latest log:

```bash
ls -lt logs/frontiercs | head
```

Follow it:

```bash
tail -n 200 -f logs/frontiercs/<run-log>.log
```

Check Ray job status and generation progress:

```bash
CID=$(sudo docker ps --filter ancestor=slimerl/slime:latest --format '{{.ID}}' | head -1)
sudo docker exec "${CID}" bash -lc '
  ray job list
  grep -h "POST /generate HTTP/1.1\" 200 OK" /tmp/ray/session_latest/logs/worker-*-02000000-*.out 2>/dev/null | wc -l
'
```

Inspect saved rollout rewards:

```bash
CID=$(sudo docker ps --filter ancestor=slimerl/slime:latest --format '{{.ID}}' | head -1)
sudo docker exec "${CID}" bash -lc 'python3 - <<PY
import os, torch
base="/root/logs/frontiercs_rollouts/<RUN_NAME>"
for fn in sorted(os.listdir(base), key=lambda x: int(x[:-3]) if x.endswith(".pt") and x[:-3].isdigit() else 10**9):
    if not fn.endswith(".pt") or not fn[:-3].isdigit():
        continue
    rid=int(fn[:-3])
    obj=torch.load(os.path.join(base, fn), map_location="cpu")
    rewards=[float(s.get("reward") or 0.0) for s in obj.get("samples", [])]
    if rewards:
        print(rid, "mean", sum(rewards)/len(rewards), "max", max(rewards), "pos", sum(r > 0 for r in rewards), "/", len(rewards))
PY'
```

Watch for:

- judge health and request errors
- SGLang worker unhealthy / fatal OOM
- rollout generation progress
- saved rollout `raw_reward`, max reward, and positive sample count
- training step metrics and tracebacks

## Logs And Artifacts

- Main logs: `logs/frontiercs/<RUN_NAME>.log`
- Rollout dumps inside container: `/root/logs/frontiercs_rollouts/<RUN_NAME>/`
- Host rollout path usually maps to: `logs/frontiercs_rollouts/<RUN_NAME>/`
- Checkpoints use the script's `CKPT_DIR` / `SAVE_DIR` logic.

## Cleanup

Stop a Ray job:

```bash
CID=$(sudo docker ps --filter ancestor=slimerl/slime:latest --format '{{.ID}}' | head -1)
sudo docker exec "${CID}" ray job stop <JOB_ID>
```

Kill GPU processes only when explicitly needed:

```bash
sudo kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -u)
```

## Experiment Tracking

For each launched 27B experiment, append one row to:

```bash
scripts/frontier_cs/exp.csv
```

Use `RUN_NAME` as `ExperienceName`, `27B` as `ModelSize`, include the exact command, and set fail reason to `-` unless the run failed.
