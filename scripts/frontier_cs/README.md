# FrontierCS Qwen3.5-27B — 64-GPU (8×H200)

## Prerequisites

```bash
# 1. Prepare data
uv run python scripts/frontier_cs/prepare_frontiercs_jsonl.py --full-for-both
# → ${BASE_FOLDER}/data/frontiercs_full/train.jsonl
# → ${BASE_FOLDER}/data/frontiercs/val.jsonl

# 2. Download model weights (HF format)
# → ${BASE_FOLDER}/models/Qwen3.5-27B/

# 3. Convert to torch_dist format
# → ${BASE_FOLDER}/models/Qwen3.5-27B_torch_dist/

# 4. Start FrontierCS judge on port 8082 (see FrontierSmith/Frontier-CS/)

# 5. Create log dir
mkdir -p ${BASE_FOLDER}/logs/frontiercs_rollouts/
```

## Launch

```bash
MASTER_ADDR=<worker-0-ip> \
HOSTFILE=/path/to/hostfile \
WANDB_KEY=<your_key> \
BASE_FOLDER=/shared/dev/zhuofeng/slime \
bash scripts/frontier_cs/run-frontiercs-qwen3.5-27B_debug-64_h200.sh
```

`HOSTFILE`: one worker IP per line (all 8 nodes).

The script kills stale processes, starts Ray on the head node, SSHs into workers to join the cluster, then submits the training job.

## Key Defaults

| Var | Default |
|---|---|
| `NUM_ROLLOUT` | 2000 |
| `ROLLOUT_MAX_RESPONSE_LEN` | 81920 |
| `N_SAMPLES_PER_PROMPT` | 8 |
| `GLOBAL_BATCH_SIZE` | 256 |
| `SGLANG_MEM_FRACTION_STATIC` | 0.80 |
| `FRONTIER_JUDGE_URL` | http://localhost:8082 |

Override any default via env var. Put changes in `RUN_NAME` for tracking.

## Monitor

```bash
# Ray job status
ray job list

# Follow logs
ray job logs --follow <JOB_ID>

# Rollout rewards
python3 - <<PY
import os, torch
base="${BASE_FOLDER}/logs/frontiercs_rollouts/<RUN_NAME>"
for fn in sorted(os.listdir(base)):
    if not fn.endswith(".pt"): continue
    obj = torch.load(os.path.join(base, fn), map_location="cpu")
    rewards = [float(s.get("reward") or 0) for s in obj.get("samples", [])]
    if rewards:
        print(fn, "mean", sum(rewards)/len(rewards), "pos", sum(r>0 for r in rewards), "/", len(rewards))
PY
```

## Experiment Tracking

Append a row to `scripts/frontier_cs/exp.csv` for each run: `RUN_NAME`, `ModelSize=27B`, command, fail reason (`-` if successful).
