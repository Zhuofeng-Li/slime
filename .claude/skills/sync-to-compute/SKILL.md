---
name: sync-to-compute
description: Sync the slime project from dev machine to S3, then provide the command to pull it onto a compute node. Use when user wants to push code changes to the compute node, sync before a training run, or says things like "同步代码", "sync to compute", "push to node", "upload to S3".
---

# Sync to Compute Node

## Environment

| Location | Path |
|---|---|
| Dev machine | `/local/home/zhuofeng/slime/` |
| S3 staging | `s3://rufus-post-training-users-272436634516-us-west-2-an/zhuofeng/slime/` |
| Compute node | `/shared/dev/zhuofeng/slime/` |

## s5cmd Location (dev machine)

```
/local/home/zhuofeng/slime/.venv/bin/s5cmd
```

## Step 1: Dev machine → S3 (if not already done by hook)

The PostToolUse hook in `.claude/settings.json` auto-syncs on every Edit/Write. If needed manually:

```bash
/local/home/zhuofeng/slime/.venv/bin/s5cmd sync \
  --exclude ".git/*" \
  --exclude ".venv/*" \
  --exclude "__pycache__/*" \
  --exclude "*.pyc" \
  --exclude ".pytest_cache/*" \
  /local/home/zhuofeng/slime/ \
  s3://rufus-post-training-users-272436634516-us-west-2-an/zhuofeng/slime/
```

## Step 2: kubectl exec into compute node

```bash
kubectl exec -it <pod-name> -n application-nonprod -- bash
```

## Step 3: MANDATORY — Sync S3 → compute node before running anything

**Always run this immediately after logging in, before any training/eval command:**

```bash
cd /shared/dev/zhuofeng
s5cmd sync \
  "s3://rufus-post-training-users-272436634516-us-west-2-an/zhuofeng/slime/*" \
  /shared/dev/zhuofeng/slime/
```

Note: `s5cmd sync` from S3 requires a trailing `/*` wildcard on the source path.

## Step 4: Run your script

```bash
cd /shared/dev/zhuofeng/slime
bash scripts/<your-training-script>.sh
```

## Notes

- Always `cd /shared/dev/zhuofeng`, NOT `/shared/dev/xliucr` (the default login dir after kubectl exec)
- Never skip Step 3 — the compute node may have stale code from a previous session
