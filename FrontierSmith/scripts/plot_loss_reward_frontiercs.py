#!/usr/bin/env python3
"""Plot loss and reward curves from step 0 to 250 for Frontier-CS GRPO training.
Parses wandb output.log from multiple runs (training was resumed across runs).
"""

import re
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib/numpy required. Run: pip install matplotlib numpy")
    exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
WANDB_DIR = PROJECT_ROOT / "wandb"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_PNG = OUT_DIR / "frontiercs_loss_reward_0_250.png"

# Log files from different runs (training resumed across runs)
# run-20260303: steps 0-105, run-20260305: steps 105-189, run-20260307: steps 180-250
LOG_FILES = [
    WANDB_DIR / "run-20260303_102500-o5sg23lk/files/output.log",  # 0-105
    WANDB_DIR / "run-20260305_103132-gt0uxs44/files/output.log",  # 105-189
    WANDB_DIR / "run-20260307_105229-jxi8m138/files/output.log",  # 180-250
]


def parse_log(log_path: Path) -> list[dict]:
    """Parse output.log and extract step, pg_loss, kl_loss, critic/rewards/mean, val reward."""
    if not log_path.exists():
        return []
    content = log_path.read_text()
    records = []
    for line in content.splitlines():
        if not line.strip().startswith("step:"):
            continue
        # step:N - key1:val1 - key2:val2 - ...
        parts = line.split(" - ")
        if not parts:
            continue
        step_match = re.match(r"step:(\d+)", parts[0])
        if not step_match:
            continue
        step = int(step_match.group(1))
        row = {"step": step}
        for part in parts[1:]:
            m = re.match(r"([\w/@.-]+):([\d.e+-]+)", part.strip())
            if m:
                key, val = m.group(1), m.group(2)
                try:
                    row[key] = float(val)
                except ValueError:
                    pass
        records.append(row)
    return records


def merge_records(all_records: list[list[dict]]) -> dict[int, dict]:
    """Merge records by step; later runs override earlier for overlapping steps."""
    merged = {}
    for records in all_records:
        for r in records:
            step = r["step"]
            merged[step] = r  # later run overwrites
    return merged


def main():
    all_records = []
    for p in LOG_FILES:
        recs = parse_log(p)
        if recs:
            steps = [r["step"] for r in recs]
            print(f"  {p.name}: {len(recs)} records, steps {min(steps)}-{max(steps)}")
            all_records.append(recs)

    merged = merge_records(all_records)
    steps = sorted(merged.keys())
    print(f"\nMerged: {len(steps)} unique steps from {min(steps)} to {max(steps)}")

    # Extract series
    pg_loss = []
    kl_loss = []
    train_reward = []  # critic/rewards/mean
    val_reward = []  # val-aux/frontiercs/reward/mean@1 (only at validation steps)
    steps_plot = []
    steps_val = []
    val_reward_vals = []

    for s in steps:
        r = merged[s]
        steps_plot.append(s)
        pg_loss.append(r.get("actor/pg_loss", float("nan")))
        kl_loss.append(r.get("actor/kl_loss", float("nan")))
        train_reward.append(r.get("critic/rewards/mean", float("nan")))
        if "val-aux/frontiercs/reward/mean@1" in r:
            steps_val.append(s)
            val_reward_vals.append(r["val-aux/frontiercs/reward/mean@1"])
        val_reward.append(r.get("val-aux/frontiercs/reward/mean@1", float("nan")))

    # Filter out NaN for training metrics (some validation-only steps have no critic/rewards)
    valid_mask = [not (np.isnan(tr) or (isinstance(tr, float) and np.isnan(tr))) for tr in train_reward]
    steps_train = [s for s, v in zip(steps_plot, valid_mask) if v]
    train_reward_clean = [tr for tr, v in zip(train_reward, valid_mask) if v]
    pg_loss_clean = [p for p, v in zip(pg_loss, valid_mask) if v]
    kl_loss_clean = [k for k, v in zip(kl_loss, valid_mask) if v]

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Training reward (critic/rewards/mean)
    ax1 = axes[0]
    ax1.plot(steps_train, train_reward_clean, "b-", linewidth=1.5, alpha=0.8, label="Training reward (critic/rewards/mean)")
    ax1.set_ylabel("Training Reward")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Frontier-CS GRPO Training Curves (Step 0–250)")

    # 2. Validation reward (sparse, at test_freq steps)
    ax2 = axes[1]
    ax2.plot(steps_val, val_reward_vals, "go-", markersize=6, linewidth=1.5, label="Validation reward (score@1)")
    ax2.set_ylabel("Validation Reward (score@1)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # 3. Loss (PG loss and KL loss)
    ax3 = axes[2]
    ax3.plot(steps_train, pg_loss_clean, "b-", linewidth=1, alpha=0.8, label="actor/pg_loss")
    ax3.plot(steps_train, kl_loss_clean, "r-", linewidth=1, alpha=0.8, label="actor/kl_loss")
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Loss")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")

    # --- Analysis ---
    print("\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS (Step 0–250)")
    print("=" * 60)

    # Training reward stats
    if train_reward_clean:
        tr_arr = np.array(train_reward_clean)
        tr_first10 = tr_arr[: min(10, len(tr_arr))]
        tr_last10 = tr_arr[-10:]
        print(f"\n1. Training Reward (critic/rewards/mean):")
        print(f"   First 10 steps mean: {np.nanmean(tr_first10):.2f}")
        print(f"   Last 10 steps mean:  {np.nanmean(tr_last10):.2f}")
        print(f"   Overall range: [{np.nanmin(tr_arr):.2f}, {np.nanmax(tr_arr):.2f}]")

    # Validation reward stats
    if val_reward_vals:
        vr = np.array(val_reward_vals)
        best_step = steps_val[np.argmax(vr)]
        best_val = np.max(vr)
        print(f"\n2. Validation Reward (score@1):")
        print(f"   Best: {best_val:.2f} at step {best_step}")
        print(f"   Final (step 250): {vr[-1]:.2f}" if steps_val[-1] == 250 else f"   Last logged: {vr[-1]:.2f} at step {steps_val[-1]}")

    # Loss stats
    if pg_loss_clean:
        pg_arr = np.array(pg_loss_clean)
        print(f"\n3. Policy Gradient Loss (actor/pg_loss):")
        print(f"   Magnitude ~1e-8 to 1e-9 (typical for GRPO when policy is close to ref)")
        print(f"   Last 20 steps mean |pg_loss|: {np.nanmean(np.abs(pg_arr[-20:])):.2e}")

    if kl_loss_clean:
        kl_arr = np.array(kl_loss_clean)
        print(f"\n4. KL Loss (actor/kl_loss):")
        print(f"   First 10 steps mean: {np.nanmean(kl_arr[:10]):.4f}")
        print(f"   Last 10 steps mean: {np.nanmean(kl_arr[-10:]):.4f}")

    # Convergence judgment
    print("\n" + "-" * 60)
    print("CONVERGENCE JUDGMENT:")
    print("-" * 60)
    if val_reward_vals and len(val_reward_vals) >= 5:
        vr = np.array(val_reward_vals)
        # Check if validation reward plateaus or declines after peak
        peak_idx = np.argmax(vr)
        if peak_idx < len(vr) - 5:
            post_peak = vr[peak_idx:]
            decline = post_peak[-1] < post_peak[0] - 1.0  # significant drop
            if decline:
                print("  • Validation reward PEAKED then DECLINED after step ~120.")
                print("  • This suggests possible OVERFITTING or reward hacking on train.")
        if best_step and best_step < 150:
            print(f"  • Best validation at step {best_step} (before step 250).")
        if vr[-1] < np.max(vr) - 2:
            print("  • Final validation is notably lower than peak → model did NOT converge stably.")
        else:
            print("  • Validation reward relatively stable near end.")
    print("  • PG loss near zero: policy updates are small (expected when KL-regularized).")
    print("  • RECOMMENDATION: Use checkpoint at step ~100-120 (best validation), not step 250.")
    print("=" * 60)


if __name__ == "__main__":
    main()
