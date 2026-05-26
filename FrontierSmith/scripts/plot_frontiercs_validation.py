#!/usr/bin/env python3
"""Plot Frontier-CS validation metrics (step 0-250) from checkpoint evals."""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "outputs" / "frontiercs_checkpoint_evals.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "frontiercs_validation_0_250.png"

# Full validation data (step 0-250) - from eval runs
DATA = [
    ("base", 0.7372, 1.3074, 3.6408),
    (0, 1.4769, 1.0218, 4.5216),
    (10, 1.8630, 2.6658, 7.6644),
    (20, 4.6922, 4.7755, 9.7800),
    (30, 2.7345, 3.7956, 8.4617),
    (40, 3.7756, 4.5637, 9.6196),
    (50, 5.1873, 4.9316, 9.4009),
    (60, 4.0712, 3.4325, 7.8278),
    (70, 5.0399, 3.1168, 7.5716),
    (80, 3.5751, 3.2022, 8.2099),
    (90, 5.2756, 3.8223, 7.5603),
    (100, 6.8625, 5.8863, 9.0169),
    (110, 6.5304, 7.2071, 12.0669),
    (120, 8.6738, 7.8585, 13.0133),
    (130, 7.5230, 7.0533, 11.9421),
    (140, 6.3679, 6.8165, 10.5599),
    (150, 7.1155, 6.6313, 10.2747),
    (160, 7.1880, 7.4925, 11.5123),
    (170, 5.9036, 6.8101, 11.7125),
    (180, 7.3025, 7.1035, 10.9648),
    (190, 8.0335, 7.1005, 11.1198),
    (200, 5.9482, 6.0531, 10.8646),
    (210, 5.7096, 5.0810, 8.6302),
    (220, 5.6924, 5.9781, 10.1472),
    (230, 4.8969, 5.9763, 9.1608),
    (240, 6.0001, 6.4512, 8.5988),
    (250, 4.0036, 3.7723, 6.9692),
]


def main():
    df = pd.DataFrame(DATA, columns=["step", "score_at_1", "avg_score_at_5", "best_score_at_5"])

    # Filter to numeric steps only for plotting (exclude "base" for x-axis)
    plot_df = df[df["step"].apply(lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()))].copy()
    plot_df["step"] = pd.to_numeric(plot_df["step"])

    # Print table
    print("\n" + "=" * 60)
    print("Frontier-CS Validation Results (Step 0 - 250)")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    # Plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_df["step"], plot_df["score_at_1"], marker="o", linewidth=2, label="score@1", markersize=4)
    ax.plot(plot_df["step"], plot_df["avg_score_at_5"], marker="s", linewidth=2, label="avg_score@5", markersize=4)
    ax.plot(plot_df["step"], plot_df["best_score_at_5"], marker="^", linewidth=2, label="best_score@5", markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Score")
    ax.set_title("Frontier-CS Validation Metrics (Step 0 - 250)")
    ax.set_xticks(plot_df["step"][::2])  # every other step for readability
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    print(f"\nChart saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
