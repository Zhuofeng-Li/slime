#!/usr/bin/env python3
"""Plot training loss from WandB output.log"""
import re
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    exit(1)

LOG = Path(__file__).parent.parent / "wandb/run-20260301_100414-78g2h9o1/files/output.log"
OUT = Path(__file__).parent.parent / "wandb/run-20260301_100414-78g2h9o1/training_loss.png"

content = LOG.read_text()
steps, pg_loss, kl_loss = [], [], []
for m in re.finditer(r'step:(\d+).*?actor/pg_loss:([\d.e+-]+).*?actor/kl_loss:([\d.e+-]+)', content):
    steps.append(int(m.group(1)))
    pg_loss.append(float(m.group(2)))
    kl_loss.append(float(m.group(3)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(steps, pg_loss, 'b-o', markersize=4, linewidth=1, label='actor/pg_loss')
ax1.set_ylabel('PG Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('VERL GRPO Training Loss (8h run, 40 steps)')
ax2.plot(steps, kl_loss, 'r-o', markersize=4, linewidth=1, label='actor/kl_loss')
ax2.set_xlabel('Step')
ax2.set_ylabel('KL Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=120, bbox_inches='tight')
print(f"Saved: {OUT}")
