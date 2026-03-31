"""
Read Q-Q data from stdin (CSV: empirical,theoretical) and display the plot.

Usage:
    cargo run -p quantflow-core --example hawkes_qq | python3 scripts/plot_qq.py
"""
from __future__ import annotations

import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Read CSV from stdin ───────────────────────────────────────────────────────
lines = sys.stdin.read().strip().splitlines()
header, *rows = lines
empirical = []
theoretical = []
for row in rows:
    e, t = row.split(",")
    empirical.append(float(e))
    theoretical.append(float(t))

empirical = np.array(empirical)
theoretical = np.array(theoretical)

# ── Layout: Q-Q plot + residuals ─────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.05)

ax_qq   = fig.add_subplot(gs[0, :])  # Q-Q plot spans full width
ax_res  = fig.add_subplot(gs[1, :], sharex=ax_qq)

# ── Q-Q plot ─────────────────────────────────────────────────────────────────
ax_qq.scatter(theoretical, empirical, s=1.5, alpha=0.4, color="#2196F3", rasterized=True)

lim = max(theoretical.max(), empirical.max()) * 1.05
ax_qq.plot([0, lim], [0, lim], color="#F44336", linewidth=1.2, label="Perfect fit")
ax_qq.set_xlim(0, lim)
ax_qq.set_ylim(0, lim)
ax_qq.set_ylabel("Empirical quantiles  Λ_k", fontsize=12)
ax_qq.set_title(
    "Q-Q Plot — Time-rescaled inter-arrival times vs Exp(1)\n"
    "(Points on diagonal = model fits data well)",
    fontsize=13,
)
ax_qq.legend(fontsize=10)
ax_qq.tick_params(labelbottom=False)

# ── Residuals ─────────────────────────────────────────────────────────────────
residuals = empirical - theoretical
ax_res.axhline(0, color="#F44336", linewidth=1.0)
ax_res.scatter(theoretical, residuals, s=1.5, alpha=0.4, color="#2196F3", rasterized=True)
ax_res.set_xlabel("Theoretical quantiles  −ln(1 − k/n)", fontsize=12)
ax_res.set_ylabel("Residual", fontsize=11)

n = len(empirical)
ks_d = np.max(np.abs(
    np.arange(1, n + 1) / n - (1 - np.exp(-np.sort(empirical)))
))
fig.text(
    0.99, 0.01,
    f"n = {n:,}   KS D = {ks_d:.4f}",
    ha="right", va="bottom", fontsize=9, color="#555555",
)

plt.tight_layout()
plt.show()
