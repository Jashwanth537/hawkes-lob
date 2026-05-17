#!/usr/bin/env python3
"""
Generate two plots from Hawkes calibration results CSVs.
Usage: python3 plot_results.py results/results_btcusdt.csv results/results_ethusdt.csv
"""
import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: plot_results.py <btcusdt_csv> <ethusdt_csv>")
    sys.exit(1)

btc_path = sys.argv[1]
eth_path = sys.argv[2]
out_dir  = os.path.dirname(os.path.abspath(btc_path))

btc = pd.read_csv(btc_path)
eth = pd.read_csv(eth_path)

# Keep only converged windows
btc = btc[btc['converged'] == 1].copy()
eth = eth[eth['converged'] == 1].copy()
print(f"BTC converged windows: {len(btc)}  ETH: {len(eth)}")

MARKET_OPENS = {
    'London open':   8,
    'NY pre-market': 13,
    'NY open':       14,
    'Asia open':     22,
}

COLORS = {'BTC': '#1f77b4', 'ETH': '#ff7f0e'}

def plot_by_hour(ax, df_btc, df_eth, col, title):
    for label, df, color in [('BTCUSDT', df_btc, COLORS['BTC']),
                               ('ETHUSDT', df_eth, COLORS['ETH'])]:
        g = df.groupby('utc_hour')[col]
        med = g.median()
        q25 = g.quantile(0.25)
        q75 = g.quantile(0.75)
        hours = med.index
        ax.plot(hours, med.values, color=color, label=label, linewidth=1.8)
        ax.fill_between(hours, q25.values, q75.values,
                        color=color, alpha=0.2)
    for label, h in MARKET_OPENS.items():
        ax.axvline(h, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.text(h + 0.1, ax.get_ylim()[1] * 0.95, label,
                fontsize=6, color='grey', rotation=90, va='top')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('UTC Hour')
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# ── Plot 1: kernel norms 2×2 ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Hawkes Kernel Norms by UTC Hour — 6 Days Binance Data", fontsize=13)

combos = [
    (axes[0,0], 'phi_BB', 'φ_BB (BUY self-excitation)'),
    (axes[0,1], 'phi_SS', 'φ_SS (SELL self-excitation)'),
    (axes[1,0], 'phi_BS', 'φ_BS (SELL→BUY cross-excitation)'),
    (axes[1,1], 'phi_SB', 'φ_SB (BUY→SELL cross-excitation)'),
]
for ax, col, title in combos:
    plot_by_hour(ax, btc, eth, col, title)
    # Fix vlines after ylim is set
    for label, h in MARKET_OPENS.items():
        ax.axvline(h, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

plt.tight_layout()
p1 = os.path.join(out_dir, 'kernel_norms_by_hour.png')
fig.savefig(p1, dpi=300, bbox_inches='tight')
print(f"Saved: {p1}")
plt.close(fig)

# ── Plot 2: total endogeneity ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Total Endogeneity (η) by UTC Hour", fontsize=13)
plot_by_hour(ax, btc, eth, 'eta_total', '')
ax.set_ylabel('η_total = (α_BB + α_SS + α_BS + α_SB) / β')
ax.set_title("Total Endogeneity (η) by UTC Hour", fontsize=13)
for label, h in MARKET_OPENS.items():
    ax.axvline(h, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

plt.tight_layout()
p2 = os.path.join(out_dir, 'total_excitation_by_hour.png')
fig.savefig(p2, dpi=300, bbox_inches='tight')
print(f"Saved: {p2}")
plt.close(fig)
