#!/usr/bin/env python3
"""Phase 0: reproduce constant-mu per-UTC-hour figure on 1 month of BTC.

Matched (RAW streams, constant-mu soe Hawkes beta=[100,1], one
fit per (day, UTC hour), median across days per hour). Added across-day bootstrap CIs and
the open-vs-placebo contrast that we will compare in Phase 1 mu-flexibility.

Reproducibility: n_workers=1 (no thread-RNG multistart wobble). Conditional on the 2-exp
kernel throughout.

Outputs (results/):
  phase0_btc_const_<month>.csv          per-window constant-mu fits
  phase0_kernel_norms_by_hour.png       7-panel by-hour figure with across-day CI bands
  phase0_open_contrast.csv              open(UTC-13) vs placebo(UTC-3) contrast + CI
  phase0_tie_fraction_by_hour.csv       tie diagnostics
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_base = Path(__file__).resolve().parent.parent
HAWKES = _base if (_base / "src").is_dir() else _base / "hawkes"
sys.path.insert(0, str(HAWKES))
from src.data import load_one_file                  # noqa: E402
from src.calibration import rolling_calibration     # noqa: E402

KERNELS = ['phi_BB1', 'phi_SS1', 'phi_SB2', 'phi_BS2']   # fast-self x2, slow-cross x2
PANEL_VARS = ['mu_B', 'mu_S', 'eta'] + KERNELS
OPEN_HOUR, PLACEBO_HOUR = 13, 3
HOURS = np.arange(24)


def across_day_median_ci(values, B=2000, seed=0, lo=2.5, hi=97.5):
    """Median + across-day percentile-bootstrap CI from a 1-D array of daily values."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    if len(v) == 1:
        return float(v[0]), np.nan, np.nan, 1
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(v, size=(B, len(v)), replace=True), axis=1)
    return float(np.median(v)), float(np.percentile(boot, lo)), float(np.percentile(boot, hi)), len(v)


def tie_fraction_by_hour(timestamp_us, B_label='stream'):
    """Fraction of events sharing a microsecond timestamp with the previous event,
    grouped by UTC hour. Ties (dt=0) are the aggregation concern for the fast kernel."""
    ts = np.sort(timestamp_us.astype(np.int64))
    hour = ((ts // 1_000_000) % 86400) // 3600
    is_tie = np.zeros(len(ts), dtype=bool)
    is_tie[1:] = np.diff(ts) == 0
    out = {}
    for h in HOURS:
        m = hour == h
        out[h] = float(is_tie[m].mean()) if m.any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', default='2025-12')
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('-B', type=int, default=2000, help='across-day bootstrap reps')
    a = ap.parse_args()

    pq = HAWKES / 'data' / 'download' / f'{a.symbol}-aggTrades-{a.month}.parquet'
    if not pq.exists():
        pq = HAWKES / 'data' / 'download' / a.symbol / f'{a.symbol}-aggTrades-{a.month}.parquet'
    print(f"loading {pq.name} ...")
    df = load_one_file(str(pq))                       # dedup applied
    buy_us = df.loc[df.event_type == 0, 'timestamp_us'].values
    sell_us = df.loc[df.event_type == 1, 'timestamp_us'].values
    buy = buy_us / 1e6
    sell = sell_us / 1e6
    print(f"  {a.symbol} {a.month}: buy={len(buy):,} sell={len(sell):,}  (RAW streams, matches advisor fig)")

    # --- Phase 0.3: tie fraction by hour (raw streams) ---
    tie_buy = tie_fraction_by_hour(buy_us)
    tie_sell = tie_fraction_by_hour(sell_us)
    tie_df = pd.DataFrame({'utc_hour': HOURS,
                           'tie_frac_buy': [tie_buy[h] for h in HOURS],
                           'tie_frac_sell': [tie_sell[h] for h in HOURS]})
    tie_df.to_csv(HAWKES / 'results' / 'phase0_tie_fraction_by_hour.csv', index=False)
    overall_tie = (np.diff(np.sort(buy_us)) == 0).mean()
    print(f"  tie fraction (buy, overall) = {overall_tie:.4f}; "
          f"tie timescale = 1 us << fast kernel 1/beta1 = {1/100*1e3:.0f} ms << slow 1/beta2 = {1/1*1e3:.0f} ms")

    # --- Phase 0.1: constant-mu rolling calibration, n_workers=1 (reproducible) ---
    print("rolling calibration (constant-mu, RAW, n_workers=1) ... this is the slow part")
    df_fit = rolling_calibration(buy, sell, a.symbol, n_workers=1)
    out_csv = HAWKES / 'results' / f'phase0_{a.symbol.lower()}_const_{a.month}.csv'
    df_fit.to_csv(out_csv, index=False)
    print(f"  saved {out_csv.name}  ({len(df_fit)} windows; days/hour ~ {len(df_fit)//24})")

    # --- per-hour median + across-day CI for each panel var ---
    rows = []
    for v in PANEL_VARS:
        for h in HOURS:
            vals = df_fit.loc[df_fit.utc_hour == h, v].values
            med, clo, chi, n = across_day_median_ci(vals, B=a.B)
            rows.append({'var': v, 'utc_hour': h, 'median': med, 'ci_lo': clo, 'ci_hi': chi, 'n_days': n})
    by_hour = pd.DataFrame(rows)

    # --- Phase 0.1 figure: 7 panels, across-day CI bands, open marked ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle(f'Phase 0 baseline: constant-mu per UTC hour — {a.symbol} {a.month} (RAW streams)\n'
                 f'median across days +/- across-day bootstrap 95% CI (B={a.B}); '
                 f'conditional on 2-exp kernel beta=[100,1]',
                 fontsize=12, fontweight='bold')
    for ax, v in zip(axes.ravel(), PANEL_VARS):
        d = by_hour[by_hour['var'] == v].sort_values('utc_hour')
        ax.plot(d.utc_hour, d['median'], 'o-', color='#1f77b4', ms=4)
        ax.fill_between(d.utc_hour, d.ci_lo, d.ci_hi, alpha=0.25, color='#1f77b4')
        ax.axvline(OPEN_HOUR, color='#F44336', lw=1.3, alpha=0.8)
        ax.axvline(PLACEBO_HOUR, color='#888', lw=1.0, ls='--', alpha=0.7)
        ax.set_title(v, fontweight='bold'); ax.set_xlabel('UTC hour'); ax.set_xticks(range(0, 24, 4))
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(PANEL_VARS):]:
        ax.axis('off')
    axes.ravel()[len(PANEL_VARS)].text(0.05, 0.5,
        'red = NYSE open (UTC-13)\ngrey dash = placebo (UTC-3)',
        fontsize=10, va='center')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = HAWKES / 'results' / 'phase0_kernel_norms_by_hour.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {fig_path.name}")

    # --- open vs placebo contrast (per kernel) with across-day bootstrap CI ---
    # contrast per day-hour is not paired; use hour-pooled bootstrap of (median13 - median3)
    def contrast_ci(v, B=a.B, seed=1):
        v13 = df_fit.loc[df_fit.utc_hour == OPEN_HOUR, v].values
        v3 = df_fit.loc[df_fit.utc_hour == PLACEBO_HOUR, v].values
        v13 = v13[~np.isnan(v13)]; v3 = v3[~np.isnan(v3)]
        if len(v13) < 2 or len(v3) < 2:
            return np.nan, np.nan, np.nan, len(v13), len(v3)
        rng = np.random.default_rng(seed)
        b13 = np.median(rng.choice(v13, size=(B, len(v13)), replace=True), axis=1)
        b3 = np.median(rng.choice(v3, size=(B, len(v3)), replace=True), axis=1)
        diff = b13 - b3
        return (float(np.median(v13) - np.median(v3)),
                float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)),
                len(v13), len(v3))

    crows = []
    print(f"\nOpen (UTC-{OPEN_HOUR}) vs placebo (UTC-{PLACEBO_HOUR}) contrast — constant-mu baseline (spec A):")
    print(f"  {'var':<8}{'med_open':>10}{'med_plac':>10}{'contrast':>10}{'95% CI':>22}{'excl 0':>8}")
    for v in PANEL_VARS:
        c, lo, hi, n13, n3 = contrast_ci(v)
        m13 = df_fit.loc[df_fit.utc_hour == OPEN_HOUR, v].median()
        m3 = df_fit.loc[df_fit.utc_hour == PLACEBO_HOUR, v].median()
        excl = (lo > 0 or hi < 0)
        crows.append({'var': v, 'med_open': m13, 'med_placebo': m3, 'contrast': c,
                      'ci_lo': lo, 'ci_hi': hi, 'n_open_days': n13, 'n_placebo_days': n3,
                      'ci_excludes_0': excl})
        print(f"  {v:<8}{m13:>10.4f}{m3:>10.4f}{c:>+10.4f}  [{lo:+.4f},{hi:+.4f}] {str(excl):>6}")
    pd.DataFrame(crows).to_csv(HAWKES / 'results' / 'phase0_open_contrast.csv', index=False)

    # --- confirm the four-kernel open peak ---
    print("\nFour-kernel open peak check (is UTC-13 the argmax of the by-hour median?):")
    for v in KERNELS:
        d = by_hour[by_hour['var'] == v]
        peak_h = int(d.loc[d['median'].idxmax(), 'utc_hour'])
        print(f"  {v}: argmax hour = {peak_h}  {'<- OPEN' if peak_h == OPEN_HOUR else ''}")
    print("\nPhase 0 done. Artifacts in results/. STOP — awaiting go-ahead for Phase 1.")


if __name__ == '__main__':
    main()
