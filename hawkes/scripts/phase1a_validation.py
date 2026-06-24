#!/usr/bin/env python3
"""Phase 1a: quick validation checks for the phi_SB2 NYSE-open effect
(BTC, Dec 2025 – Feb 2026, EST).

We have identified in Phase 0 that phi_SB2 is the only remaining candidate signal, showing a
modest increase around the NYSE open under the constant-baseline model. Before
introducing the more expensive time-varying baseline in Phase 1b, we run a set
of inexpensive robustness checks on the pooled 3-month dataset.

All fits use the constant-mu SOE Hawkes model with beta=[100, 1] on raw trade
streams, matching the Phase 0 setup. Confidence intervals are computed using
across-day percentile bootstrapping.

Checks:
  1. Window definition: verify whether the effect is concentrated at the open
     or spread across nearby time windows.
  2. Placebo test: compare the open against the full off-open hourly
     distribution rather than a single reference hour.
  3. Multiplicity: treat December as the discovery sample and January-February
     as out-of-sample confirmation.
  4. Ties: refit with sub-microsecond jitter to ensure the result is not driven
     by timestamp ties.

Phase 1b proceeds only if the phi_SB2 effect remains stable across these
checks. Results are written to results/phase1a_*.csv along with a pass/fail
recommendation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_base = Path(__file__).resolve().parent.parent
HAWKES = _base if (_base / "src").is_dir() else _base / "hawkes"
sys.path.insert(0, str(HAWKES))
from src.data import load_one_file              # noqa: E402
from src.model import fit_window_soe_fast       # noqa: E402

RESULTS = HAWKES / "results"
MONTHS = ['2025-12', '2026-01', '2026-02']
DEC_CACHE = RESULTS / 'phase0_btcusdt_const_2025-12.csv'   # reuse Phase 0 hourly Dec
B = 2000
OPEN_HOUR, MACRO_HOUR, DATA_HOUR = 14, 13, 15
NORMS = ['phi_SB2', 'phi_BS2', 'phi_BB1', 'phi_SS1', 'eta', 'mu_B', 'mu_S']
DAY = 86400


def load_month_streams(month):
    pq = HAWKES / 'data' / 'download' / f'BTCUSDT-aggTrades-{month}.parquet'
    df = load_one_file(str(pq))                  # dedup applied
    buy = np.sort(df.loc[df.event_type == 0, 'timestamp_us'].values) / 1e6
    sell = np.sort(df.loc[df.event_type == 1, 'timestamp_us'].values) / 1e6
    return buy, sell


def fit_window(buy, sell, start, dur):
    """constant-mu fit on [start, start+dur); returns dict of norms or None."""
    T_B = buy[(buy >= start) & (buy < start + dur)] - start
    T_S = sell[(sell >= start) & (sell < start + dur)] - start
    r = fit_window_soe_fast(T_B, T_S)
    if r is None:
        return None
    return {k: r[k] for k in NORMS} | {'n_B': len(T_B), 'n_S': len(T_S)}


def hourly_backbone():
    """Per-day, per-UTC-hour constant-mu fits across the 3 months. Reuse Dec cache."""
    frames = []
    dec = pd.read_csv(DEC_CACHE)
    dec['month'] = '2025-12'
    frames.append(dec)
    for month in MONTHS[1:]:
        buy, sell = load_month_streams(month)
        t0 = (np.floor(min(buy[0], sell[0]) / DAY) * DAY)
        t_end = max(buy[-1], sell[-1])
        rows = []
        start = t0
        while start + 3600 <= t_end:
            r = fit_window(buy, sell, start, 3600)
            if r is not None:
                rows.append({'window_start': start, 'utc_hour': int((start % DAY) // 3600), **r})
            start += 3600
        f = pd.DataFrame(rows); f['month'] = month
        f.to_csv(RESULTS / f'phase1a_hourly_{month}.csv', index=False)
        print(f"  {month}: {len(f)} hourly windows")
        frames.append(f)
    bb = pd.concat(frames, ignore_index=True)
    bb['date'] = (bb['window_start'] // DAY).astype(int)
    bb.to_csv(RESULTS / 'phase1a_btc_hourly_decfeb.csv', index=False)
    print(f"  backbone: {len(bb)} hourly windows, {bb['date'].nunique()} days")
    return bb


def aligned_fits():
    """Open-aligned sub-hour windows (14:30-15:30, 14:30-15:00) + a quiet-hour 30-min
    placebo panel, across all 3 months."""
    placebo_hours = [2, 5, 8, 20, 23]            # quiet, away from open/macro/data
    rows = []
    for month in MONTHS:
        buy, sell = load_month_streams(month)
        t0 = float(np.floor(min(buy[0], sell[0]) / DAY) * DAY)
        t_end = max(buy[-1], sell[-1])
        day = t0
        while day + DAY <= t_end + DAY:
            date = int(day // DAY)
            # open-aligned (14:30 = 14*3600 + 1800)
            o = day + 14 * 3600 + 1800
            for dur, tag in [(3600, 'open_1430_60'), (1800, 'open_1430_30')]:
                r = fit_window(buy, sell, o, dur)
                if r: rows.append({'date': date, 'slot': tag, 'phi_SB2': r['phi_SB2'],
                                   'phi_BS2': r['phi_BS2'], 'n_B': r['n_B']})
            # 30-min placebo panel (h:30)
            for h in placebo_hours:
                s = day + h * 3600 + 1800
                r = fit_window(buy, sell, s, 1800)
                if r: rows.append({'date': date, 'slot': f'plac30_h{h}', 'phi_SB2': r['phi_SB2'],
                                   'phi_BS2': r['phi_BS2'], 'n_B': r['n_B']})
            day += DAY
        print(f"  aligned/{month} done")
    a = pd.DataFrame(rows)
    a.to_csv(RESULTS / 'phase1a_btc_aligned.csv', index=False)
    return a


def boot_contrast(open_vals, rest_vals, B=B, seed=1, one_sided=True):
    """median(open) - median(rest), across-day percentile bootstrap CI."""
    o = np.asarray(open_vals, float); o = o[~np.isnan(o)]
    r = np.asarray(rest_vals, float); r = r[~np.isnan(r)]
    if len(o) < 3 or len(r) < 3:
        return dict(contrast=np.nan, lo=np.nan, hi=np.nan, n_open=len(o), n_rest=len(r))
    rng = np.random.default_rng(seed)
    bo = np.median(rng.choice(o, (B, len(o)), replace=True), axis=1)
    br = np.median(rng.choice(r, (B, len(r)), replace=True), axis=1)
    diff = bo - br
    lo, hi = (np.percentile(diff, 5), np.inf) if one_sided else tuple(np.percentile(diff, [2.5, 97.5]))
    return dict(contrast=float(np.median(o) - np.median(r)),
                lo=float(lo), hi=(float(hi) if np.isfinite(hi) else np.inf),
                n_open=len(o), n_rest=len(r))


def main():
    np.random.seed(0)
    RESULTS.mkdir(exist_ok=True)
    print("Phase 1a — hourly backbone (constant-mu, RAW, n_workers=1) ...")
    bb = hourly_backbone()
    print("Phase 1a — open-aligned + placebo sub-hour fits ...")
    al = aligned_fits()

    def hour_vals(df, h, col='phi_SB2'):
        return df.loc[df.utc_hour == h, col].values

    offopen = bb[~bb.utc_hour.isin([MACRO_HOUR, OPEN_HOUR, DATA_HOUR])]   # exclude open+neighbors

    print("\n================ TEST 1: WINDOW DEFINITION (phi_SB2 open contrast) ================")
    print("one-sided across-day bootstrap; rest = off-open hourly (excl 13,14,15)")
    rest = offopen.phi_SB2.values
    for h, lbl in [(MACRO_HOUR, 'hour13 macro'), (OPEN_HOUR, 'hour14 OPEN'), (DATA_HOUR, 'hour15 10am')]:
        c = boot_contrast(hour_vals(bb, h), rest)
        print(f"  {lbl:14s} contrast={c['contrast']:+.4f}  5%lo={c['lo']:+.4f}  "
              f"{'>0' if c['lo']>0 else 'covers0'}  (n_open={c['n_open']})")
    for tag, lbl in [('open_1430_60', 'aligned 14:30-15:30'), ('open_1430_30', 'aligned 14:30-15:00')]:
        ov = al.loc[al.slot == tag, 'phi_SB2'].values
        c = boot_contrast(ov, rest)
        print(f"  {lbl:20s} contrast={c['contrast']:+.4f}  5%lo={c['lo']:+.4f}  "
              f"{'>0' if c['lo']>0 else 'covers0'}  (n_open={c['n_open']})")
    plac30 = al.loc[al.slot.str.startswith('plac30'), 'phi_SB2'].values
    c30 = boot_contrast(al.loc[al.slot == 'open_1430_30', 'phi_SB2'].values, plac30)
    print(f"  [30-min open vs 30-min quiet placebo] contrast={c30['contrast']:+.4f} 5%lo={c30['lo']:+.4f}")

    print("\n================ TEST 2: PLACEBO ROBUSTNESS (open vs full off-open distribution) ====")
    med_by_hour = bb.groupby('utc_hour').phi_SB2.median()
    open_med = med_by_hour[OPEN_HOUR]
    off = med_by_hour.drop([MACRO_HOUR, OPEN_HOUR, DATA_HOUR])
    pct = (off < open_med).mean() * 100
    print(f"  open(h14) median phi_SB2={open_med:.4f}; off-open hour medians: "
          f"min={off.min():.4f} max={off.max():.4f} mean={off.mean():.4f}")
    print(f"  open exceeds {pct:.0f}% of off-open hour-medians; "
          f"{'UPPER TAIL' if pct>=90 else 'NOT clearly upper-tail'}")

    print("\n================ TEST 3: MULTIPLICITY (held-out Jan+Feb confirmation) ==============")
    janfeb = bb[bb.month != '2025-12']
    rest_jf = janfeb[~janfeb.utc_hour.isin([MACRO_HOUR, OPEN_HOUR, DATA_HOUR])].phi_SB2.values
    c_jf = boot_contrast(hour_vals(janfeb, OPEN_HOUR), rest_jf)
    c_all = boot_contrast(hour_vals(bb, OPEN_HOUR), rest)
    print(f"  pooled 3mo : contrast={c_all['contrast']:+.4f} 5%lo={c_all['lo']:+.4f} (n={c_all['n_open']})")
    print(f"  Jan+Feb only (held out, single pre-registered test): "
          f"contrast={c_jf['contrast']:+.4f} 5%lo={c_jf['lo']:+.4f} (n={c_jf['n_open']})")

    print("\n================ TEST 4: TIES (sub-us jitter, hour-14 slice) =======================")
    buy, sell = load_month_streams('2026-01')
    t0 = float(np.floor(buy[0] / DAY) * DAY)
    # one representative hour-14 window
    s = t0 + OPEN_HOUR * 3600
    base = fit_window(buy, sell, s, 3600)
    rng = np.random.default_rng(0)
    bj = np.sort(buy + rng.uniform(-5e-7, 5e-7, len(buy)))
    sj = np.sort(sell + rng.uniform(-5e-7, 5e-7, len(sell)))
    jit = fit_window(bj, sj, s, 3600)
    print(f"  hour14 phi_SB2: raw={base['phi_SB2']:.5f}  +jitter={jit['phi_SB2']:.5f}  "
          f"delta={abs(base['phi_SB2']-jit['phi_SB2']):.2e} (expect ~0)")

    # ---- GATE verdict ----
    pass1 = c_all['lo'] > 0
    pass2 = pct >= 90
    pass3 = c_jf['lo'] > 0
    print("\n================ GATE ================")
    print(f"  Test1 (open contrast >0, pooled best window): {'PASS' if pass1 else 'FAIL'}")
    print(f"  Test2 (open in upper tail of off-open):       {'PASS' if pass2 else 'FAIL'}")
    print(f"  Test3 (held-out Jan+Feb confirms):            {'PASS' if pass3 else 'FAIL'}")
    verdict = ('PROCEED to Phase 1b (build spline)' if (pass1 and pass2 and pass3)
               else 'CLEAN NEGATIVE — phi_SB2 not robust; do NOT build spline')
    print(f"  -> {verdict}")
    print("\nPhase 1a done. STOP — awaiting go-ahead.")


if __name__ == '__main__':
    main()
