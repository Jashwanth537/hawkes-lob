#!/usr/bin/env python3
"""
Phase 1b: baseline-flexibility and activity-matching checks for phi_SB2.

Phase 1a found a persistent phi_SB2 increase around the NYSE open, but the open
hour is also more active and exhibits more timestamp ties, both of which can
inflate estimates. Phase 1b tests whether the result survives after controlling
for these effects.

The primary analysis uses tie-collapsed (aggregated) data. In parallel, each
open window is matched with off-open windows of similar activity to directly
test whether the effect is driven by event density rather than market timing.

We compare three baseline specifications:
    A. Constant baseline
    B. Spline baseline selected by BIC
    C. More flexible spline baseline

Fits use multiple random initializations, and paired bootstrap intervals are
used to assess whether the phi_SB2 open contrast remains stable or collapses
under increased baseline flexibility. We also check whether the ordering
open > macro-release hour > 10am hour is preserved.

All results are conditional on the 2-exponential kernel. Outputs are written to
results/phase1b_*.csv along with a final recommendation.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

_base = Path(__file__).resolve().parent.parent
HAWKES = _base if (_base / "src").is_dir() else _base / "hawkes"
sys.path.insert(0, str(HAWKES))
from src.data import load_one_file               # noqa: E402
from src.preprocessing import aggregate_sweeps   # noqa: E402
from src.model import fit_window_soe_fast        # noqa: E402  (spec A constant)
from src.spline import fit_spline, select_n_seg_bic, test_reduces_to_constant  # noqa: E402

RESULTS = HAWKES / "results"
MONTHS = ['2025-12', '2026-01', '2026-02']
GAP_US, DAY, NINITS = 100, 86400, 10
OPEN_H, MACRO_H, DATA_H = 14, 13, 15
NORMS = ['phi_SB2', 'phi_BS2', 'phi_BB1', 'phi_SS1', 'eta', 'mu_B', 'mu_S']
B = 2000


def load_all_aggregated():
    buys, sells = [], []
    for m in MONTHS:
        df = load_one_file(str(HAWKES / 'data' / 'download' / f'BTCUSDT-aggTrades-{m}.parquet'))
        buys.append(aggregate_sweeps(df.loc[df.event_type == 0, 'timestamp_us'].values, GAP_US) / 1e6)
        sells.append(aggregate_sweeps(df.loc[df.event_type == 1, 'timestamp_us'].values, GAP_US) / 1e6)
    buy = np.sort(np.concatenate(buys)); sell = np.sort(np.concatenate(sells))
    return buy, sell


def slice_win(buy, sell, start, dur=3600.0):
    T_B = buy[(buy >= start) & (buy < start + dur)] - start
    T_S = sell[(sell >= start) & (sell < start + dur)] - start
    return T_B, T_S


def count_win(buy, sell, start, dur=3600.0):
    import bisect
    nb = np.searchsorted(buy, start + dur) - np.searchsorted(buy, start)
    ns = np.searchsorted(sell, start + dur) - np.searchsorted(sell, start)
    return int(nb + ns)


def fit_spec(buy, sell, start, spec, S_B, S_C, seed):
    T_B, T_S = slice_win(buy, sell, start)
    if len(T_B) < 50 or len(T_S) < 50:
        return None
    if spec == 'A':
        np.random.seed(seed)
        r = fit_window_soe_fast(T_B, T_S, n_inits=NINITS)
        return None if r is None else {k: r[k] for k in NORMS}
    S = S_B if spec == 'B' else S_C
    r = fit_spline(T_B, T_S, S, n_inits=NINITS, seed=seed)
    return None if r is None else {k: r[k] for k in NORMS}


def boot_paired(open_vals, ctrl_vals, B=B, seed=1, one_sided=False):
    """Across-day bootstrap of median(open)-median(ctrl). Paired by resampling day index."""
    o = np.asarray(open_vals, float); c = np.asarray(ctrl_vals, float)
    m = ~(np.isnan(o) | np.isnan(c)); o, c = o[m], c[m]
    if len(o) < 3:
        return dict(contrast=np.nan, lo=np.nan, hi=np.nan, n=len(o))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(o), (B, len(o)))
    diff = np.median(o[idx], axis=1) - np.median(c[idx], axis=1)
    if one_sided:
        return dict(contrast=float(np.median(o) - np.median(c)),
                    lo=float(np.percentile(diff, 5)), hi=np.inf, n=len(o))
    return dict(contrast=float(np.median(o) - np.median(c)),
                lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)), n=len(o))


def main():
    np.random.seed(0); RESULTS.mkdir(exist_ok=True)
    print("verify spline reduces to constant-mu ...", test_reduces_to_constant(), "OK")
    print("loading 3 months aggregated (tie-collapsed) ...")
    buy, sell = load_all_aggregated()
    t0 = float(np.floor(min(buy[0], sell[0]) / DAY) * DAY)
    t_end = max(buy[-1], sell[-1])
    days = [int((t0 + d * DAY) // DAY) for d in range(int((t_end - t0) // DAY) + 1)]
    print(f"  {len(buy):,} buy / {len(sell):,} sell aggregated; {len(days)} days")

    # --- BIC: pick S_B on a sample of open windows; S_C = over-flexible ---
    sample_starts = [d * DAY + OPEN_H * 3600 for d in days[::12]]
    bic_votes = []
    for s in sample_starts:
        T_B, T_S = slice_win(buy, sell, s)
        if len(T_B) < 50: continue
        bs, _ = select_n_seg_bic(T_B, T_S, candidates=(2, 4, 6, 8, 12), n_inits=6, seed=int(s) % 2**31)
        if bs: bic_votes.append(bs)
    S_B = int(np.median(bic_votes)) if bic_votes else 6
    S_C = 12   # deliberately over-flexible: 12 segments over 60 min = 5-min baseline
    print(f"  BIC votes {bic_votes} -> S_B={S_B}; S_C(over-flexible)={S_C}")

    # --- activity-matched control: per open day, nearest-count off-open window ---
    off_pool = [(d, h) for d in days for h in range(24) if h not in (MACRO_H, OPEN_H, DATA_H)]
    pool_starts = np.array([d * DAY + h * 3600 for d, h in off_pool], float)
    pool_counts = np.array([count_win(buy, sell, s) for s in pool_starts])

    rows = []
    t_start = time.time()
    for d in days:
        o_start = d * DAY + OPEN_H * 3600
        if count_win(buy, sell, o_start) < 100:
            continue
        oc = count_win(buy, sell, o_start)
        ci = int(np.argmin(np.abs(pool_counts - oc)))
        c_start = pool_starts[ci]
        seed_o, seed_c = int(o_start) % 2**31, int(c_start) % 2**31
        rec = {'date': d, 'open_count': oc, 'ctrl_count': int(pool_counts[ci])}
        ok = True
        for spec in ('A', 'B', 'C'):
            ro = fit_spec(buy, sell, o_start, spec, S_B, S_C, seed_o)
            rc = fit_spec(buy, sell, c_start, spec, S_B, S_C, seed_c)
            if ro is None or rc is None:
                ok = False; break
            for k in NORMS:
                rec[f'{k}_open_{spec}'] = ro[k]
                rec[f'{k}_ctrl_{spec}'] = rc[k]
        if not ok:
            continue
        # tie-collapsed ORDERING (spec A only, cheap): phi_SB2 at h13 (macro) and h15 (10am)
        for h, lbl in [(MACRO_H, 'h13'), (DATA_H, 'h15')]:
            rh = fit_spec(buy, sell, d * DAY + h * 3600, 'A', S_B, S_C, int(d * DAY + h * 3600) % 2**31)
            rec[f'phi_SB2_{lbl}_A'] = rh['phi_SB2'] if rh else np.nan
        rec['phi_SB2_h14_A'] = rec['phi_SB2_open_A']
        rows.append(rec)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(RESULTS / 'phase1b_sweep_matched.csv', index=False)
    print(f"  sweep: {len(sweep)} matched day-pairs in {time.time()-t_start:.0f}s")

    # --- multistart noise floor per spec (one open window, 8 seeds) ---
    probe_start = days[len(days)//2] * DAY + OPEN_H * 3600
    print("\nMultistart noise floor (phi_SB2, one open window, 8 seeds):")
    for spec in ('A', 'B', 'C'):
        vals = []
        for sd in range(8):
            r = fit_spec(buy, sell, probe_start, spec, S_B, S_C, sd)
            if r: vals.append(r['phi_SB2'])
        vals = np.array(vals)
        print(f"  spec {spec}: median={np.median(vals):.5f} CV={vals.std()/vals.mean():.2f}")

    # --- sweep contrasts: open vs activity-matched control, per spec, paired CI ---
    print("\n=== phi_SB2 OPEN vs ACTIVITY-MATCHED control (tie-collapsed), paired across-day CI ===")
    print(f"  matched on count: open median {sweep.open_count.median():.0f} vs ctrl {sweep.ctrl_count.median():.0f}")
    print(f"  {'spec':<6}{'contrast':>10}{'95% CI':>24}{'excl 0':>8}")
    sb_ci = {}
    for spec in ('A', 'B', 'C'):
        c = boot_paired(sweep[f'phi_SB2_open_{spec}'], sweep[f'phi_SB2_ctrl_{spec}'])
        sb_ci[spec] = c
        print(f"  {spec:<6}{c['contrast']:>+10.4f}  [{c['lo']:+.4f},{c['hi']:+.4f}] {str(c['lo']>0):>6}")
    # other kernels for context (spec A vs C)
    print("\n  other kernels (contrast A -> C):")
    for k in ['phi_BB1', 'phi_SS1', 'phi_BS2', 'eta']:
        cA = boot_paired(sweep[f'{k}_open_A'], sweep[f'{k}_ctrl_A'])
        cC = boot_paired(sweep[f'{k}_open_C'], sweep[f'{k}_ctrl_C'])
        print(f"    {k:<8} A={cA['contrast']:+.4f}[{cA['lo']:+.4f},{cA['hi']:+.4f}]  "
              f"C={cC['contrast']:+.4f}[{cC['lo']:+.4f},{cC['hi']:+.4f}]")

    # --- tie-collapsed ORDERING (spec A): phi_SB2 at h13/h14/h15 vs matched control ---
    print("\n=== ORDERING (tie-collapsed, spec A): phi_SB2 open-hours vs matched control ===")
    for lbl in ['h13_A', 'h14_A', 'h15_A']:
        c = boot_paired(sweep[f'phi_SB2_{lbl}'], sweep['phi_SB2_ctrl_A'], one_sided=True)
        tag = {'h13_A': 'h13 macro', 'h14_A': 'h14 OPEN', 'h15_A': 'h15 10am'}[lbl]
        print(f"  {tag:11s} contrast={c['contrast']:+.4f}  5%lo={c['lo']:+.4f}  {'>0' if c['lo']>0 else 'covers0'}")
    o13, o14, o15 = (sweep['phi_SB2_h13_A'].median(), sweep['phi_SB2_h14_A'].median(),
                     sweep['phi_SB2_h15_A'].median())
    print(f"  medians: h14={o14:.4f} h13={o13:.4f} h15={o15:.4f}  "
          f"ordering h14>h13 & h14>h15: {o14 > o13 and o14 > o15}")

    # --- stability verdict: do A and C phi_SB2 CIs overlap? does C exclude 0? ---
    A, C = sb_ci['A'], sb_ci['C']
    overlap = not (A['hi'] < C['lo'] or C['hi'] < A['lo'])
    c_excl0 = C['lo'] > 0
    print("\n=== VERDICT ===")
    print(f"  phi_SB2 A contrast {A['contrast']:+.4f}[{A['lo']:+.4f},{A['hi']:+.4f}] ; "
          f"C {C['contrast']:+.4f}[{C['lo']:+.4f},{C['hi']:+.4f}]")
    print(f"  A-vs-C CI overlap: {overlap} ; C excludes 0: {c_excl0}")
    if c_excl0 and overlap:
        print("  -> phi_SB2 STABLE across mu-flexibility AND survives activity-matching -> REAL -> Phase 1c")
    elif not c_excl0:
        print("  -> phi_SB2 COLLAPSES under flexible mu (C covers 0) -> it was mu -> CLEAN NEGATIVE")
    else:
        print("  -> phi_SB2 attenuates/shifts -> inconclusive; inspect noise floor vs movement")
    print("\nPhase 1b done. STOP — awaiting go-ahead.")


if __name__ == '__main__':
    main()
