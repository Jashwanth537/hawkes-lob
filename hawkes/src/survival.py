"""phi_SB2 UTC-13 survival test: shared fitting + contrast + reporting.

The NYSE-open (UTC-13) rise in phi_SB2 (slow SELL->BUY cross-excitation) under a
constant-mu fit could be an artifact of an intra-hour ramping baseline leaking into the
slowest kernel term (Filimonov-Sornette). We test this by fitting every window under
BOTH a constant-mu and a piecewise-mu(K) baseline and checking whether the UTC-13 lift
in phi_SB2 survives.

Judge survival by EFFECT SIZE + the change in Delta%/MW_p across the two baselines, not
by LR p-values (which reject ~100% from sheer n). Exploratory (single asset at a time).
"""

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import mannwhitneyu

try:
    from tqdm import tqdm
except ImportError:                       # tqdm optional
    def tqdm(it, **kw):
        return it

from .model import fit_window_soe_fast
from .timevarying import fit_pw_full

# variables compared at the open, in display order
SURVIVAL_VARS = ['phi_SB2', 'eta', 'phi_BB1', 'phi_SS1', 'mu_B', 'mu_S']
OPEN_HOUR_UTC = 13                        # NYSE open (13:30 UTC under EDT)
PLACEBO_HOUR_UTC = 3                       # control hour with no expected effect


def fit_survival_windows(buy, sell, K=4, beta1=100.0, beta2=1.0,
                         window_sec=3600.0, n_workers=6, desc='survival'):
    """Fit every `window_sec` window under constant-mu AND piecewise-mu(K).

    buy/sell are event times in SECONDS (already sweep-aggregated upstream). Returns a
    DataFrame with one row per window: window_start, utc_hour, and `<var>_const` /
    `<var>_pw` for each var in SURVIVAL_VARS. Threaded; seed np.random beforehand for
    a (still thread-order-dependent) reproducible-ish run, or n_workers=1 for exact.
    """
    buy = np.asarray(buy, dtype=np.float64)
    sell = np.asarray(sell, dtype=np.float64)
    t_start, t_end = min(buy[0], sell[0]), max(buy[-1], sell[-1])
    windows = np.arange(t_start, t_end - window_sec, window_sec)

    def fit_one(w):
        T_B = buy[(buy >= w) & (buy < w + window_sec)] - w
        T_S = sell[(sell >= w) & (sell < w + window_sec)] - w
        const = fit_window_soe_fast(T_B, T_S, beta1=beta1, beta2=beta2)
        pw = fit_pw_full(T_B, T_S, K=K, beta1=beta1, beta2=beta2, window_sec=window_sec)
        if const is None or pw is None:
            return None
        row = {'window_start': float(w), 'utc_hour': int((w % 86400) // 3600)}
        for v in SURVIVAL_VARS:
            row[f'{v}_const'] = const[v]
            row[f'{v}_pw'] = pw[v]
        return row

    fit_one(windows[0])                   # warm the numba JIT before threading
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        rows = [r for r in tqdm(ex.map(fit_one, windows), total=len(windows), desc=desc)
                if r is not None]
    return pd.DataFrame(rows)


def contrast(frame, col, hour=OPEN_HOUR_UTC):
    """One-sided (hour > rest) Mann-Whitney on `col`. Returns (median_hour,
    median_rest, delta_pct, p); NaNs if either group has < 3 observations."""
    hour_vals = frame.loc[frame.utc_hour == hour, col].dropna()
    rest_vals = frame.loc[frame.utc_hour != hour, col].dropna()
    if len(hour_vals) < 3 or len(rest_vals) < 3:
        return np.nan, np.nan, np.nan, np.nan
    delta = 100.0 * (hour_vals.median() - rest_vals.median()) / rest_vals.median()
    _, p = mannwhitneyu(hour_vals, rest_vals, alternative='greater')
    return float(hour_vals.median()), float(rest_vals.median()), float(delta), float(p)


def survival_verdict(delta_const, p_const, delta_pw, p_pw):
    """SURVIVES if still significant and >= half the constant-mu effect; VANISHES if it
    loses significance and most of its size; ATTENUATES in between."""
    if p_pw < 0.05 and delta_pw >= 0.5 * delta_const:
        return 'SURVIVES'
    if p_pw >= 0.05 and delta_pw < 0.5 * delta_const:
        return 'VANISHES'
    return 'ATTENUATES'


def survival_report(frame, symbol='', open_hour=OPEN_HOUR_UTC, placebo_hour=PLACEBO_HOUR_UTC):
    """Print the const-mu vs piecewise-mu UTC-open contrast table, the phi_SB2 verdict,
    and a placebo-hour check. Returns the {(var, model): (delta, p)} dict."""
    n_open = int((frame.utc_hour == open_hour).sum())
    n_rest = int((frame.utc_hour != open_hour).sum())
    print(f"\n=== {symbol}  UTC-{open_hour} (n={n_open}) vs rest (n={n_rest})"
          f" : const-mu vs piecewise-mu ===")
    print(f"{'var':<8}{'model':<7}{'med_open':>10}{'med_rest':>10}{'D%':>8}{'MW_p':>11}")
    print('-' * 54)
    results = {}
    for base in SURVIVAL_VARS:
        for tag in ('const', 'pw'):
            m_open, m_rest, delta, p = contrast(frame, f'{base}_{tag}', open_hour)
            results[(base, tag)] = (delta, p)
            flag = 'OK' if (p == p and p < 0.05) else ''
            print(f"{base:<8}{tag:<7}{m_open:>10.4f}{m_rest:>10.4f}{delta:>+7.1f}%{p:>11.3g}  {flag}")

    delta_c, p_c = results[('phi_SB2', 'const')]
    delta_p, p_p = results[('phi_SB2', 'pw')]
    verdict = survival_verdict(delta_c, p_c, delta_p, p_p)
    print(f"\nVERDICT {symbol} phi_SB2 under piecewise-mu: {verdict}")
    print(f"  const Delta={delta_c:+.1f}% (p={p_c:.3g})  ->  "
          f"piecewise Delta={delta_p:+.1f}% (p={p_p:.3g})")
    if verdict == 'SURVIVES':
        print("  => genuine reactive slow SELL->BUY cross-excitation at the open "
              "(NOT a constant-mu artifact).")
    elif verdict == 'VANISHES':
        print("  => open effect was purely exogenous (mu); clean null on endogeneity.")

    print(f"PLACEBO UTC-{placebo_hour} (n={int((frame.utc_hour == placebo_hour).sum())}) "
          f"-- expect no lift:")
    for tag in ('const', 'pw'):
        _, _, delta, p = contrast(frame, f'phi_SB2_{tag}', placebo_hour)
        note = 'SPURIOUS LIFT' if (p == p and p < 0.05 and delta > 0) else 'no lift (good)'
        print(f"  phi_SB2 {tag:<6} D={delta:+.1f}% p={p:.3g}  {note}")
    return results
