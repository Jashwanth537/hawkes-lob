"""Threaded rolling Hawkes calibration (per 1-hour window) + streaming driver."""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
try:
    from tqdm import tqdm
except ImportError:                       # tqdm optional
    def tqdm(it, **kw): return it
from .model import fit_window_soe_fast
from .data import load_one_file


def _rv_in(t0, t1, price_t, price_px):
    p = price_px[(price_t >= t0) & (price_t < t1)]
    return float(np.sqrt(np.sum(np.diff(np.log(p))**2))) if len(p) >= 2 else np.nan


def rolling_calibration(buy_arr, sell_arr, symbol,
                        price_t=None, price_px=None,
                        beta1=100.0, beta2=1.0, window_sec=3600, n_workers=6):
    """Threaded rolling calibration. Each 1-hour window is independent and the numba
    log-likelihood runs nogil, so threads parallelise with NO array copies (RAM-flat).
    RV per window is folded in when price arrays are supplied.

    APPROXIMATION NOTE: per-window parameter estimates use the SAME data + optimizer as
    the serial version. The only non-determinism is the ORDER of the random multi-start
    noise across threads, which in rare ill-conditioned windows can land on a different
    local optimum. Statistically equivalent, but NOT bit-reproducible vs serial.
    Pass n_workers=1 for exact serial reproducibility."""
    t_start = min(buy_arr[0], sell_arr[0])
    t_end   = max(buy_arr[-1], sell_arr[-1])
    windows = np.arange(t_start, t_end - window_sec, window_sec)

    def task(t0):
        t1 = t0 + window_sec
        T_B = buy_arr [(buy_arr  >= t0) & (buy_arr  < t1)] - t0
        T_S = sell_arr[(sell_arr >= t0) & (sell_arr < t1)] - t0
        r = fit_window_soe_fast(T_B, T_S, beta1=beta1, beta2=beta2)
        if r is None:
            return None
        rec = {'window_start': float(t0), 'utc_hour': int((t0 % 86400)//3600),
               'symbol': symbol, 'n_B': len(T_B), 'n_S': len(T_S), **r}
        if price_t is not None:
            rec['rv'] = _rv_in(t0, t1, price_t, price_px)
        return rec

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        out = [r for r in tqdm(ex.map(task, windows), total=len(windows), desc=symbol)
               if r is not None]
    return pd.DataFrame(out)


def streaming_rolling_calibration(paths, symbol='BTCUSDT', n_workers=6, **kw):
    parts = []
    for f in paths:
        d  = load_one_file(f)
        t  = d['timestamp_us'].values / 1e6
        et = d['event_type'].values
        buy, sell = t[et == 0], t[et == 1]
        print(f"{os.path.basename(f)}: n={len(d):,}")
        parts.append(rolling_calibration(buy, sell, symbol,
                                         price_t=t, price_px=d['price'].values,
                                         n_workers=n_workers, **kw))
        del d, t, et, buy, sell
    return pd.concat(parts, ignore_index=True)
