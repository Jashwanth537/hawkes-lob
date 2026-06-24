"""Event-stream preprocessing: sweep aggregation, windowing, unit conversion."""

import numpy as np

def aggregate_sweeps(timestamps_us: np.ndarray, gap_us: int = 100) -> np.ndarray:
    """
    Collapse consecutive same-direction aggTrade events within `gap_us`
    microseconds into a single event, keeping the LATEST timestamp of each burst.
    Call separately for the buy and sell streams.

    gap_us=100 is the empirically optimal threshold for BTC/USDT (minimises KS
    statistic across thresholds 50–5000µs). A single market order sweeping
    multiple price levels on Binance completes in under 100µs.
    """
    ts = np.sort(timestamps_us.astype(np.int64))
    if len(ts) == 0:
        return ts
    is_last = np.empty(len(ts), dtype=bool)
    is_last[:-1] = np.diff(ts) > gap_us
    is_last[-1]  = True
    return ts[is_last]


def _first_hour_us(df, t0_us):
    """
    First hour data
    """
    t1_us = t0_us + 3_600_000_000
    b = df.loc[df['event_type'] == 0, 'timestamp_us'].values
    s = df.loc[df['event_type'] == 1, 'timestamp_us'].values
    return b[(b >= t0_us) & (b < t1_us)], s[(s >= t0_us) & (s < t1_us)]


def to_seconds(us_B, us_S, t0_us):
    """
    Convert microsecond timestamps to seconds relative to t0_us.
    """
    return (us_B - t0_us) / 1e6, (us_S - t0_us) / 1e6
