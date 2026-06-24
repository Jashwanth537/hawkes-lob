"""Binance aggTrades loaders: Parquet-first (CSV fallback), with agg_id deduplication"""

import os
import pandas as pd

_COLS  = ['agg_id', 'price', 'quantity', 'first_id', 'last_id',
          'timestamp_us', 'is_buyer_maker', 'is_best_match']
_CALIB = ['timestamp_us', 'price', 'is_buyer_maker']   # subset based on our needs for an unmarked model


def load_one_file(path, dedupe=True):
    """Load one month -> DataFrame[timestamp_us(int64 us), event_type(0=buy,1=sell), price].
    Prefers the .parquet archive; falls back to .csv.

    dedupe=True drops duplicate agg_id rows. 2026-02 has 3,000 such rows from a Binance
    day-boundary """
    cols = (['agg_id'] + _CALIB) if dedupe else _CALIB
    pq_path = path if path.endswith('.parquet') else os.path.splitext(path)[0] + '.parquet'
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path, columns=cols)
    else:
        df = pd.read_csv(path, header=None, names=_COLS, usecols=cols,
                         dtype={'agg_id': 'int64', 'timestamp_us': 'int64',
                                'price': 'float64', 'is_buyer_maker': 'bool'})
    if dedupe:
        n0 = len(df)
        df = df.drop_duplicates(subset='agg_id', keep='first')
        if n0 - len(df):
            print(f"    deduped {n0-len(df):,} duplicate agg_id rows in {os.path.basename(path)}")
        df = df.drop(columns='agg_id')
    df['event_type'] = df['is_buyer_maker'].astype('int8')
    return df[['timestamp_us', 'event_type', 'price']]


def load_aggtrades(paths, dedupe=True):
    chunks = []
    for f in paths:
        d = load_one_file(f, dedupe=dedupe)
        print(f"  {os.path.basename(f)}: {len(d):,} rows")
        chunks.append(d)
    out = pd.concat(chunks, ignore_index=True)
    print(f"  -> total: {out.shape}  ({out.memory_usage(deep=True).sum()/1e9:.2f} GB)")
    return out
