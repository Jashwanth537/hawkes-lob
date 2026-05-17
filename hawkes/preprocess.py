#!/usr/bin/env python3
"""
Filter full LOB event CSVs down to market-order rows only (event_type 0 or 1).
Streams line-by-line — never loads full file into memory.
"""
import sys
import os

DATA_DIR   = os.path.expanduser("~/lob_data")
OUT_DIR    = os.path.dirname(os.path.abspath(__file__)) + "/data"
SYMBOLS    = ["btcusdt", "ethusdt"]
HEADER     = "timestamp_us,event_type,price,quantity,symbol\n"

os.makedirs(OUT_DIR, exist_ok=True)

for sym in SYMBOLS:
    src = os.path.join(DATA_DIR, f"{sym}_events.csv")
    dst = os.path.join(OUT_DIR,  f"{sym}_mo.csv")

    n_buy = n_sell = n_skip = 0
    print(f"[{sym}] reading {src} ...", flush=True)

    with open(src, "r") as fin, open(dst, "w") as fout:
        fout.write(HEADER)
        for i, line in enumerate(fin):
            if i == 0:          # skip header
                continue
            # fast check before splitting
            # event_type is the second comma-separated field
            p1 = line.index(',')
            p2 = line.index(',', p1 + 1)
            et = line[p1+1:p2]
            if et == '0':
                fout.write(line)
                n_buy += 1
            elif et == '1':
                fout.write(line)
                n_sell += 1
            else:
                n_skip += 1

            if (i % 5_000_000) == 0 and i > 0:
                print(f"  {i//1_000_000}M rows processed...", flush=True)

    total = n_buy + n_sell
    ratio = n_buy / n_sell if n_sell else float('inf')
    print(f"[{sym}] done: {n_buy:,} BUY + {n_sell:,} SELL = {total:,} MO rows  "
          f"(BUY:SELL = {ratio:.2f})  -> {dst}", flush=True)
