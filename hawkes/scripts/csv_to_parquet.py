#!/usr/bin/env python3
"""Convert raw Binance monthly aggTrades CSVs to Parquet (lossless). The aggTrades dumps have NO header.

Design:
  * Streams the CSV in blocks via pyarrow so memory stays bounded 
  * Writes Parquet with zstd compression (good ratio, fast).
  * Verifies each output by an INDEPENDENT raw-newline count of the source CSV
    against the Parquet row-group metadata, and reports basic statistics.
  * Idempotent: skips a month whose Parquet already exists and whose row count
    matches, unless --overwrite is given.

Usage
-----
    python scripts/csv_to_parquet.py                 # convert + verify all months
    python scripts/csv_to_parquet.py --overwrite     # force re-convert
    python scripts/csv_to_parquet.py --pattern 'BTCUSDT-aggTrades-2025-12.csv'
    python scripts/csv_to_parquet.py --input-dir /some/dir --output-dir /other/dir

Schema (preserved exactly)
--------------------------
    agg_id          int64
    price           float64
    quantity        float64
    first_id        int64
    last_id         int64
    timestamp_us    int64    (microseconds since Unix epoch)
    is_buyer_maker  bool     (True  -> taker SELL / event_type 1, False -> taker BUY  / event_type 0)
    is_best_match   bool
"""
from __future__ import annotations

import argparse
import csv as _csv
import datetime as dt
import glob
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

COLUMN_NAMES = ['agg_id', 'price', 'quantity', 'first_id', 'last_id',
                'timestamp_us', 'is_buyer_maker', 'is_best_match']
COLUMN_TYPES = {
    'agg_id':         pa.int64(),
    'price':          pa.float64(),
    'quantity':       pa.float64(),
    'first_id':       pa.int64(),
    'last_id':        pa.int64(),
    'timestamp_us':   pa.int64(),
    'is_buyer_maker': pa.bool_(),
    'is_best_match':  pa.bool_(),
}

def hawkes_dir() -> Path:
    """Locate the `hawkes` project dir whether this script lives in <repo>/scripts/
    or <repo>/hawkes/scripts/ (it's the ancestor that contains data/)."""
    base = Path(__file__).resolve().parent.parent
    return base if (base / 'data').is_dir() else base / 'hawkes'


def count_csv_rows(path: Path, bufsize: int = 1 << 24) -> int:
    """Independent row count = number of newlines (+1 if no trailing newline).

    The files are headerless, so lines == data rows. This deliberately does NOT
    reuse the conversion code, so it is a genuine cross-check of the row count.
    """
    n = 0
    last = b'\n'
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            n += chunk.count(b'\n')
            last = chunk[-1:]
    if last not in (b'\n', b''):
        n += 1
    return n


class StatAccumulator:
    """Merges per-batch min/max/sum so stats need only one streaming pass."""

    def __init__(self) -> None:
        self.rows = 0
        self.n_maker = 0          # is_buyer_maker == True  (taker SELL)
        self.n_best = 0
        self.price_min = self.price_max = None
        self.price_sum = 0.0
        self.qty_min = self.qty_max = None
        self.qty_sum = 0.0
        self.ts_min = self.ts_max = None
        self.aggid_min = self.aggid_max = None

    @staticmethod
    def _merge(lo, hi, col):
        mm = pc.min_max(col).as_py()
        b_lo, b_hi = mm['min'], mm['max']
        lo = b_lo if lo is None else min(lo, b_lo)
        hi = b_hi if hi is None else max(hi, b_hi)
        return lo, hi

    def update(self, batch: pa.RecordBatch) -> None:
        self.rows += batch.num_rows
        self.n_maker += pc.sum(batch.column('is_buyer_maker')).as_py() or 0
        self.n_best += pc.sum(batch.column('is_best_match')).as_py() or 0
        self.price_min, self.price_max = self._merge(
            self.price_min, self.price_max, batch.column('price'))
        self.qty_min, self.qty_max = self._merge(
            self.qty_min, self.qty_max, batch.column('quantity'))
        self.ts_min, self.ts_max = self._merge(
            self.ts_min, self.ts_max, batch.column('timestamp_us'))
        self.aggid_min, self.aggid_max = self._merge(
            self.aggid_min, self.aggid_max, batch.column('agg_id'))
        self.price_sum += pc.sum(batch.column('price')).as_py() or 0.0
        self.qty_sum += pc.sum(batch.column('quantity')).as_py() or 0.0

    @staticmethod
    def _utc(ts_us):
        if ts_us is None:
            return ''
        return dt.datetime.fromtimestamp(ts_us / 1e6, dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    def as_dict(self) -> dict:
        taker_sell = self.n_maker
        taker_buy = self.rows - self.n_maker
        return {
            'rows': self.rows,
            'taker_buy': taker_buy,
            'taker_sell': taker_sell,
            'price_min': self.price_min,
            'price_max': self.price_max,
            'price_mean': (self.price_sum / self.rows) if self.rows else None,
            'qty_min': self.qty_min,
            'qty_max': self.qty_max,
            'qty_sum': self.qty_sum,
            'ts_start_utc': self._utc(self.ts_min),
            'ts_end_utc': self._utc(self.ts_max),
            'agg_id_min': self.aggid_min,
            'agg_id_max': self.aggid_max,
        }


def convert_file(csv_path: Path, pq_path: Path, compression: str,
                 row_group_size: int, block_size: int) -> StatAccumulator:
    read_opts = pacsv.ReadOptions(column_names=COLUMN_NAMES, block_size=block_size)
    parse_opts = pacsv.ParseOptions(delimiter=',')
    convert_opts = pacsv.ConvertOptions(
        column_types=COLUMN_TYPES,
        true_values=['True', 'true'], false_values=['False', 'false'],
        strings_can_be_null=False)

    reader = pacsv.open_csv(csv_path, read_options=read_opts,
                            parse_options=parse_opts, convert_options=convert_opts)

    tmp_path = pq_path.with_suffix(pq_path.suffix + '.tmp')
    stats = StatAccumulator()
    writer = None
    pending, pending_rows = [], 0
    try:
        for batch in reader:
            stats.update(batch)
            pending.append(batch)
            pending_rows += batch.num_rows
            if pending_rows >= row_group_size:
                tbl = pa.Table.from_batches(pending)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, tbl.schema, compression=compression)
                writer.write_table(tbl, row_group_size=row_group_size)
                pending, pending_rows = [], 0
        if pending:
            tbl = pa.Table.from_batches(pending)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, tbl.schema, compression=compression)
            writer.write_table(tbl, row_group_size=row_group_size)
    finally:
        if writer is not None:
            writer.close()
    # atomic-ish swap so an interrupted run never leaves a half-written .parquet
    os.replace(tmp_path, pq_path)
    return stats


def stats_from_parquet(pq_path: Path) -> StatAccumulator:
    """Recompute stats by scanning an existing Parquet (used when skipping a CSV
    that is already converted, so the stats sidecar stays complete)."""
    acc = StatAccumulator()
    for batch in pq.ParquetFile(pq_path).iter_batches():
        acc.update(batch)
    return acc


def verify(csv_path: Path, pq_path: Path, stats: StatAccumulator) -> tuple[bool, int, int]:
    csv_rows = count_csv_rows(csv_path)
    pq_rows = pq.ParquetFile(pq_path).metadata.num_rows
    ok = (csv_rows == pq_rows == stats.rows)
    return ok, csv_rows, pq_rows


def fmt_bytes(n: int) -> str:
    return f'{n / 1e9:.2f} GB' if n >= 1e9 else f'{n / 1e6:.1f} MB'


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    default_in = hawkes_dir() / 'data' / 'download'
    p.add_argument('--input-dir', type=Path, default=default_in)
    p.add_argument('--output-dir', type=Path, default=None,
                   help='default: same as --input-dir')
    p.add_argument('--pattern', default='BTCUSDT-aggTrades-*.csv')
    p.add_argument('--compression', default='zstd',
                   choices=['zstd', 'snappy', 'gzip', 'brotli', 'lz4', 'none'])
    p.add_argument('--row-group-size', type=int, default=1_000_000)
    p.add_argument('--block-size-mb', type=int, default=128,
                   help='CSV read block size (controls peak memory)')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--no-verify', action='store_true')
    p.add_argument('--stats-out', type=Path, default=None,
                   help='default: <output-dir>/parquet_stats.csv')
    args = p.parse_args(argv)

    in_dir = args.input_dir
    out_dir = args.output_dir or in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_out = args.stats_out or (out_dir / 'parquet_stats.csv')
    compression = None if args.compression == 'none' else args.compression

    csv_files = sorted(Path(p_) for p_ in glob.glob(str(in_dir / args.pattern)))
    if not csv_files:
        print(f'No files match {in_dir / args.pattern}', file=sys.stderr)
        return 1

    # disk guard: parquet of these CSVs is typically ~0.3x their size
    total_csv = sum(f.stat().st_size for f in csv_files)
    free = os.statvfs(out_dir).f_bavail * os.statvfs(out_dir).f_frsize
    print(f'{len(csv_files)} file(s), {fmt_bytes(total_csv)} CSV -> '
          f'~{fmt_bytes(int(total_csv * 0.3))} Parquet est. | {fmt_bytes(free)} free')
    if free < total_csv * 0.4:
        print('WARNING: low disk space for the estimated Parquet output.', file=sys.stderr)

    rows = []
    all_ok = True
    for csv_path in csv_files:
        pq_path = out_dir / (csv_path.stem + '.parquet')
        print(f'\n=== {csv_path.name} ({fmt_bytes(csv_path.stat().st_size)}) ===')

        if pq_path.exists() and not args.overwrite:
            existing = pq.ParquetFile(pq_path).metadata.num_rows
            src_rows = count_csv_rows(csv_path)
            if existing == src_rows:
                print(f'  skip (exists, {existing:,} rows match). Use --overwrite to redo.')
                d = stats_from_parquet(pq_path).as_dict()
                row = {'file': pq_path.name, 'csv_rows': src_rows,
                       'parquet_rows': existing, 'verified': True,
                       'csv_bytes': csv_path.stat().st_size,
                       'parquet_bytes': pq_path.stat().st_size,
                       'note': 'skipped-existing'}
                row.update(d)
                rows.append(row)
                continue
            print(f'  exists but row mismatch ({existing:,} vs {src_rows:,}); reconverting.')

        t0 = time.time()
        stats = convert_file(csv_path, pq_path, compression,
                             args.row_group_size, args.block_size_mb << 20)
        dt_s = time.time() - t0
        d = stats.as_dict()
        pq_bytes = pq_path.stat().st_size
        print(f'  wrote {pq_path.name}  {d["rows"]:,} rows  '
              f'{fmt_bytes(pq_bytes)}  ({fmt_bytes(csv_path.stat().st_size)} CSV, '
              f'{pq_bytes / csv_path.stat().st_size:.2f}x)  in {dt_s:.0f}s')
        print(f'  taker buy/sell : {d["taker_buy"]:,} / {d["taker_sell"]:,}')
        print(f'  price          : min {d["price_min"]:.2f}  max {d["price_max"]:.2f}  '
              f'mean {d["price_mean"]:.2f}')
        print(f'  quantity       : min {d["qty_min"]:g}  max {d["qty_max"]:g}  '
              f'sum {d["qty_sum"]:.4f}')
        print(f'  time span (UTC): {d["ts_start_utc"]}  ->  {d["ts_end_utc"]}')
        print(f'  agg_id         : {d["agg_id_min"]:,}  ->  {d["agg_id_max"]:,}  '
              f'(contiguous span={d["agg_id_max"] - d["agg_id_min"] + 1 == d["rows"]})')

        verified = True
        if not args.no_verify:
            ok, csv_rows, pq_rows = verify(csv_path, pq_path, stats)
            verified = ok
            status = 'OK' if ok else 'MISMATCH'
            print(f'  verify rows    : csv={csv_rows:,}  parquet={pq_rows:,}  '
                  f'stream={stats.rows:,}  -> {status}')
            all_ok = all_ok and ok

        row = {'file': pq_path.name, 'csv_rows': stats.rows, 'parquet_rows': stats.rows,
               'verified': verified, 'csv_bytes': csv_path.stat().st_size,
               'parquet_bytes': pq_bytes, 'note': ''}
        row.update(d)
        rows.append(row)

    # write stats sidecar
    if rows:
        import csv as csvmod
        # Merge into any existing sidecar (keyed by file) so running one month at a
        # time -- and deleting the CSV after -- keeps a CUMULATIVE record instead of
        # overwriting it with only the current run's files.
        merged = {}
        if stats_out.exists():
            with open(stats_out, newline='') as fh:
                for r in csvmod.DictReader(fh):
                    merged[r['file']] = r
        for r in rows:
            merged[r['file']] = r
        out_rows = [merged[k] for k in sorted(merged)]
        head = ['file', 'csv_rows', 'parquet_rows', 'verified', 'taker_buy',
                'taker_sell', 'price_min', 'price_max', 'price_mean',
                'qty_min', 'qty_max', 'qty_sum', 'ts_start_utc', 'ts_end_utc',
                'agg_id_min', 'agg_id_max', 'csv_bytes', 'parquet_bytes', 'note']
        cols = head + [c for c in sorted({k for r in out_rows for k in r}) if c not in head]
        with open(stats_out, 'w', newline='') as fh:
            w = csvmod.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            for r in out_rows:
                w.writerow(r)
        print(f'\nStats written: {stats_out}  ({len(out_rows)} files tracked)')

    print('\nAll files verified.' if all_ok else '\nWARNING: some files failed verification.')
    return 0 if all_ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
