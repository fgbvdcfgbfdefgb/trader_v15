#!/usr/bin/env python3
"""One-time dataset builder (NOT needed for training - the built dataset ships
inside the repo). Downloads Binance spot 1-minute klines for BTC/ETH/LTC USDT
from data.binance.vision and writes per-coin per-year CSV.gz files into data/.
"""

import argparse
import io
import os
import urllib.request
import zipfile

import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/klines"
PAIRS = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
COLS = ["ts", "open", "high", "low", "close", "volume", "close_time",
        "qav", "trades", "tbb", "tbq", "ignore"]


def fetch_month(pair, year, month):
    url = f"{BASE}/{pair}/1m/{pair}-1m-{year}-{month:02d}.zip"
    try:
        data = urllib.request.urlopen(url, timeout=180).read()
    except Exception as e:  # noqa: BLE001
        print(f"  skip {url}: {e}")
        return None
    zf = zipfile.ZipFile(io.BytesIO(data))
    df = pd.read_csv(zf.open(zf.namelist()[0]), header=None, names=COLS)
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    ts = df["ts"].to_numpy("int64")
    if ts.max() > 10**14:  # Binance switched to microsecond timestamps in 2025
        ts = ts // 1000
        df["ts"] = ts
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2023)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--end-month", type=int, default=9,
                    help="last month (inclusive) for the end year")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for pair in PAIRS:
        by_year = {}
        for year in range(args.start_year, args.end_year + 1):
            last_month = 12 if year < args.end_year else args.end_month
            for month in range(1, last_month + 1):
                df = fetch_month(pair, year, month)
                if df is not None and len(df):
                    by_year.setdefault(year, []).append(df)
                    print(f"  {pair} {year}-{month:02d}: {len(df)} rows")
        for year, dfs in sorted(by_year.items()):
            df = pd.concat(dfs, ignore_index=True)
            df = df.drop_duplicates("ts").sort_values("ts")
            out = os.path.join(args.outdir, f"{pair}_{year}_1m.csv.gz")
            df.to_csv(out, index=False, float_format="%.10g", compression="gzip")
            print(f"wrote {out} ({len(df)} rows, {os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
