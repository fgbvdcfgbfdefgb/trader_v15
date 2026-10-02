"""Offline dataset loading.

The repo ships 1-minute bars for BTCUSDT / ETHUSDT / LTCUSDT as per-coin
per-year CSV.gz files in data/. This module loads them, aligns the three
coins on their common minutes and slices them into UTC calendar days.

No network access is used anywhere.
"""

import glob
import os

import numpy as np
import pandas as pd

PAIRS = ("BTCUSDT", "ETHUSDT", "LTCUSDT")
FIELDS = ("open", "high", "low", "close", "volume")
MIN_MINUTES_PER_DAY = 1400  # a UTC day is kept only if >= this many common minutes


def _load_pair_frames(data_dir, pair):
    frames = []
    paths = sorted(glob.glob(os.path.join(data_dir, f"{pair}_*_1m.csv.gz")))
    if not paths:
        raise FileNotFoundError(f"no data files matching {pair}_*_1m.csv.gz in {data_dir}")
    for path in paths:
        df = pd.read_csv(path)
        ts = df["ts"].to_numpy(dtype=np.int64)
        if ts.size and int(ts.max()) > 10**14:  # microseconds -> milliseconds
            ts = ts // 1000
        df = df[["ts", *FIELDS]].copy()
        df["ts"] = ts
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    idx = pd.to_datetime(df["ts"].to_numpy(), unit="ms", utc=True)
    return pd.DataFrame(
        df[list(FIELDS)].to_numpy(dtype=np.float32), index=idx, columns=list(FIELDS)
    )


def load_days(data_dir, max_days=0, verbose=True):
    """Return { 'YYYY-MM-DD': { pair: {field: np.float32[1440] } } } for every
    valid UTC day, oldest first."""
    per = {p: _load_pair_frames(data_dir, p) for p in PAIRS}
    common = per[PAIRS[0]].index
    for p in PAIRS[1:]:
        common = common.intersection(per[p].index)
    common = common.sort_values()
    if len(common) == 0:
        raise RuntimeError("no common timestamps between the three coins")

    df = pd.concat([per[p].reindex(common) for p in PAIRS], axis=1, keys=PAIRS)
    del per, common

    days = {}
    for date, g in df.groupby(df.index.date):
        if len(g) < MIN_MINUTES_PER_DAY:
            continue
        full_idx = pd.date_range(f"{date} 00:00:00", periods=1440, freq="min", tz="UTC")
        gg = g.reindex(full_idx)
        prices = gg.ffill().bfill()
        day = {}
        for p in PAIRS:
            sub = prices[p]
            day[p] = {k: np.nan_to_num(sub[k].to_numpy(dtype=np.float32)) for k in FIELDS}
            # volume: 0 where the minute was missing (no ffill of fake volume)
            vol = gg[p]["volume"]
            day[p]["volume"] = np.nan_to_num(vol.fillna(0.0).to_numpy(dtype=np.float32))
        days[str(date)] = day
    del df

    dates = sorted(days)
    if max_days and len(dates) > max_days:
        keep = set(dates[-max_days:])
        days = {d: days[d] for d in dates if d in keep}
        dates = sorted(days)
    if verbose:
        print(f"[data] loaded {len(dates)} valid trading days "
              f"({dates[0]} .. {dates[-1]}) from {os.path.abspath(data_dir)}")
    return days
