"""Feature / target / label construction for one trading day.

v10 changes vs v9:
- Predictor forecasts FOUR horizons (5, 15, 60, 240 minutes).
- Targets are VOLATILITY-NORMALIZED: the raw % return is divided by the
  (causally computed) rolling 30-min per-minute vol scaled to the horizon.
  The model therefore predicts "how many vols" the move is - a stationary,
  learnable signal - instead of a near-zero raw percentage. This is the main
  cure for the flat-line predictor of v9 (together with the correlation loss
  in train.py).
- vol_scale is kept so forecasts can be converted back to % for plotting and
  evaluation.

All statistics are computed causally (rolling windows over the past only).
"""

import math

import numpy as np

from .data import PAIRS

W = 64                     # trade-maker observation window (minutes)
N_PAIRS = len(PAIRS)
HORIZONS = (5, 15, 60, 240)  # predictor forecast horizons (minutes)
IDX15, IDX60 = 1, 2        # positions of the 15m / 60m horizons in HORIZONS
N_FEATURES = N_PAIRS * 7 + 2
VOL_THRESH = (0.07, 0.22)  # 1-min realized-vol class boundaries (in %)
TREND_DEADBAND = 0.15      # cumulative 60-min move (in %) counted as "flat"
LABEL_HORIZON = 60         # analyzer label horizon (minutes)


def _shift_diff(x, n):
    out = np.zeros_like(x)
    if n < len(x):
        out[n:] = x[n:] - x[:-n]
    return out


def _roll_mean(x, n):
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(T)
    for i in range(min(n - 1, T)):  # expanding mean for the first minutes
        out[i] = c[i + 1] / (i + 1)
    if T >= n:
        out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def _roll_std(x, n):
    x = np.asarray(x, dtype=np.float64)
    m = _roll_mean(x, n)
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    out = np.empty_like(x)
    var = (c2[n:] - c2[:-n]) / n - ((c1[n:] - c1[:-n]) / n) ** 2
    out[n - 1:] = np.sqrt(np.maximum(var, 0.0))
    for i in range(min(n - 1, len(x))):  # expanding window for the first minutes
        out[i] = np.std(x[: i + 1])
    return out


def build_features(day):
    """(T, N_FEATURES) float32 matrix for one day. Per coin: 1m/5m/15m log
    returns (%), 30m realised vol, volume z-score, bar range, distance from
    30m mean - plus time-of-day sin/cos."""
    T = len(day[PAIRS[0]]["close"])
    F = np.zeros((T, N_FEATURES), dtype=np.float32)
    col = 0
    for p in PAIRS:
        d = day[p]
        close = np.asarray(d["close"], dtype=np.float64)
        high = np.asarray(d["high"], dtype=np.float64)
        low = np.asarray(d["low"], dtype=np.float64)
        volume = np.nan_to_num(np.asarray(d["volume"], dtype=np.float64))
        lc = np.log(np.maximum(close, 1e-9))
        r1 = np.diff(lc, prepend=lc[0]) * 100.0
        r5 = _shift_diff(lc, 5) * 100.0
        r15 = _shift_diff(lc, 15) * 100.0
        vol30 = _roll_std(r1, 30)
        vz = (volume - _roll_mean(volume, 30)) / (_roll_std(volume, 30) + 1e-8)
        rng = (high - low) / np.maximum(close, 1e-9) * 100.0
        dma = (close / _roll_mean(close, 30) - 1.0) * 100.0
        for j, arr in enumerate((r1, r5, r15, vol30, vz, rng, dma)):
            F[:, col + j] = np.clip(np.nan_to_num(arr, nan=0.0), -15.0, 15.0)
        col += 7
    mins = np.arange(T, dtype=np.float64)
    F[:, col] = np.sin(2 * np.pi * mins / 1440.0)
    F[:, col + 1] = np.cos(2 * np.pi * mins / 1440.0)
    return F


def build_targets(day, horizons=HORIZONS):
    """Predictor targets for every (coin, horizon).

    Returns (Y_pct, Y_norm, M, vol_scale):
      Y_pct     (T, P*H) raw cumulative log return (%) over the next h minutes
      Y_norm    (T, P*H) Y_pct / vol_scale  ("how many vols") - the training
                target; clipped to +/-10
      M         (T, P*H) validity mask (False where the horizon crosses the
                day boundary)
      vol_scale (T, P*H) causal per-minute vol (30m rolling std of 1m returns)
                scaled by sqrt(h), in %
    """
    T = len(day[PAIRS[0]]["close"])
    n_out = N_PAIRS * len(horizons)
    Y_pct = np.zeros((T, n_out), dtype=np.float32)
    Y_norm = np.zeros((T, n_out), dtype=np.float32)
    M = np.zeros((T, n_out), dtype=bool)
    vol_scale = np.ones((T, n_out), dtype=np.float32)
    for i, p in enumerate(PAIRS):
        close = np.asarray(day[p]["close"], dtype=np.float64)
        lc = np.log(np.maximum(close, 1e-9))
        r1 = np.diff(lc, prepend=lc[0]) * 100.0
        std30 = np.clip(_roll_std(r1, 30), 1e-3, 20.0)  # causal per-minute vol
        for k, h in enumerate(horizons):
            col = i * len(horizons) + k
            y = np.full(T, np.nan)
            if T > h:
                y[: T - h] = (lc[h:] - lc[: T - h]) * 100.0
            M[:, col] = ~np.isnan(y)
            y = np.nan_to_num(y, nan=0.0)
            scale = std30 * math.sqrt(h)
            Y_pct[:, col] = y
            vol_scale[:, col] = scale
            Y_norm[:, col] = np.clip(y / scale, -10.0, 10.0)
    return Y_pct, Y_norm, M, vol_scale


def build_labels(day, horizon=LABEL_HORIZON):
    """Analyzer training labels computed from the *future* (labels only, never
    inputs): next-hour realised-vol class and next-hour trend class per coin."""
    T = len(day[PAIRS[0]]["close"])
    vol_lab = np.zeros((T, N_PAIRS), dtype=np.int64)
    trend_lab = np.zeros((T, N_PAIRS), dtype=np.int64)
    mask = np.zeros(T, dtype=bool)
    if T > horizon:
        mask[: T - horizon] = True
    for i, p in enumerate(PAIRS):
        lc = np.log(np.maximum(np.asarray(day[p]["close"], dtype=np.float64), 1e-9))
        r1 = np.diff(lc, prepend=lc[0]) * 100.0
        std_h = _roll_std(r1, horizon)               # window ending at t
        fut = np.zeros(T)
        fut[: T - horizon] = std_h[horizon:]         # std of r1[t+1 .. t+h]
        vol_lab[:, i] = np.digitize(fut, VOL_THRESH)
        y = np.zeros(T)
        y[: T - horizon] = (lc[horizon:] - lc[: T - horizon]) * 100.0
        trend_lab[:, i] = np.where(
            y > TREND_DEADBAND, 1, np.where(y < -TREND_DEADBAND, 2, 0)
        )
    return vol_lab, trend_lab, mask
