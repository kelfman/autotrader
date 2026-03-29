"""
strategy.py — Track B: calendar / trading session effects.
Signal class: calendar_session

Best fitness: -0.2376 (iter 8). Tue-Thu US session + 10-day trend filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS: dict = {
    "us_session_start": 13,
    "us_session_end": 20,
    "momentum_window": 96,
    "zscore_entry": 0.5,
    "zscore_exit": -0.4,
    "trend_window": 240,
    "atr_window": 48,
    "atr_stop_mult": 3.0,
    "peak_window": 168,
}


def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    """
    Tue-Thu US session trend-following momentum.

    Key finding: Tue-Thu filter removes Monday post-weekend noise and Friday
    position unwinding. Combined with 10-day trend filter, this produces
    consistent positive Sharpe in W2-W4 but struggles in choppy W1.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    hour = df.index.hour
    dow = df.index.dayofweek

    mom_w = int(params["momentum_window"])
    trend_w = int(params["trend_window"])
    atr_w = int(params["atr_window"])
    peak_w = int(params["peak_window"])

    us_session = (hour >= int(params["us_session_start"])) & (hour < int(params["us_session_end"]))
    mid_week = (dow >= 1) & (dow <= 3)

    roll_mean = close.rolling(mom_w).mean()
    roll_std = close.rolling(mom_w).std()
    zscore = (close - roll_mean) / roll_std
    momentum_up = zscore > params["zscore_entry"]

    trend_ma = close.rolling(trend_w).mean()
    trend_up = close > trend_ma

    entries = (us_session & mid_week & momentum_up & trend_up).fillna(False)

    momentum_fade = zscore < params["zscore_exit"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_w).mean()
    peak = close.rolling(peak_w, min_periods=1).max()
    stop_hit = close < (peak - params["atr_stop_mult"] * atr)

    exits = (momentum_fade | stop_hit).fillna(False)

    return entries, exits
