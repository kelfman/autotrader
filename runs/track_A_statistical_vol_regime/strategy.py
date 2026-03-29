"""
strategy.py — Track A: volatility regime filtered momentum.
Signal class: statistical_vol_regime
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS: dict = {
    "vol_window": 96,
    "vol_rank_window": 2016,
    "vol_entry_thresh": 0.53,
    "vol_exit_thresh": 0.75,
    "momentum_window": 96,
    "zscore_entry": 0.45,
    "zscore_exit": -0.3,
    "trend_window": 48,
    "vov_window": 168,
    "vov_rank_window": 2016,
    "vov_thresh": 0.72,
    "atr_window": 48,
    "atr_stop_mult": 3.0,
    "peak_window": 168,
}


def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    """
    Vol-regime momentum — iter 2 structure with slightly widened thresholds.
    vol_entry: 0.52 (from 0.50), vov: 0.70 (from 0.65).
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    opn = df["open"]

    vol_w = int(params["vol_window"])
    rank_w = int(params["vol_rank_window"])
    mom_w = int(params["momentum_window"])
    trend_w = int(params["trend_window"])
    vov_w = int(params["vov_window"])
    vov_rank_w = int(params["vov_rank_window"])
    atr_w = int(params["atr_window"])
    peak_w = int(params["peak_window"])

    # Garman-Klass realized vol
    log_hl = np.log(high / low)
    log_co = np.log(close / opn)
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    realized_vol = np.sqrt(gk_var.rolling(vol_w).mean().clip(lower=0) * 24 * 365)
    vol_pct = realized_vol.rolling(rank_w, min_periods=vol_w).rank(pct=True)

    # Vol-of-vol stability
    vol_of_vol = realized_vol.rolling(vov_w).std()
    vov_pct = vol_of_vol.rolling(vov_rank_w, min_periods=vov_w).rank(pct=True)
    stable_vov = vov_pct < params["vov_thresh"]

    # Price z-score
    roll_mean = close.rolling(mom_w).mean()
    roll_std = close.rolling(mom_w).std()
    zscore = (close - roll_mean) / roll_std

    # Trend confirmation
    trend_up = close > close.shift(trend_w)

    # Entry: calm vol + stable vov + momentum + trend
    calm = vol_pct < params["vol_entry_thresh"]
    momentum_up = zscore > params["zscore_entry"]
    entries = (calm & stable_vov & momentum_up & trend_up).fillna(False)

    # Exit: vol spike OR momentum fade OR ATR trailing stop
    vol_spike = vol_pct > params["vol_exit_thresh"]
    momentum_fade = zscore < params["zscore_exit"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_w).mean()
    peak = close.rolling(peak_w, min_periods=1).max()
    stop_hit = close < (peak - params["atr_stop_mult"] * atr)

    exits = (vol_spike | momentum_fade | stop_hit).fillna(False)

    return entries, exits
