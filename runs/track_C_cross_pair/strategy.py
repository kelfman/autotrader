"""
strategy.py — Track C: cross-pair signals (BTC / ETH).
Signal class: cross_pair
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
    "eth_momentum_window": 72,
    "eth_zscore_thresh": 0.0,
    "eth_exit_zscore": -0.5,
    "atr_window": 48,
    "atr_stop_mult": 3.0,
    "peak_window": 168,
}


def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    """
    Vol-regime + vov + ETH confirmation (Track A best + ETH cross-pair signal).

    Entry: BTC vol calm + stable vov + BTC z-score > 0.45 + BTC trend up
      + ETH z-score > 0 (market-wide bullish).
    Exit: vol spike OR BTC momentum fades OR ETH collapses (z < -0.8)
      OR ATR trailing stop.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    opn = df["open"]
    eth_close = df["eth_close"]

    vol_w = int(params["vol_window"])
    rank_w = int(params["vol_rank_window"])
    mom_w = int(params["momentum_window"])
    trend_w = int(params["trend_window"])
    vov_w = int(params["vov_window"])
    vov_rank_w = int(params["vov_rank_window"])
    eth_mom_w = int(params["eth_momentum_window"])
    atr_w = int(params["atr_window"])
    peak_w = int(params["peak_window"])

    # BTC vol regime
    log_hl = np.log(high / low)
    log_co = np.log(close / opn)
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    realized_vol = np.sqrt(gk_var.rolling(vol_w).mean().clip(lower=0) * 24 * 365)
    vol_pct = realized_vol.rolling(rank_w, min_periods=vol_w).rank(pct=True)
    calm = vol_pct < params["vol_entry_thresh"]

    # Vol-of-vol stability
    vol_of_vol = realized_vol.rolling(vov_w).std()
    vov_pct = vol_of_vol.rolling(vov_rank_w, min_periods=vov_w).rank(pct=True)
    stable_vov = vov_pct < params["vov_thresh"]

    # BTC z-score
    btc_mean = close.rolling(mom_w).mean()
    btc_std = close.rolling(mom_w).std()
    btc_z = (close - btc_mean) / btc_std
    btc_momentum = btc_z > params["zscore_entry"]

    # BTC trend
    trend_up = close > close.shift(trend_w)

    # ETH momentum confirmation
    eth_mean = eth_close.rolling(eth_mom_w).mean()
    eth_std = eth_close.rolling(eth_mom_w).std()
    eth_z = (eth_close - eth_mean) / eth_std
    eth_confirm = eth_z > params["eth_zscore_thresh"]

    entries = (calm & stable_vov & btc_momentum & trend_up & eth_confirm).fillna(False)

    # Exit: vol spike OR BTC fades OR ETH collapses OR ATR stop
    vol_spike = vol_pct > params["vol_exit_thresh"]
    btc_fade = btc_z < params["zscore_exit"]
    eth_collapse = eth_z < params["eth_exit_zscore"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_w).mean()
    peak = close.rolling(peak_w, min_periods=1).max()
    stop_hit = close < (peak - params["atr_stop_mult"] * atr)

    exits = (vol_spike | btc_fade | eth_collapse | stop_hit).fillna(False)

    return entries, exits
