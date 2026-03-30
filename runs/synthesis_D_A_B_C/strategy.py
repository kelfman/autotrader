"""
strategy.py — Stage 2 synthesis: D+A base with optional B (session) and C (ratio) filters.

Base strategy is the Bayesian-optimized D+A from Stage 1 (fitness +2.030).
Boolean switches allow Optuna to test whether calendar session effects (Track B)
or BTC/ETH ratio signals (Track C) add marginal value to the already-strong base.

If both use_session_filter and use_ratio_filter are False, this strategy is
identical to the Stage 1 D+A optimum.

Requires: funding_rate column. eth_close column if use_ratio_filter is True.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS: dict = {
    # D+A base (Stage 1 optimized defaults)
    "sma_period": 256,
    "fr_pct_window": 408,
    "fr_entry_pct": 0.6879,
    "fr_exit_pct": 0.7784,
    "exit_lookback": 60,
    "vol_lookback": 53,
    "vol_pct_window": 1104,
    "vol_entry_pct": 0.4553,
    "vol_exit_pct": 0.5766,
    # B: session filter (off by default)
    "use_session_filter": False,
    "session_start_hour": 13,
    "session_end_hour": 21,
    # C: ratio filter (off by default)
    "use_ratio_filter": False,
    "ratio_lookback": 168,
    "ratio_z_entry": -1.0,
}


def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    funding = df["funding_rate"]

    # -- D: funding rate percentile + SMA momentum --
    sma = close.rolling(int(params["sma_period"])).mean()
    price_momentum = close > sma

    fr_window = int(params["fr_pct_window"])
    fr_pct = funding.rolling(fr_window).rank(pct=True)
    funding_ok = fr_pct < params["fr_entry_pct"]

    # -- A: GK vol percentile --
    vol_lb = int(params["vol_lookback"])
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    realized_vol = np.sqrt(gk_var.rolling(vol_lb).mean() * 252 * 24)

    vol_pct_win = int(params["vol_pct_window"])
    vol_pct = realized_vol.rolling(vol_pct_win).rank(pct=True)
    vol_calm = vol_pct < params["vol_entry_pct"]

    # -- Combined entry (D+A base) --
    entry_condition = price_momentum & funding_ok & vol_calm

    # -- B: session filter (optional) --
    if params.get("use_session_filter", False):
        hour = df.index.hour
        start_h = int(params["session_start_hour"])
        end_h = int(params["session_end_hour"])
        if start_h <= end_h:
            in_session = (hour >= start_h) & (hour < end_h)
        else:
            in_session = (hour >= start_h) | (hour < end_h)
        entry_condition = entry_condition & pd.Series(in_session, index=df.index)

    # -- C: ratio filter (optional) --
    if params.get("use_ratio_filter", False) and "eth_close" in df.columns:
        eth_close = df["eth_close"]
        ratio = close / eth_close
        ratio_mean = ratio.rolling(int(params["ratio_lookback"])).mean()
        ratio_std = ratio.rolling(int(params["ratio_lookback"])).std()
        ratio_z = (ratio - ratio_mean) / ratio_std.replace(0, np.nan)
        ratio_ok = ratio_z < params["ratio_z_entry"]
        entry_condition = entry_condition & ratio_ok.fillna(False)

    entries = entry_condition.fillna(False)

    # -- Combined exit (same as D+A) --
    exit_lb = int(params["exit_lookback"])
    trailing_low = close.shift(1).rolling(exit_lb).min()
    funding_crowded = fr_pct > params["fr_exit_pct"]
    vol_spike = vol_pct > params["vol_exit_pct"]

    exits = (funding_crowded | (close < trailing_low) | vol_spike).fillna(False)

    return entries, exits
