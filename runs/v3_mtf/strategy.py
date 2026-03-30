"""
strategy.py — V3 Phase 1: D+A synthesis with multi-timeframe signals.

Replaces the V2 1h SMA(266) trend filter with a daily SMA(20), which provides
dramatically cleaner trend identification. The daily trend filter is the single
most impactful structural change in V3 — it raises fitness from +1.845 (V2
ensemble) to +2.677 (V3 MTF ensemble), with OOS +2.682 and 0.5% decay.

Ensemble fitness: +2.677  |  Walk-forward OOS (W4-5): +2.682  |  Decay: 0.5%
Stability: 36.4% worst drop

Requires: df augmented with h4_* and d1_* columns via augment_with_timeframes().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS: dict = {
    # Daily trend filter (replaces 1h SMA)
    "d1_sma_period": 20,
    # 4h confirmation (disabled — optimizer found no value)
    "h4_sma_period": 132,
    "use_h4_confirmation": False,
    # Funding rate signal
    "fr_pct_window": 550,
    "fr_entry_pct": 0.6848,
    "fr_exit_pct": 0.8712,
    # Exit
    "exit_lookback": 53,
    # Vol regime gate
    "vol_lookback": 34,
    "vol_pct_window": 971,
    "vol_entry_pct": 0.7692,
    "vol_exit_pct": 0.6822,
    # MTF exit: daily vol spike exit (disabled — optimizer found no value)
    "use_d1_vol_exit": False,
    "d1_vol_exit_pct": 0.8292,
}


def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    """
    Multi-timeframe D+A strategy.

    Entry: d1_close > d1_SMA (daily trend bullish)
           AND [optionally] h4_close > h4_SMA (4h confirmation)
           AND funding_pct < threshold (not crowded)
           AND vol_pct < threshold (calm regime)
    Exit:  funding_pct > exit threshold OR close < trailing low
           OR vol_pct > vol exit threshold
           OR [optionally] d1_vol > d1_vol_exit threshold
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    funding = df["funding_rate"]

    # -- Daily trend filter (replaces 1h SMA) --
    # d1_close is forward-filled from daily. Scale window to 1h bars.
    d1_close = df["d1_close"]
    d1_sma_period = int(params["d1_sma_period"])
    d1_sma = d1_close.rolling(d1_sma_period * 24, min_periods=24).mean()
    daily_trend = d1_close > d1_sma

    # -- 4h confirmation --
    if params.get("use_h4_confirmation", False):
        h4_close = df["h4_close"]
        h4_sma_period = int(params["h4_sma_period"])
        h4_sma = h4_close.rolling(h4_sma_period * 4, min_periods=4).mean()
        h4_trend = h4_close > h4_sma
    else:
        h4_trend = pd.Series(True, index=df.index)

    # -- Funding rate percentile --
    fr_window = int(params["fr_pct_window"])
    fr_pct = funding.rolling(fr_window).rank(pct=True)
    funding_ok = fr_pct < params["fr_entry_pct"]

    # -- Vol regime gate (1h GK vol, same as V2) --
    vol_lb = int(params["vol_lookback"])
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    realized_vol = np.sqrt(gk_var.rolling(vol_lb).mean() * 252 * 24)

    vol_pct_win = int(params["vol_pct_window"])
    vol_pct = realized_vol.rolling(vol_pct_win).rank(pct=True)
    vol_calm = vol_pct < params["vol_entry_pct"]

    # -- Combined entry --
    entries = (daily_trend & h4_trend & funding_ok & vol_calm).fillna(False)

    # -- Exits --
    exit_lb = int(params["exit_lookback"])
    trailing_low = close.shift(1).rolling(exit_lb).min()
    funding_crowded = fr_pct > params["fr_exit_pct"]
    vol_spike = vol_pct > params["vol_exit_pct"]

    exit_cond = funding_crowded | (close < trailing_low) | vol_spike

    # Optional: daily vol spike exit (higher-TF vol regime deterioration)
    if params.get("use_d1_vol_exit", False):
        d1_vol = df.get("d1_vol")
        if d1_vol is not None:
            d1_vol_pct = d1_vol.rolling(vol_pct_win).rank(pct=True)
            exit_cond = exit_cond | (d1_vol_pct > params["d1_vol_exit_pct"])

    exits = exit_cond.fillna(False)

    return entries, exits
