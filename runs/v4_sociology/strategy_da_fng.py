"""
strategy_da_fng.py — D+A+FNG synthesis: V2 base + behavioral overlay.

Combines the proven D+A strategy (funding rate + vol regime + SMA momentum)
with Fear & Greed Index as a behavioral filter:
  - FNG washout requirement: recent fear clears overcrowded positioning
  - FNG greed exit: force exit when retail has arrived

The D+A base provides ~100 trades/year; the FNG filter reduces this to
trades that occur after genuine positioning washouts, improving per-trade
quality while maintaining reasonable frequency.

Requires: 'fng_value' and 'funding_rate' columns in DataFrame.
FNG must be lagged by 1 day (done by data loader) to prevent look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS: dict = {
    # D+A base (ensemble params from V2)
    "sma_period": 266,
    "fr_pct_window": 418,
    "fr_entry_pct": 0.6898,
    "fr_exit_pct": 0.7604,
    "exit_lookback": 55,
    "vol_lookback": 52,
    "vol_pct_window": 1070,
    "vol_entry_pct": 0.5088,
    "vol_exit_pct": 0.5645,

    # FNG behavioral overlay
    "fng_washout_level": 35,
    "fng_washout_lookback": 504,    # hours to look back for washout (21 days)
    "fng_exit_level": 75,           # greed exit
    "use_fng_entry": True,          # FNG washout gate on entries
    "use_fng_exit": True,           # FNG greed exit
}


def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    funding = df["funding_rate"]
    fng = df["fng_value"].astype(float)

    # -- D+A base: funding rate percentile + SMA momentum + vol regime --
    sma = close.rolling(int(params["sma_period"])).mean()
    price_momentum = close > sma

    fr_window = int(params["fr_pct_window"])
    fr_pct = funding.rolling(fr_window).rank(pct=True)
    funding_ok = fr_pct < params["fr_entry_pct"]

    vol_lb = int(params["vol_lookback"])
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    realized_vol = np.sqrt(gk_var.rolling(vol_lb).mean() * 252 * 24)
    vol_pct_win = int(params["vol_pct_window"])
    vol_pct = realized_vol.rolling(vol_pct_win).rank(pct=True)
    vol_calm = vol_pct < params["vol_entry_pct"]

    # -- D+A entries (before FNG overlay) --
    base_entries = (price_momentum & funding_ok & vol_calm).fillna(False)

    # -- FNG behavioral overlay: washout gate --
    if params.get("use_fng_entry", True):
        washout_lb = int(params["fng_washout_lookback"])
        fng_min_recent = fng.rolling(washout_lb, min_periods=1).min()
        had_washout = fng_min_recent < params["fng_washout_level"]
        entries = (base_entries & had_washout).fillna(False)
    else:
        entries = base_entries

    # -- D+A exits (before FNG overlay) --
    exit_lb = int(params["exit_lookback"])
    trailing_low = close.shift(1).rolling(exit_lb).min()
    funding_crowded = fr_pct > params["fr_exit_pct"]
    vol_spike = vol_pct > params["vol_exit_pct"]
    base_exits = (funding_crowded | (close < trailing_low) | vol_spike).fillna(False)

    # -- FNG greed exit --
    if params.get("use_fng_exit", True):
        fng_greed = fng > params["fng_exit_level"]
        exits = (base_exits | fng_greed).fillna(False)
    else:
        exits = base_exits

    return entries, exits
