"""
strategy.py — V4 Sociology: washout → compression → ignition.

Macro contrarian-to-momentum strategy targeting ~10-20 trades/year with
>100bp edge per trade. Uses Fear & Greed Index as primary behavioral
signal combined with funding rate positioning and vol regime.

The mechanism: after a fear washout clears leveraged positions, vol compresses
as selling exhausts. Entry fires when fear starts recovering (ignition).
Exit when greed/funding extremes signal participant exhaustion.

Requires: 'fng_value' column in DataFrame (daily Fear & Greed, forward-filled
to 1h by the data loader). Also requires 'funding_rate' (existing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS: dict = {
    # Washout detection: FNG must have been below this recently
    "fng_washout_level": 24,
    "fng_washout_lookback": 624,    # hours (~26 days)

    # Ignition entry: FNG rising from washout, crosses above this
    "fng_entry_level": 29,

    # Exit: greed exhaustion
    "fng_exit_level": 81,

    # Funding rate confirmation
    "fr_pct_window": 504,
    "fr_entry_pct": 0.3691,
    "fr_exit_pct": 0.7917,

    # Vol regime (Garman-Klass)
    "vol_lookback": 51,
    "vol_pct_window": 1776,
    "vol_entry_max_pct": 0.3324,

    # Trend confirmation: price above SMA
    "sma_period": 120,

    # Protective exit
    "exit_lookback": 156,
}


def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    fng = df["fng_value"].astype(float)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    funding = df["funding_rate"]

    # -- Fear & Greed: washout detection --
    washout_level = params["fng_washout_level"]
    washout_lb = int(params["fng_washout_lookback"])
    fng_min_recent = fng.rolling(washout_lb, min_periods=1).min()
    had_washout = fng_min_recent < washout_level

    # -- Fear & Greed: ignition (rising from washout) --
    entry_level = params["fng_entry_level"]
    fng_rising = fng > fng.shift(24)  # FNG higher than 1 day ago
    fng_entry = (fng >= entry_level) & fng_rising & had_washout

    # -- Funding rate percentile --
    fr_window = int(params["fr_pct_window"])
    fr_pct = funding.rolling(fr_window, min_periods=max(1, fr_window // 4)).rank(pct=True)
    funding_ok = fr_pct < params["fr_entry_pct"]

    # -- Vol regime (Garman-Klass) --
    vol_lb = int(params["vol_lookback"])
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    realized_vol = np.sqrt(gk_var.rolling(vol_lb).mean() * 252 * 24)
    vol_pct_win = int(params["vol_pct_window"])
    vol_pct = realized_vol.rolling(vol_pct_win, min_periods=max(1, vol_pct_win // 4)).rank(pct=True)
    vol_ok = vol_pct < params["vol_entry_max_pct"]

    # -- Trend confirmation --
    sma = close.rolling(int(params["sma_period"])).mean()
    trend_ok = close > sma

    # -- Combined entry --
    entries = (fng_entry & funding_ok & vol_ok & trend_ok).fillna(False)

    # -- Exits --
    fng_greed = fng > params["fng_exit_level"]
    funding_crowded = fr_pct > params["fr_exit_pct"]
    exit_lb = int(params["exit_lookback"])
    trailing_low = close.shift(1).rolling(exit_lb).min()
    price_stop = close < trailing_low

    exits = (fng_greed | funding_crowded | price_stop).fillna(False)

    return entries, exits
