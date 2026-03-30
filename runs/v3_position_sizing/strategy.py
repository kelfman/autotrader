"""
strategy.py — V3 Phase 1: D+A synthesis with continuous position sizing.

Extends the ensemble D+A strategy (§5.15.9) with a third output: a position
size Series (0.0–1.0) that replaces the binary vol/funding gates with
continuous conviction modulation.

Key change from V2: instead of vol_calm being a hard boolean gate, vol_pct
modulates position size continuously — more vol means smaller position, not
no position. This addresses the core failure mode from §5.14 where AND-gating
choked trade count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS: dict = {
    # Trend filter
    "sma_period": 266,
    # Funding rate signal
    "fr_pct_window": 418,
    "fr_entry_pct": 0.6898,
    "fr_exit_pct": 0.7604,
    # Exit
    "exit_lookback": 55,
    # Vol regime (now modulates size instead of gating entry)
    "vol_lookback": 52,
    "vol_pct_window": 1070,
    # Position sizing: vol-based
    "vol_size_floor": 0.2,       # minimum position size in calm vol
    "vol_size_ceiling": 0.8,     # maximum position size
    "vol_size_midpoint": 0.50,   # vol_pct at which size = midpoint of floor/ceiling
    # Position sizing: funding-based
    "fr_size_weight": 0.3,       # how much funding distance from threshold boosts size
}


def compute_signals(
    df: pd.DataFrame, params: dict,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    V3 contract: returns (entries, exits, size).

    Entry: close > SMA (trend) AND funding_pct < threshold (not crowded).
           Vol regime no longer gates entry — it modulates position size.
    Exit:  funding_pct > exit threshold (crowding) OR close < trailing low.
    Size:  continuous 0.0–1.0 based on vol regime and funding distance.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    funding = df["funding_rate"]

    # -- Trend filter --
    sma = close.rolling(int(params["sma_period"])).mean()
    price_momentum = close > sma

    # -- Funding rate percentile --
    fr_window = int(params["fr_pct_window"])
    fr_pct = funding.rolling(fr_window).rank(pct=True)
    funding_ok = fr_pct < params["fr_entry_pct"]

    # -- Entry: trend + funding only (vol modulates size, not entry) --
    entries = (price_momentum & funding_ok).fillna(False)

    # -- Exits: funding crowding OR trailing low --
    exit_lb = int(params["exit_lookback"])
    trailing_low = close.shift(1).rolling(exit_lb).min()
    funding_crowded = fr_pct > params["fr_exit_pct"]
    exits = (funding_crowded | (close < trailing_low)).fillna(False)

    # -- Position sizing: vol regime → continuous size --
    vol_lb = int(params["vol_lookback"])
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    realized_vol = np.sqrt(gk_var.rolling(vol_lb).mean() * 252 * 24)

    vol_pct_win = int(params["vol_pct_window"])
    vol_pct = realized_vol.rolling(vol_pct_win).rank(pct=True).fillna(0.5)

    floor = params["vol_size_floor"]
    ceiling = params["vol_size_ceiling"]
    midpoint = params["vol_size_midpoint"]

    # Linear interpolation: low vol_pct → ceiling, high vol_pct → floor
    # vol_pct=0 → ceiling, vol_pct=1 → floor
    vol_size = ceiling - (ceiling - floor) * vol_pct

    # Funding distance bonus: the further below the entry threshold, the larger
    fr_weight = params["fr_size_weight"]
    fr_distance = (params["fr_entry_pct"] - fr_pct).clip(lower=0)
    fr_bonus = fr_distance * fr_weight

    size = (vol_size + fr_bonus).clip(0.0, 1.0)

    return entries, exits, size
