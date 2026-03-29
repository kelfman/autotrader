"""
strategy.py — Manual synthesis: Track D (funding rates) + Track A (vol regime).

Combines the two best structural signals from V2:
  - Track D: Funding rate percentile rank as primary entry filter (+1.057 fitness)
  - Track A: Garman-Klass vol + vol-of-vol stability as regime gate (+0.311 fitness)

Hypothesis: vol-of-vol gates out the unstable periods that weaken Track D's
W4/W5 performance, while funding rate provides the directional signal that
Track A lacked (fixing its low trade count problem).

Requires both funding_rate and OHLCV columns in the dataframe.
"""

from __future__ import annotations

import pandas as pd

PARAMS: dict = {
    "sma_period": 192,
    "fr_pct_window": 720,
    "fr_entry_pct": 0.52,
    "fr_exit_pct": 0.90,
    "exit_lookback": 38,
    "vol_lookback": 24,
    "vol_pct_window": 1440,
    "vol_entry_pct": 0.65,
    "vol_exit_pct": 0.80,
}


def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    """
    Funding percentile + vol regime gated momentum.

    Entry: close > SMA (trend) AND funding_pct < threshold (not crowded)
           AND vol_pct < threshold (calm regime)
    Exit:  funding_pct > exit threshold (crowding) OR close < trailing low
           OR vol_pct > exit threshold (vol spike)
    """
    import numpy as np

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    funding = df["funding_rate"]

    # -- Track D signals: funding rate percentile + SMA momentum --
    sma = close.rolling(int(params["sma_period"])).mean()
    price_momentum = close > sma

    fr_window = int(params["fr_pct_window"])
    fr_pct = funding.rolling(fr_window).rank(pct=True)
    funding_ok = fr_pct < params["fr_entry_pct"]

    # -- Track A signals: GK vol percentile --
    vol_lb = int(params["vol_lookback"])
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    realized_vol = np.sqrt(gk_var.rolling(vol_lb).mean() * 252 * 24)

    vol_pct_win = int(params["vol_pct_window"])
    vol_pct = realized_vol.rolling(vol_pct_win).rank(pct=True)
    vol_calm = vol_pct < params["vol_entry_pct"]

    # -- Combined entry --
    entries = (price_momentum & funding_ok & vol_calm).fillna(False)

    # -- Combined exit --
    exit_lb = int(params["exit_lookback"])
    trailing_low = close.shift(1).rolling(exit_lb).min()
    funding_crowded = fr_pct > params["fr_exit_pct"]
    vol_spike = vol_pct > params["vol_exit_pct"]

    exits = (funding_crowded | (close < trailing_low) | vol_spike).fillna(False)

    return entries, exits
