"""
strategy.py — Bayesian-optimized synthesis: Track D (funding) + Track A (vol regime).

Combines Track D's funding rate percentile with Track A's Garman-Klass vol
regime gate. Parameters jointly optimized via Optuna TPE (200 trials).

Key finding from optimization: the vol filter does heavy lifting for regime
selection (vol_entry_pct=0.455), allowing the funding filter to be more
permissive (fr_entry_pct=0.688) than its standalone optimum (0.52).

Full 5-window fitness: +2.030  |  Walk-forward OOS (W4-5): +1.991
"""

from __future__ import annotations

import pandas as pd

PARAMS: dict = {
    "sma_period": 256,
    "fr_pct_window": 408,
    "fr_entry_pct": 0.6879,
    "fr_exit_pct": 0.7784,
    "exit_lookback": 60,
    "vol_lookback": 53,
    "vol_pct_window": 1104,
    "vol_entry_pct": 0.4553,
    "vol_exit_pct": 0.5766,
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
