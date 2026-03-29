"""
strategy.py — initial strategy for Track D (Perpetual futures funding rate signals).

This is a minimal starting point. The agent will replace it.
Signal class: funding_rates
"""

from __future__ import annotations

import pandas as pd

PARAMS: dict = {
    "sma_period": 192,
    "fr_pct_window": 720,
    "fr_entry_pct": 0.52,
    "fr_exit_pct": 0.9,
    "exit_lookback": 38,
}


def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    """
    Funding percentile-rank momentum strategy.

    Entry: close > SMA (slow momentum) AND funding percentile rank below
           threshold (not crowded longs — neutral to negative funding).
    Exit:  funding percentile rises above exit threshold (extreme crowding)
           OR close drops below trailing low.
    """
    close = df["close"]
    funding = df["funding_rate"]

    sma = close.rolling(int(params["sma_period"])).mean()
    price_momentum = close > sma

    fr_pct_window = int(params["fr_pct_window"])
    fr_pct = funding.rolling(fr_pct_window).rank(pct=True)

    entries = (price_momentum & (fr_pct < params["fr_entry_pct"])).fillna(False)

    exit_lookback = int(params["exit_lookback"])
    trailing_low = close.shift(1).rolling(exit_lookback).min()
    exits = ((fr_pct > params["fr_exit_pct"]) | (close < trailing_low)).fillna(False)

    return entries, exits
