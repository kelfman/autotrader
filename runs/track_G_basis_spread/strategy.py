"""
strategy.py — Track G: Basis spread z-score as speculative demand signal.

Enter when basis z-score is extreme negative (distress/backwardation) AND
price is above SMA (trend confirmation prevents catching falling knives).
Exit when basis z-score reverts positive (demand returns) or trailing stop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS: dict = {
    "sma_period": 192,
    "z_window": 720,
    "z_entry": -1.0,
    "z_exit": 1.5,
    "exit_lookback": 38,
}


def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    close = df["close"]
    basis_pct = df["basis_pct"]

    sma = close.rolling(int(params["sma_period"])).mean()
    price_momentum = close > sma

    z_win = int(params["z_window"])
    bp_mean = basis_pct.rolling(z_win).mean()
    bp_std = basis_pct.rolling(z_win).std()
    bp_z = (basis_pct - bp_mean) / bp_std.replace(0, np.nan)

    entries = (price_momentum & (bp_z < params["z_entry"])).fillna(False)

    exit_lb = int(params["exit_lookback"])
    trailing_low = close.shift(1).rolling(exit_lb).min()
    basis_reverted = bp_z > params["z_exit"]

    exits = (basis_reverted | (close < trailing_low)).fillna(False)

    return entries, exits
