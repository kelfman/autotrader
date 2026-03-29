"""
strategy.py — initial strategy for Track A (Statistical / volatility regime detection).

This is a minimal starting point. The agent will replace it.
Signal class: statistical_vol_regime
"""

from __future__ import annotations

import pandas as pd

PARAMS: dict = {
    "vol_lookback": 24,
    "percentile_window": 1440,
    "entry_vol_pct": 0.47,
    "exit_vol_pct": 0.65,
    "vov_lookback": 168,
    "vov_threshold": 0.4,
}


def compute_signals(df, params):
    import numpy as np
    import pandas as pd

    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']

    vol_lookback = int(params['vol_lookback'])
    pct_window = int(params['percentile_window'])
    entry_vol = params['entry_vol_pct']
    exit_vol = params['exit_vol_pct']
    vov_lookback = int(params['vov_lookback'])
    vov_thresh = params['vov_threshold']

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    realized_vol = np.sqrt(gk_var.rolling(vol_lookback).mean() * 252 * 24)

    vol_pct = realized_vol.rolling(pct_window).rank(pct=True)

    # Vol-of-vol: stability of vol itself
    vol_of_vol = realized_vol.rolling(vov_lookback).std()
    vov_pct = vol_of_vol.rolling(pct_window).rank(pct=True)

    # Enter when vol is calm AND stable
    entries = ((vol_pct < entry_vol) & (vov_pct < vov_thresh)).fillna(False)
    # Exit when vol rises to turbulent
    exits = (vol_pct > exit_vol).fillna(False)

    return entries, exits
