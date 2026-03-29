"""
strategy.py — initial strategy for Track C (Cross-pair signals (BTC / ETH ratio)).

This is a minimal starting point. The agent will replace it.
Signal class: cross_pair
"""

from __future__ import annotations

import pandas as pd

PARAMS: dict = {
    "lookback": 168,
    "entry_pct": 0.3,
    "exit_pct": 0.4,
}


def compute_signals(df, params):
    import pandas as pd
    import numpy as np
    close = df['close']
    eth_close = df['eth_close']
    ratio = close / eth_close
    lookback = int(params['lookback'])
    ratio_pctile = ratio.rolling(lookback).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    entries = (ratio_pctile < params['entry_pctile']).fillna(False)
    exits = (ratio_pctile > params['exit_pctile']).fillna(False)
    return entries, exits
