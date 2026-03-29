"""
strategy.py — initial strategy for Track B (Calendar / trading session effects).

This is a minimal starting point. The agent will replace it.
Signal class: calendar_session
"""

from __future__ import annotations

import pandas as pd

PARAMS: dict = {
    "us_open_hour": 13,
    "us_close_hour": 21,
}


def compute_signals(df, params):
    hour = df.index.hour
    entries = (hour == int(params['us_open_hour']))
    exits = (hour == int(params['us_close_hour']))
    return entries.fillna(False), exits.fillna(False)
