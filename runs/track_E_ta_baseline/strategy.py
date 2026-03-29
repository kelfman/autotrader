"""
strategy.py — the only file the agent is allowed to edit.

Structure contract (must always be preserved):
  PARAMS             dict of named numeric parameters
  compute_signals()  function with signature (df, params) -> (entries, exits)

The research loop imports this module and calls compute_signals(df, PARAMS).
backtest.py / evaluate.py never import strategy.py directly — the loop
passes the function reference in, keeping harness and strategy fully decoupled.

Notes on using `ta` for indicators:
  pandas-ta is incompatible with Python 3.10+, so we use the `ta` library.
  API: ta.momentum.rsi(close, window=N), ta.trend.ema_indicator(close, window=N), etc.
  Full reference: https://technical-analysis-library-in-python.readthedocs.io/
"""

from __future__ import annotations

import pandas as pd
import ta

# ── Editable parameter block ─────────────────────────────────────────────────
# The agent modifies values here (or adds new keys) as part of its StrategySpec.
PARAMS: dict = {
    "ema_fast":            12,
    "ema_slow":            26,
    "rsi_period":          14,
    "rsi_entry_min":       55,
    "rsi_exit_overbought": 70,
    "min_volume_ma":       20,
    "adx_period":          14,
    "adx_threshold":       20,
    "ema_trend":           50,
}


# ── Signal logic — the agent may rewrite this function ───────────────────────
def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    """
    Compute long entry / exit signals for a single asset.

    Args:
        df:     OHLCV DataFrame with columns [open, high, low, close, volume]
                and a UTC DatetimeIndex.
        params: parameter dict (use PARAMS or an agent-supplied override).

    Returns:
        entries: boolean Series — True on bars where we go long
        exits:   boolean Series — True on bars where we close the long

    Strategy (baseline — EMA crossover with RSI confirmation):
        Entry:  EMA fast crosses above EMA slow (bullish cross)
                AND RSI > rsi_entry_min (momentum confirmed, not a false breakout)
                AND volume above rolling average
        Exit:   EMA fast crosses below EMA slow (trend ends)
                OR RSI crosses above rsi_exit_overbought (overbought, take profit)

    Design rationale:
        EMA crossover is the primary trend signal; RSI acts as a secondary
        momentum filter to reduce whipsaw entries during choppy markets.
        Both conditions can co-occur naturally — unlike combining RSI oversold
        with EMA uptrend, which are structurally contradictory.

    The agent may replace this logic entirely, add short signals,
    introduce new indicators, etc. The function signature must not change.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── indicators ────────────────────────────────────────────────────────────
    ema_fast  = ta.trend.ema_indicator(close, window=int(params["ema_fast"]))
    ema_slow  = ta.trend.ema_indicator(close, window=int(params["ema_slow"]))
    ema_trend = ta.trend.ema_indicator(close, window=int(params["ema_trend"]))
    rsi       = ta.momentum.rsi(close, window=int(params["rsi_period"]))
    vol_ma    = volume.rolling(int(params["min_volume_ma"])).mean()
    adx       = ta.trend.adx(high, low, close, window=int(params["adx_period"]))

    # ── derived conditions ────────────────────────────────────────────────────
    ema_cross_up   = (ema_fast.shift(1) <= ema_slow.shift(1)) & (ema_fast > ema_slow)
    ema_cross_down = (ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)

    momentum_ok  = (rsi > params["rsi_entry_min"]) & (rsi > rsi.shift(1))
    high_vol     = volume > vol_ma
    is_trending  = adx > params["adx_threshold"]
    bull_regime  = close > ema_trend   # macro filter: only long above EMA50

    rsi_prev     = rsi.shift(1)
    rsi_cross_ob = (rsi_prev <= params["rsi_exit_overbought"]) & (rsi > params["rsi_exit_overbought"])

    # ── entries & exits ───────────────────────────────────────────────────────
    entries = ema_cross_up & momentum_ok & high_vol & is_trending & bull_regime
    trend_fading = (adx < params["adx_threshold"]) & (adx.shift(1) < params["adx_threshold"])
    exits   = ema_cross_down | rsi_cross_ob | trend_fading

    entries = entries.fillna(False)
    exits   = exits.fillna(False)

    return entries, exits
