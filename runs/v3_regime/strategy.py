"""
strategy.py — V3 regime-switching D+A strategy.

Extends the V2 D+A synthesis with a regime quality gate that filters out
low-conviction setups. The goal: fewer trades of higher quality, reducing
slippage drag (the binding constraint at 5bp/side with ~100 trades/year).

Regime classification uses three signals already computed by D+A:
  1. SMA slope — trend must be strengthening, not just above a declining average
  2. Vol-of-vol — vol must be stable, not just calm (avoids entry during
     vol regime transitions that often precede reversals)
  3. Funding raw level — funding should be positive (market paying for longs)
     as an additional confidence filter beyond percentile rank

When regime is unfavorable: no entries, force exit any open position.
When regime is favorable: run standard D+A logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS: dict = {
    # -- D+A base (from V2 ensemble) --
    "sma_period": 266,
    "fr_pct_window": 418,
    "fr_entry_pct": 0.6898,
    "fr_exit_pct": 0.7604,
    "exit_lookback": 55,
    "vol_lookback": 52,
    "vol_pct_window": 1070,
    "vol_entry_pct": 0.5088,
    "vol_exit_pct": 0.5645,
    # -- Regime gate --
    "sma_slope_lookback": 48,
    "min_sma_slope": 0.0,
    "vov_window": 168,
    "vov_threshold": 0.15,
    "min_funding_raw": 0.0,
    "use_regime_gate": True,
}


def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    funding = df["funding_rate"]

    # -- Standard D+A signals --
    sma = close.rolling(int(params["sma_period"])).mean()
    price_momentum = close > sma

    fr_window = int(params["fr_pct_window"])
    fr_pct = funding.rolling(fr_window).rank(pct=True)
    funding_ok = fr_pct < params["fr_entry_pct"]

    vol_lb = int(params["vol_lookback"])
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    realized_vol = np.sqrt(gk_var.rolling(vol_lb).mean() * 252 * 24)

    vol_pct_win = int(params["vol_pct_window"])
    vol_pct = realized_vol.rolling(vol_pct_win).rank(pct=True)
    vol_calm = vol_pct < params["vol_entry_pct"]

    # -- Regime quality gate --
    if params.get("use_regime_gate", True):
        slope_lb = int(params["sma_slope_lookback"])
        sma_slope = (sma - sma.shift(slope_lb)) / sma.shift(slope_lb)
        trend_rising = sma_slope > params["min_sma_slope"]

        vov_win = int(params["vov_window"])
        vol_of_vol = vol_pct.rolling(vov_win).std()
        vol_stable = vol_of_vol < params["vov_threshold"]

        funding_positive = funding > params["min_funding_raw"]

        regime_favorable = trend_rising & vol_stable & funding_positive
    else:
        regime_favorable = pd.Series(True, index=df.index)

    # -- Entry: D+A logic gated by regime --
    entries = (price_momentum & funding_ok & vol_calm & regime_favorable).fillna(False)

    # -- Exit: standard D+A exits + force exit when regime turns unfavorable --
    exit_lb = int(params["exit_lookback"])
    trailing_low = close.shift(1).rolling(exit_lb).min()
    funding_crowded = fr_pct > params["fr_exit_pct"]
    vol_spike = vol_pct > params["vol_exit_pct"]

    exits = (funding_crowded | (close < trailing_low) | vol_spike | ~regime_favorable).fillna(False)

    return entries, exits
