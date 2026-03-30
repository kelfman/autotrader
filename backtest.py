"""
backtest.py — stable evaluation harness. DO NOT EDIT during research loop.

Takes a windowed slice of OHLCV data + a strategy's compute_signals function,
runs a vectorbt portfolio simulation, and returns a BacktestResult.

The research loop calls run_backtest() once per window. evaluate.py calls it
five times (one per regime window) to build the fitness score.

Usage:
    from backtest import run_backtest
    from strategy import compute_signals, PARAMS

    result = run_backtest(df_window, PARAMS, compute_signals)
    print(result.sharpe, result.total_return_pct, result.max_drawdown_pct)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Outcome of a single backtest window."""
    sharpe:            float
    sortino:           float
    total_return_pct:  float
    max_drawdown_pct:  float
    n_trades:          int
    win_rate:          float        # fraction of closed trades that were profitable
    window_start:      pd.Timestamp
    window_end:        pd.Timestamp
    # raw stats dict from vectorbt (for debugging / tearsheet)
    raw_stats:         dict = field(default_factory=dict, repr=False)

    def is_valid(self) -> bool:
        """True if the backtest had enough trades to trust the metrics."""
        return self.n_trades >= 3

    def summary(self) -> str:
        return (
            f"[{self.window_start.date()} → {self.window_end.date()}] "
            f"Sharpe={self.sharpe:+.3f}  Return={self.total_return_pct:+.1f}%  "
            f"MDD={self.max_drawdown_pct:.1f}%  Trades={self.n_trades}  "
            f"WinRate={self.win_rate*100:.0f}%"
        )


# ── Core function ─────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    params: dict,
    compute_signals_fn: Callable,
    init_cash: float = 10_000.0,
    fees: float = 0.001,         # 0.1 % per side — conservative for BTC spot on Binance
    slippage: float = 0.0,       # set >0 to model market-impact
) -> BacktestResult:
    """
    Run a single backtest over the provided OHLCV window.

    Args:
        df:                 OHLCV DataFrame (UTC DatetimeIndex, columns: open high low close volume).
        params:             Strategy parameter dict.
        compute_signals_fn: Strategy function with signature (df, params) -> (entries, exits).
        init_cash:          Starting portfolio cash in USD.
        fees:               Taker fee fraction per trade (e.g. 0.001 = 0.1%).
        slippage:           Additional slippage fraction per trade.

    Returns:
        BacktestResult with performance metrics.
    """
    if len(df) < 50:
        # Not enough data to produce meaningful indicators
        return _empty_result(df)

    # ── signals ───────────────────────────────────────────────────────────────
    try:
        result = compute_signals_fn(df, params)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("compute_signals failed: %s", e)
        return _empty_result(df)

    # V3 contract: compute_signals may return (entries, exits, size) where
    # size is a 0.0–1.0 Series representing fraction of capital to allocate.
    # V1/V2 contract: (entries, exits) — backward compatible, defaults to 1.0.
    size = None
    if isinstance(result, tuple) and len(result) == 3:
        entries, exits, size = result
    else:
        entries, exits = result

    entries = entries.fillna(False).astype(bool)
    exits   = exits.fillna(False).astype(bool)

    # Guard: if no entries at all, skip (avoids vbt divide-by-zero on Sharpe)
    if not entries.any():
        return _empty_result(df)

    # ── portfolio simulation ──────────────────────────────────────────────────
    total_fees = fees + slippage

    if size is not None:
        size_clean = size.fillna(1.0).clip(0.0, 1.0)
        # Per-bar sizing: use strategy's fraction on entry bars, np.inf elsewhere
        # so exits close the full position regardless of the size Series.
        size_arr = np.where(entries.values, size_clean.values, np.inf)
        portfolio = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=entries,
            exits=exits,
            size=size_arr,
            size_type=SizeType.Percent,
            init_cash=init_cash,
            fees=total_fees,
            freq="1h",
        )
    else:
        portfolio = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=entries,
            exits=exits,
            init_cash=init_cash,
            fees=total_fees,
            freq="1h",
        )

    stats = portfolio.stats()

    # ── extract metrics ───────────────────────────────────────────────────────
    sharpe           = _safe_float(stats.get("Sharpe Ratio"))
    sortino          = _safe_float(stats.get("Sortino Ratio"))
    total_return_pct = _safe_float(stats.get("Total Return [%]"))
    max_drawdown_pct = abs(_safe_float(stats.get("Max Drawdown [%]")))
    n_trades         = int(stats.get("Total Closed Trades", 0) or 0)

    # Win rate: vectorbt doesn't expose directly — compute from trade records
    win_rate = _compute_win_rate(portfolio)

    return BacktestResult(
        sharpe=sharpe,
        sortino=sortino,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        n_trades=n_trades,
        win_rate=win_rate,
        window_start=df.index[0],
        window_end=df.index[-1],
        raw_stats=dict(stats),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(value) -> float:
    """Convert a potentially NaN/None metric to a float, defaulting to 0."""
    try:
        v = float(value)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _compute_win_rate(portfolio: vbt.Portfolio) -> float:
    try:
        trades = portfolio.trades.records_readable
        if trades.empty:
            return 0.0
        wins = (trades["PnL"] > 0).sum()
        return wins / len(trades)
    except Exception:
        return 0.0


def _empty_result(df: pd.DataFrame) -> BacktestResult:
    start = df.index[0] if len(df) > 0 else pd.Timestamp.now(tz="UTC")
    end   = df.index[-1] if len(df) > 0 else pd.Timestamp.now(tz="UTC")
    return BacktestResult(
        sharpe=0.0,
        sortino=0.0,
        total_return_pct=0.0,
        max_drawdown_pct=0.0,
        n_trades=0,
        win_rate=0.0,
        window_start=start,
        window_end=end,
    )


if __name__ == "__main__":
    import logging
    from data import fetch_ohlcv
    from strategy import compute_signals, PARAMS

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = fetch_ohlcv("BTC/USDT", "1h", days=90)
    result = run_backtest(df, PARAMS, compute_signals)
    print(result.summary())
    print(f"  Sortino: {result.sortino:.3f}")
