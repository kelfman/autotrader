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

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

log = logging.getLogger(__name__)


# ── Look-ahead bias guard (§6.8.8) ───────────────────────────────────────────

RETURN_CORR_WARN = 0.10
RETURN_CORR_FAIL = 0.15


@dataclass
class SignalIntegrityResult:
    daily_return_corr: float
    status: str            # "clean", "warning", "fail"

    @property
    def clean(self) -> bool:
        return self.status == "clean"

    def summary(self) -> str:
        return f"daily_return_corr={self.daily_return_corr:+.4f} [{self.status.upper()}]"


def check_signal_integrity(
    df: pd.DataFrame,
    params: dict,
    compute_signals_fn: Callable,
    entries: pd.Series,
    exits: pd.Series,
) -> SignalIntegrityResult:
    """
    Detect look-ahead bias via daily return correlation.

    Resamples entry signals and close prices to daily resolution, then
    measures the correlation between daily entry rate and same-day returns.
    A clean strategy's daily entry rate has near-zero correlation with
    same-day returns (~0.02). Look-ahead bias inflates this because the
    strategy "sees" today's price direction and enters accordingly (~0.19
    for the known-broken MTF case).

    Thresholds (calibrated against known clean/broken strategies):
      < 0.10  clean    (V2 D+A ≈ 0.02, fixed MTF ≈ 0.02)
      0.10–0.15  warning  (mild momentum bias, investigate)
      > 0.15  fail     (broken MTF ≈ 0.19, likely look-ahead)
    """
    return_corr = 0.0
    try:
        daily_entry_rate = entries.astype(float).resample("1D").mean()
        daily_return = df["close"].resample("1D").last().pct_change()
        common_idx = daily_entry_rate.dropna().index.intersection(daily_return.dropna().index)
        if len(common_idx) > 30:
            return_corr = float(
                daily_entry_rate.loc[common_idx].corr(daily_return.loc[common_idx])
            )
            if not np.isfinite(return_corr):
                return_corr = 0.0
    except Exception:
        return_corr = 0.0

    if abs(return_corr) >= RETURN_CORR_FAIL:
        status = "fail"
    elif abs(return_corr) >= RETURN_CORR_WARN:
        status = "warning"
    else:
        status = "clean"

    return SignalIntegrityResult(
        daily_return_corr=return_corr,
        status=status,
    )


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
    slippage: float = 0.0005,    # 0.05% per side — realistic market impact
) -> BacktestResult:
    """
    Run a single backtest over the provided OHLCV window.

    Args:
        df:                 OHLCV DataFrame (UTC DatetimeIndex, columns: open high low close volume).
        params:             Strategy parameter dict.
        compute_signals_fn: Strategy function with signature (df, params) -> (entries, exits).
        init_cash:          Starting portfolio cash in USD.
        fees:               Taker fee fraction per trade (e.g. 0.001 = 0.1%).
        slippage:           Additional slippage fraction per trade (default 0.05%).

    Returns:
        BacktestResult with performance metrics.
    """
    if len(df) < 50:
        return _empty_result(df)

    # ── signals ───────────────────────────────────────────────────────────────
    try:
        result = compute_signals_fn(df, params)
    except Exception as e:
        log.error("compute_signals failed: %s", e)
        return _empty_result(df)

    size = None
    if isinstance(result, tuple) and len(result) == 3:
        entries, exits, size = result
    else:
        entries, exits = result

    entries = entries.fillna(False).astype(bool)
    exits   = exits.fillna(False).astype(bool)

    if not entries.any():
        return _empty_result(df)

    # ── look-ahead bias guard ─────────────────────────────────────────────────
    integrity = check_signal_integrity(df, params, compute_signals_fn, entries, exits)
    if integrity.status == "fail":
        log.warning(
            "LOOK-AHEAD DETECTED: %s — signals likely contain future information!",
            integrity.summary(),
        )
    elif integrity.status == "warning":
        log.info(
            "LOOK-AHEAD CHECK: %s — mild same-day correlation, investigate if using MTF features",
            integrity.summary(),
        )

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
