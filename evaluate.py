"""
evaluate.py — multi-window fitness scorer. DO NOT EDIT during research loop.

Implements the regime-robustness fitness function from the project brief:

    fitness = mean(sharpe_across_windows) − λ × std(sharpe_across_windows)

Five rolling 3-month windows are distributed evenly across 2 years of data.
A strategy that earns Sharpe 1.4 consistently beats one that earns 2.5 in
one window and tanks in the others — the variance penalty does the work.

Usage:
    from evaluate import evaluate_strategy
    from strategy import compute_signals, PARAMS

    result = evaluate_strategy(compute_signals, PARAMS)
    print(result.fitness)
    print(result.summary())

    # or with custom symbol / lambda
    result = evaluate_strategy(compute_signals, PARAMS, symbol="ETH/USDT", lambda_penalty=0.3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from backtest import BacktestResult, run_backtest
from data import fetch_ohlcv

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_SYMBOL       = "BTC/USDT"
DEFAULT_TIMEFRAME    = "1h"
DEFAULT_DAYS         = 730           # 2 years of history
DEFAULT_WINDOW_DAYS  = 91            # 3 months per backtest window (~91 days)
DEFAULT_N_WINDOWS    = 5             # number of regime windows
DEFAULT_LAMBDA       = 0.5           # variance penalty weight


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class FitnessResult:
    """
    Full evaluation outcome for a strategy across all regime windows.

    fitness = mean(sharpe) − λ × std(sharpe)
    """
    fitness:          float           # primary optimisation target
    mean_sharpe:      float
    std_sharpe:       float
    lambda_penalty:   float
    window_results:   list[BacktestResult] = field(default_factory=list)
    symbol:           str = DEFAULT_SYMBOL
    timeframe:        str = DEFAULT_TIMEFRAME

    def summary(self) -> str:
        lines = [
            f"── Fitness: {self.fitness:+.4f}  "
            f"(mean_sharpe={self.mean_sharpe:+.3f}, std={self.std_sharpe:.3f}, λ={self.lambda_penalty})",
            f"   Symbol: {self.symbol}  Windows: {len(self.window_results)}",
        ]
        for i, r in enumerate(self.window_results, 1):
            marker = "✓" if r.is_valid() else "⚠"  # warn if too few trades
            lines.append(f"   {marker} W{i}: {r.summary()}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialisable dict for experiment_log.jsonl."""
        return {
            "fitness":        self.fitness,
            "mean_sharpe":    self.mean_sharpe,
            "std_sharpe":     self.std_sharpe,
            "lambda_penalty": self.lambda_penalty,
            "symbol":         self.symbol,
            "timeframe":      self.timeframe,
            "windows": [
                {
                    "window_start":      r.window_start.isoformat(),
                    "window_end":        r.window_end.isoformat(),
                    "sharpe":            r.sharpe,
                    "sortino":           r.sortino,
                    "total_return_pct":  r.total_return_pct,
                    "max_drawdown_pct":  r.max_drawdown_pct,
                    "n_trades":          r.n_trades,
                    "win_rate":          r.win_rate,
                    "is_valid":          r.is_valid(),
                }
                for r in self.window_results
            ],
        }


# ── Window generation ─────────────────────────────────────────────────────────

def _build_windows(
    df: pd.DataFrame,
    n_windows: int = DEFAULT_N_WINDOWS,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> list[pd.DataFrame]:
    """
    Slice df into n_windows non-overlapping windows of window_days each,
    evenly spaced across the full date range.

    With 5 windows × 3 months over 2 years, the windows land roughly at:
      [0–3], [4.5–7.5], [9–12], [13.5–16.5], [18–21] months from start
    …so they sample bull, bear, and sideways regimes across the period.
    """
    total_days = (df.index[-1] - df.index[0]).days
    if total_days < window_days * n_windows:
        log.warning(
            "Dataset (%d days) may be too short for %d × %d-day windows",
            total_days, n_windows, window_days,
        )

    step_days = (total_days - window_days) / max(n_windows - 1, 1)
    windows: list[pd.DataFrame] = []

    start_ts = df.index[0]
    for i in range(n_windows):
        w_start = start_ts + pd.Timedelta(days=round(i * step_days))
        w_end   = w_start  + pd.Timedelta(days=window_days)
        window  = df[(df.index >= w_start) & (df.index < w_end)].copy()
        if len(window) > 50:
            windows.append(window)
        else:
            log.warning("Window %d is too small (%d rows), skipping", i + 1, len(window))

    return windows


# ── Core function ─────────────────────────────────────────────────────────────

def evaluate_strategy(
    compute_signals_fn: Callable,
    params: dict,
    symbol:         str   = DEFAULT_SYMBOL,
    timeframe:      str   = DEFAULT_TIMEFRAME,
    days:           int   = DEFAULT_DAYS,
    n_windows:      int   = DEFAULT_N_WINDOWS,
    window_days:    int   = DEFAULT_WINDOW_DAYS,
    lambda_penalty: float = DEFAULT_LAMBDA,
    init_cash:      float = 10_000.0,
    fees:           float = 0.001,
    df:             pd.DataFrame | None = None,   # pass pre-loaded data to skip fetch
) -> FitnessResult:
    """
    Evaluate a strategy across N regime windows and return a FitnessResult.

    Args:
        compute_signals_fn: Strategy signal function (df, params) -> (entries, exits).
        params:             Strategy parameter dict.
        symbol:             ccxt market symbol, e.g. "BTC/USDT".
        timeframe:          Candle timeframe, e.g. "1h".
        days:               Total history to use (default 730 = 2 years).
        n_windows:          Number of regime windows (default 5).
        window_days:        Length of each window in days (default 91 ≈ 3 months).
        lambda_penalty:     Variance penalty weight λ (default 0.5).
        init_cash:          Starting portfolio value for each window.
        fees:               Taker fee per trade.
        df:                 Pre-loaded DataFrame — if None, fetches via data.py.

    Returns:
        FitnessResult with fitness score, per-window breakdown, and serialisation.
    """
    # ── data ──────────────────────────────────────────────────────────────────
    if df is None:
        log.info("Loading %s %s (%d days)…", symbol, timeframe, days)
        df = fetch_ohlcv(symbol, timeframe, days=days)

    # ── windows ───────────────────────────────────────────────────────────────
    windows = _build_windows(df, n_windows=n_windows, window_days=window_days)
    log.info("Running %d backtest windows…", len(windows))

    # ── backtests ─────────────────────────────────────────────────────────────
    results: list[BacktestResult] = []
    for i, window in enumerate(windows, 1):
        log.info(
            "  Window %d/%d  %s → %s  (%d bars)",
            i, len(windows),
            window.index[0].date(), window.index[-1].date(), len(window),
        )
        result = run_backtest(window, params, compute_signals_fn, init_cash=init_cash, fees=fees)
        results.append(result)
        log.info("    %s", result.summary())

    # ── fitness ───────────────────────────────────────────────────────────────
    sharpes = np.array([r.sharpe for r in results], dtype=float)

    # Treat windows with too few trades as 0 Sharpe (they're not informative)
    for i, r in enumerate(results):
        if not r.is_valid():
            log.warning("Window %d has too few trades (%d), treating Sharpe as 0", i + 1, r.n_trades)
            sharpes[i] = 0.0

    mean_sharpe = float(np.mean(sharpes))
    std_sharpe  = float(np.std(sharpes))
    fitness     = mean_sharpe - lambda_penalty * std_sharpe

    log.info(
        "Fitness: %.4f  (mean_sharpe=%.3f, std=%.3f, λ=%.1f)",
        fitness, mean_sharpe, std_sharpe, lambda_penalty,
    )

    return FitnessResult(
        fitness=fitness,
        mean_sharpe=mean_sharpe,
        std_sharpe=std_sharpe,
        lambda_penalty=lambda_penalty,
        window_results=results,
        symbol=symbol,
        timeframe=timeframe,
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    symbol = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SYMBOL

    from strategy import compute_signals, PARAMS

    result = evaluate_strategy(compute_signals, PARAMS, symbol=symbol)
    print()
    print(result.summary())
