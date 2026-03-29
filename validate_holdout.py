"""
validate_holdout.py — held-out validation for promoted strategies.

Evaluates a strategy on the date ranges BETWEEN the training windows
used during the research loop, providing an independent check against
overfitting to the training windows.

Training windows (5 × 91 days over 730 days):
    W1: day 0–91,  W2: day 160–251,  W3: day 320–411,
    W4: day 480–571,  W5: day 639–730

Held-out windows (the gaps):
    H1: day 91–160  (between W1 and W2)
    H2: day 251–320 (between W2 and W3)
    H3: day 411–480 (between W3 and W4)
    H4: day 571–639 (between W4 and W5)

Usage:
    python validate_holdout.py --strategy-path runs/track_D_funding_rates/strategy.py --augment-funding
    python validate_holdout.py --strategy-path runs/track_D_funding_rates/strategy.py --augment-funding --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BacktestResult, run_backtest
from data import fetch_funding_rates, fetch_ohlcv

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent

TOTAL_DAYS   = 730
WINDOW_DAYS  = 91
N_WINDOWS    = 5
LAMBDA       = 0.5


def load_strategy_module(strategy_path: Path):
    spec = importlib.util.spec_from_file_location("_strategy_holdout", strategy_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_signals, mod.PARAMS


def augment_with_eth(df: pd.DataFrame) -> pd.DataFrame:
    eth = fetch_ohlcv("ETH/USDT", "1h", days=TOTAL_DAYS)
    eth = eth.rename(columns={c: f"eth_{c}" for c in eth.columns})
    merged = df.join(eth, how="left")
    merged[list(eth.columns)] = merged[list(eth.columns)].ffill()
    return merged


def augment_with_funding(df: pd.DataFrame) -> pd.DataFrame:
    funding = fetch_funding_rates("BTC/USDT:USDT", days=TOTAL_DAYS)
    merged = df.join(funding, how="left")
    merged["funding_rate"] = merged["funding_rate"].ffill().fillna(0.0)
    return merged


def _compute_windows(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """
    Compute training and held-out windows using the same logic as evaluate.py.
    Returns (training_windows, holdout_windows) as lists of
    {start, end, df} dicts.
    """
    total_days = (df.index[-1] - df.index[0]).days
    step_days = (total_days - WINDOW_DAYS) / max(N_WINDOWS - 1, 1)
    start_ts = df.index[0]

    training = []
    for i in range(N_WINDOWS):
        w_start = start_ts + pd.Timedelta(days=round(i * step_days))
        w_end   = w_start  + pd.Timedelta(days=WINDOW_DAYS)
        w_df = df[(df.index >= w_start) & (df.index < w_end)].copy()
        training.append({"start": w_start, "end": w_end, "df": w_df})

    holdout = []
    for i in range(len(training) - 1):
        h_start = training[i]["end"]
        h_end   = training[i + 1]["start"]
        if h_end <= h_start:
            continue
        h_df = df[(df.index >= h_start) & (df.index < h_end)].copy()
        if len(h_df) > 50:
            holdout.append({"start": h_start, "end": h_end, "df": h_df})

    return training, holdout


def run_holdout_validation(
    compute_signals_fn,
    params: dict,
    df: pd.DataFrame,
    init_cash: float = 10_000.0,
    fees: float = 0.001,
) -> dict:
    """Run backtests on held-out windows and return a full results dict."""
    training_windows, holdout_windows = _compute_windows(df)

    log.info("Training windows: %d, Held-out windows: %d",
             len(training_windows), len(holdout_windows))

    training_results: list[BacktestResult] = []
    for i, w in enumerate(training_windows, 1):
        r = run_backtest(w["df"], params, compute_signals_fn, init_cash=init_cash, fees=fees)
        training_results.append(r)
        log.info("  Training W%d: %s", i, r.summary())

    holdout_results: list[BacktestResult] = []
    for i, w in enumerate(holdout_windows, 1):
        r = run_backtest(w["df"], params, compute_signals_fn, init_cash=init_cash, fees=fees)
        holdout_results.append(r)
        log.info("  Holdout  H%d: %s", i, r.summary())

    def _fitness(results: list[BacktestResult]) -> tuple[float, float, float]:
        sharpes = np.array([r.sharpe if r.is_valid() else 0.0 for r in results])
        mean_s = float(np.mean(sharpes))
        std_s  = float(np.std(sharpes))
        return mean_s - LAMBDA * std_s, mean_s, std_s

    train_fitness, train_mean, train_std = _fitness(training_results)
    holdout_fitness, holdout_mean, holdout_std = _fitness(holdout_results)

    return {
        "training": {
            "fitness": train_fitness,
            "mean_sharpe": train_mean,
            "std_sharpe": train_std,
            "windows": [
                {
                    "label": f"W{i+1}",
                    "start": str(training_windows[i]["start"].date()),
                    "end": str(training_windows[i]["end"].date()),
                    "sharpe": r.sharpe,
                    "return_pct": r.total_return_pct,
                    "mdd_pct": r.max_drawdown_pct,
                    "trades": r.n_trades,
                    "win_rate": r.win_rate,
                    "valid": r.is_valid(),
                }
                for i, r in enumerate(training_results)
            ],
        },
        "holdout": {
            "fitness": holdout_fitness,
            "mean_sharpe": holdout_mean,
            "std_sharpe": holdout_std,
            "windows": [
                {
                    "label": f"H{i+1}",
                    "start": str(holdout_windows[i]["start"].date()),
                    "end": str(holdout_windows[i]["end"].date()),
                    "sharpe": r.sharpe,
                    "return_pct": r.total_return_pct,
                    "mdd_pct": r.max_drawdown_pct,
                    "trades": r.n_trades,
                    "win_rate": r.win_rate,
                    "valid": r.is_valid(),
                }
                for i, r in enumerate(holdout_results)
            ],
        },
    }


def _print_report(results: dict, params: dict, strategy_path: str) -> None:
    train = results["training"]
    hold = results["holdout"]

    print()
    print("═" * 90)
    print(f"  HELD-OUT VALIDATION REPORT")
    print(f"  Strategy: {strategy_path}")
    print(f"  Params: {json.dumps(params)}")
    print("═" * 90)

    def _table(label: str, data: dict) -> None:
        print(f"\n  {label}")
        print(f"  Fitness: {data['fitness']:+.4f}  "
              f"(mean_sharpe={data['mean_sharpe']:+.3f}, std={data['std_sharpe']:.3f}, λ={LAMBDA})")
        print()
        print(f"    {'':>4}  {'Period':<25}  {'Sharpe':>7}  {'Return':>7}  "
              f"{'MDD':>6}  {'Trades':>6}  {'WR':>5}  {'Valid':>5}")
        print(f"    {'─'*4}  {'─'*25}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*5}")
        for w in data["windows"]:
            valid = "✓" if w["valid"] else "⚠"
            print(
                f"    {w['label']:>4}  {w['start']} → {w['end']:<10}  "
                f"{w['sharpe']:>+7.3f}  {w['return_pct']:>+6.1f}%  "
                f"{w['mdd_pct']:>5.1f}%  {w['trades']:>6}  "
                f"{w['win_rate']*100:>4.0f}%  {valid:>5}"
            )
        all_positive = all(w["sharpe"] > 0 for w in data["windows"] if w["valid"])
        print(f"\n    All windows positive: {'Yes ✓' if all_positive else 'No ✗'}")

    _table("TRAINING WINDOWS (used during research loop)", train)
    _table("HELD-OUT WINDOWS (never seen during research)", hold)

    print()
    print("─" * 90)
    fitness_decay = train["fitness"] - hold["fitness"]
    decay_pct = (fitness_decay / abs(train["fitness"]) * 100) if train["fitness"] != 0 else 0
    print(f"  Training fitness:  {train['fitness']:+.4f}")
    print(f"  Held-out fitness:  {hold['fitness']:+.4f}")
    print(f"  Decay:             {fitness_decay:+.4f}  ({decay_pct:+.1f}%)")
    print()

    if hold["fitness"] > 0.4714:
        print("  ★ HELD-OUT FITNESS BEATS TA BASELINE (+0.4714)")
    if hold["fitness"] > 0:
        print("  ✓ Held-out fitness is positive — strategy generalises beyond training windows")
    else:
        print("  ✗ Held-out fitness is negative — possible overfitting to training windows")

    print("═" * 90)
    print()


def main():
    p = argparse.ArgumentParser(
        description="Held-out validation for promoted strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--strategy-path", type=str, required=True,
                   help="Path to strategy.py to validate")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--augment-eth", action="store_true",
                   help="Merge ETH/USDT data (Track C)")
    p.add_argument("--augment-funding", action="store_true",
                   help="Merge funding rate data (Track D)")
    p.add_argument("--json", action="store_true",
                   help="Output raw JSON instead of formatted report")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    strategy_path = Path(args.strategy_path)
    compute_signals, params = load_strategy_module(strategy_path)

    df = fetch_ohlcv(args.symbol, "1h", days=TOTAL_DAYS)
    if args.augment_eth:
        df = augment_with_eth(df)
    if args.augment_funding:
        df = augment_with_funding(df)

    results = run_holdout_validation(compute_signals, params, df)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_report(results, params, str(strategy_path))


if __name__ == "__main__":
    main()
