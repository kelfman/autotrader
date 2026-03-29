"""
validate_walkforward.py — walk-forward validation for strategy generalization.

Evaluates a strategy using a temporal split: windows 1-3 are "in-sample"
(the period during which the strategy was optimized), windows 4-5 are
"out-of-sample" (unseen future data). This is a fairer OOS test than
gap-window holdout, since W4-W5 are contiguous forward periods rather
than structurally unusual transition gaps.

The 5 windows are the same as evaluate.py uses — evenly spaced 91-day
slices across 730 days of BTC/USDT 1h data.

Usage:
    python validate_walkforward.py --strategy-path runs/track_D_funding_rates/strategy.py --augment-funding
    python validate_walkforward.py --strategy-path strategy.py --json
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

TOTAL_DAYS   = 730
WINDOW_DAYS  = 91
N_WINDOWS    = 5
LAMBDA       = 0.5

TRAIN_WINDOWS = 3
TEST_WINDOWS  = 2


def load_strategy_module(strategy_path: Path):
    spec = importlib.util.spec_from_file_location("_strategy_wf", strategy_path)
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


def _build_windows(df: pd.DataFrame) -> list[dict]:
    """Build the same 5 windows as evaluate.py, returning metadata + sliced df."""
    total_days = (df.index[-1] - df.index[0]).days
    step_days = (total_days - WINDOW_DAYS) / max(N_WINDOWS - 1, 1)
    start_ts = df.index[0]

    windows = []
    for i in range(N_WINDOWS):
        w_start = start_ts + pd.Timedelta(days=round(i * step_days))
        w_end   = w_start  + pd.Timedelta(days=WINDOW_DAYS)
        w_df = df[(df.index >= w_start) & (df.index < w_end)].copy()
        windows.append({"idx": i + 1, "start": w_start, "end": w_end, "df": w_df})

    return windows


def _fitness(results: list[BacktestResult]) -> tuple[float, float, float]:
    sharpes = np.array([r.sharpe if r.is_valid() else 0.0 for r in results])
    mean_s = float(np.mean(sharpes))
    std_s  = float(np.std(sharpes))
    return mean_s - LAMBDA * std_s, mean_s, std_s


def run_walkforward(
    compute_signals_fn,
    params: dict,
    df: pd.DataFrame,
    init_cash: float = 10_000.0,
    fees: float = 0.001,
) -> dict:
    """
    Run walk-forward validation: train fitness on W1-W3, test fitness on W4-W5.
    """
    windows = _build_windows(df)

    all_results: list[BacktestResult] = []
    for w in windows:
        r = run_backtest(w["df"], params, compute_signals_fn,
                         init_cash=init_cash, fees=fees)
        all_results.append(r)

    train_results = all_results[:TRAIN_WINDOWS]
    test_results  = all_results[TRAIN_WINDOWS:]

    train_fit, train_mean, train_std = _fitness(train_results)
    test_fit, test_mean, test_std    = _fitness(test_results)
    full_fit, full_mean, full_std    = _fitness(all_results)

    def _window_dict(w: dict, r: BacktestResult) -> dict:
        return {
            "label": f"W{w['idx']}",
            "start": str(w["start"].date()),
            "end": str(w["end"].date()),
            "sharpe": r.sharpe,
            "return_pct": r.total_return_pct,
            "mdd_pct": r.max_drawdown_pct,
            "trades": r.n_trades,
            "win_rate": r.win_rate,
            "valid": r.is_valid(),
        }

    return {
        "train": {
            "fitness": train_fit,
            "mean_sharpe": train_mean,
            "std_sharpe": train_std,
            "windows": [_window_dict(windows[i], train_results[i])
                        for i in range(TRAIN_WINDOWS)],
        },
        "test": {
            "fitness": test_fit,
            "mean_sharpe": test_mean,
            "std_sharpe": test_std,
            "windows": [_window_dict(windows[TRAIN_WINDOWS + i], test_results[i])
                        for i in range(TEST_WINDOWS)],
        },
        "full": {
            "fitness": full_fit,
            "mean_sharpe": full_mean,
            "std_sharpe": full_std,
        },
    }


def _print_report(results: dict, params: dict, strategy_path: str) -> None:
    train = results["train"]
    test  = results["test"]
    full  = results["full"]

    print()
    print("=" * 90)
    print("  WALK-FORWARD VALIDATION REPORT")
    print(f"  Strategy: {strategy_path}")
    print(f"  Split: W1-W3 (in-sample) | W4-W5 (out-of-sample)")
    print("=" * 90)

    def _table(label: str, data: dict) -> None:
        print(f"\n  {label}")
        print(f"  Fitness: {data['fitness']:+.4f}  "
              f"(mean_sharpe={data['mean_sharpe']:+.3f}, "
              f"std={data['std_sharpe']:.3f}, lambda={LAMBDA})")
        if "windows" in data:
            print()
            print(f"    {'':>4}  {'Period':<25}  {'Sharpe':>7}  {'Return':>7}  "
                  f"{'MDD':>6}  {'Trades':>6}  {'WR':>5}  {'Valid':>5}")
            print(f"    {'---':>4}  {'---':<25}  {'---':>7}  {'---':>7}  "
                  f"{'---':>6}  {'---':>6}  {'---':>5}  {'---':>5}")
            for w in data["windows"]:
                v = "Y" if w["valid"] else "N"
                print(
                    f"    {w['label']:>4}  {w['start']} -> {w['end']:<10}  "
                    f"{w['sharpe']:>+7.3f}  {w['return_pct']:>+6.1f}%  "
                    f"{w['mdd_pct']:>5.1f}%  {w['trades']:>6}  "
                    f"{w['win_rate']*100:>4.0f}%  {v:>5}"
                )

    _table("IN-SAMPLE (W1-W3, optimized on these)", train)
    _table("OUT-OF-SAMPLE (W4-W5, unseen during optimization)", test)

    print()
    print("-" * 90)
    print(f"  Full 5-window fitness:     {full['fitness']:+.4f}")
    print(f"  In-sample fitness (W1-3):  {train['fitness']:+.4f}")
    print(f"  Out-of-sample (W4-5):      {test['fitness']:+.4f}")

    if train["fitness"] != 0:
        decay_pct = (train["fitness"] - test["fitness"]) / abs(train["fitness"]) * 100
        print(f"  Decay:                     {decay_pct:+.1f}%")

    print()
    if test["fitness"] > 0:
        print("  + Out-of-sample fitness is positive — strategy generalizes forward")
    else:
        print("  - Out-of-sample fitness is negative — possible overfitting")

    if test["fitness"] > 0.4714:
        print("  * OUT-OF-SAMPLE BEATS TA BASELINE (+0.4714)")

    print("=" * 90)
    print()


def main():
    p = argparse.ArgumentParser(
        description="Walk-forward validation: train on W1-W3, test on W4-W5",
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

    results = run_walkforward(compute_signals, params, df)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_report(results, params, str(strategy_path))


if __name__ == "__main__":
    main()
