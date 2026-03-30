"""
equity_curve.py — Full 2-year backtest with equity curve and trade analysis.

Runs the D+A ensemble strategy over the entire 730-day dataset as a single
continuous backtest (not windowed), producing:
  - Equity curve (portfolio value over time)
  - Trade log with entry/exit prices and PnL
  - Drawdown series
  - BTC buy-and-hold comparison
  - HTML report via quantstats

Usage:
    python equity_curve.py
    python equity_curve.py --strategy runs/track_D_funding_rates/strategy.py --augment-funding
    python equity_curve.py --output equity_report.html
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

from data import fetch_funding_rates, fetch_ohlcv, fetch_perp_ohlcv

log = logging.getLogger(__name__)


def load_strategy(path: Path):
    spec = importlib.util.spec_from_file_location("_eq_strategy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_signals, mod.PARAMS


def load_data(
    symbol: str = "BTC/USDT",
    days: int = 730,
    augment_funding: bool = False,
    augment_basis: bool = False,
) -> pd.DataFrame:
    df = fetch_ohlcv(symbol, "1h", days=days)

    if augment_funding:
        funding = fetch_funding_rates("BTC/USDT:USDT", days=days)
        df = df.join(funding, how="left")
        df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)

    if augment_basis:
        perp = fetch_perp_ohlcv("BTC/USDT:USDT", timeframe="1h", days=days)
        perp_close = perp[["close"]].rename(columns={"close": "perp_close"})
        df = df.join(perp_close, how="left")
        df["perp_close"] = df["perp_close"].ffill().fillna(df["close"])
        df["basis"] = df["perp_close"] - df["close"]
        df["basis_pct"] = df["basis"] / df["close"]

    return df


def run_full_backtest(
    compute_signals_fn,
    params: dict,
    df: pd.DataFrame,
    init_cash: float = 10_000.0,
    fees: float = 0.001,
) -> vbt.Portfolio:
    entries, exits = compute_signals_fn(df, params)
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)

    portfolio = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=fees,
        freq="1h",
    )
    return portfolio


def print_summary(portfolio: vbt.Portfolio, df: pd.DataFrame) -> None:
    stats = portfolio.stats()
    equity = portfolio.value()
    trades = portfolio.trades.records_readable

    btc_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    strat_return = float(stats.get("Total Return [%]", 0))

    print()
    print("=" * 70)
    print("  FULL 2-YEAR BACKTEST — D+A Ensemble Strategy")
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Starting capital: ${equity.iloc[0]:,.0f}")
    print("=" * 70)

    print(f"\n  Strategy return:    {strat_return:+.1f}%")
    print(f"  BTC buy-and-hold:   {btc_return:+.1f}%")
    print(f"  Final equity:       ${equity.iloc[-1]:,.0f}")

    print(f"\n  Sharpe Ratio:       {stats.get('Sharpe Ratio', 0):.3f}")
    print(f"  Sortino Ratio:      {stats.get('Sortino Ratio', 0):.3f}")
    print(f"  Max Drawdown:       {abs(float(stats.get('Max Drawdown [%]', 0))):.1f}%")
    print(f"  Max DD Duration:    {stats.get('Max Drawdown Duration', 'N/A')}")

    print(f"\n  Total Trades:       {int(stats.get('Total Closed Trades', 0))}")
    if not trades.empty:
        wins = (trades["PnL"] > 0).sum()
        total = len(trades)
        print(f"  Win Rate:           {wins}/{total} ({wins/total*100:.0f}%)")
        print(f"  Avg Win:            ${trades[trades['PnL'] > 0]['PnL'].mean():+,.0f}")
        print(f"  Avg Loss:           ${trades[trades['PnL'] <= 0]['PnL'].mean():+,.0f}")
        print(f"  Best Trade:         ${trades['PnL'].max():+,.0f}")
        print(f"  Worst Trade:        ${trades['PnL'].min():+,.0f}")
        duration_col = "Duration" if "Duration" in trades.columns else None
        if duration_col:
            print(f"  Avg Hold Duration:  {trades[duration_col].mean()}")

    print()
    print("=" * 70)


def generate_html_report(
    portfolio: vbt.Portfolio,
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate an HTML equity curve report."""
    equity = portfolio.value()
    returns = equity.pct_change().dropna()
    btc_returns = df["close"].pct_change().dropna()

    returns.index = returns.index.tz_localize(None)
    btc_returns.index = btc_returns.index.tz_localize(None)

    btc_returns = btc_returns.reindex(returns.index).fillna(0)

    try:
        import quantstats as qs
        qs.reports.html(
            returns,
            benchmark=btc_returns,
            title="D+A Ensemble Strategy — Full 2-Year Backtest",
            output=str(output_path),
        )
        print(f"\n  HTML report: {output_path}")
    except Exception as e:
        log.warning("quantstats report failed: %s — falling back to CSV", e)
        csv_path = output_path.with_suffix(".csv")
        equity_df = pd.DataFrame({
            "equity": equity,
            "btc_close": df["close"],
        })
        equity_df.to_csv(csv_path)
        print(f"\n  Equity CSV: {csv_path}")


def generate_trade_log(portfolio: vbt.Portfolio, output_path: Path) -> None:
    trades = portfolio.trades.records_readable
    if trades.empty:
        print("  No trades to log.")
        return
    trades.to_csv(output_path, index=False)
    print(f"  Trade log:   {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Full 2-year backtest with equity curve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--strategy", type=str,
                   default="runs/synthesis_D_A/strategy.py",
                   help="Path to strategy.py")
    p.add_argument("--output", type=str, default="equity_report.html",
                   help="Output HTML report path")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--init-cash", type=float, default=10_000)
    p.add_argument("--augment-funding", action="store_true")
    p.add_argument("--augment-basis", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    strategy_path = Path(args.strategy)
    auto_funding = "funding" in strategy_path.read_text().lower()
    auto_basis = "basis" in strategy_path.read_text().lower()

    compute_signals_fn, params = load_strategy(strategy_path)

    print("Loading data...", file=sys.stderr)
    df = load_data(
        symbol=args.symbol,
        days=args.days,
        augment_funding=args.augment_funding or auto_funding,
        augment_basis=args.augment_basis or auto_basis,
    )
    print(f"Data: {len(df)} bars [{df.index[0].date()} -> {df.index[-1].date()}]",
          file=sys.stderr)

    portfolio = run_full_backtest(compute_signals_fn, params, df, args.init_cash)

    print_summary(portfolio, df)

    output_path = Path(args.output)
    generate_html_report(portfolio, df, output_path)
    generate_trade_log(portfolio, output_path.with_suffix(".trades.csv"))


if __name__ == "__main__":
    main()
