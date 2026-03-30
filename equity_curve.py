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
from vectorbt.portfolio.enums import SizeType

from data import augment_with_timeframes, fetch_funding_rates, fetch_ohlcv, fetch_perp_ohlcv

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
    augment_timeframes: bool = False,
) -> pd.DataFrame:
    df = fetch_ohlcv(symbol, "1h", days=days)

    if augment_funding:
        perp_symbol = symbol.replace("/", "/") + ":USDT"
        funding = fetch_funding_rates(perp_symbol, days=days)
        df = df.join(funding, how="left")
        df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)

    if augment_basis:
        perp_symbol = symbol.replace("/", "/") + ":USDT"
        perp = fetch_perp_ohlcv(perp_symbol, timeframe="1h", days=days)
        perp_close = perp[["close"]].rename(columns={"close": "perp_close"})
        df = df.join(perp_close, how="left")
        df["perp_close"] = df["perp_close"].ffill().fillna(df["close"])
        df["basis"] = df["perp_close"] - df["close"]
        df["basis_pct"] = df["basis"] / df["close"]

    if augment_timeframes:
        df = augment_with_timeframes(df)

    return df


def run_full_backtest(
    compute_signals_fn,
    params: dict,
    df: pd.DataFrame,
    init_cash: float = 10_000.0,
    fees: float = 0.001,
    slippage: float = 0.0005,
) -> vbt.Portfolio:
    result = compute_signals_fn(df, params)

    size = None
    if isinstance(result, tuple) and len(result) == 3:
        entries, exits, size = result
    else:
        entries, exits = result

    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)

    total_fees = fees + slippage

    if size is not None:
        size_clean = size.fillna(1.0).clip(0.0, 1.0)
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
    return portfolio


def print_summary(portfolio: vbt.Portfolio, df: pd.DataFrame, strategy_name: str = "") -> None:
    stats = portfolio.stats()
    equity = portfolio.value()
    trades = portfolio.trades.records_readable

    btc_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    strat_return = float(stats.get("Total Return [%]", 0))
    total_days = (df.index[-1] - df.index[0]).days
    years_label = f"{total_days / 365:.0f}-YEAR"
    name_label = strategy_name or "Strategy"

    print()
    print("=" * 70)
    print(f"  FULL {years_label} BACKTEST — {name_label}")
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
    strategy_name: str = "",
) -> None:
    """Generate an HTML equity curve report."""
    equity = portfolio.value()
    returns = equity.pct_change().dropna()
    btc_returns = df["close"].pct_change().dropna()

    returns.index = returns.index.tz_localize(None)
    btc_returns.index = btc_returns.index.tz_localize(None)

    btc_returns = btc_returns.reindex(returns.index).fillna(0)

    total_days = (df.index[-1] - df.index[0]).days
    years_label = f"{total_days / 365:.0f}-Year"
    name_label = strategy_name or "Strategy"

    try:
        import quantstats as qs
        qs.reports.html(
            returns,
            benchmark=btc_returns,
            title=f"{name_label} — Full {years_label} Backtest",
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
        description="Full backtest with equity curve and HTML report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--strategy", type=str,
                   default="runs/synthesis_D_A/strategy.py",
                   help="Path to strategy.py")
    p.add_argument("--output", type=str, default="equity_report.html",
                   help="Output HTML report path")
    p.add_argument("--symbol", default="BTC/USDT",
                   help="Primary symbol")
    p.add_argument("--symbols", type=str, default=None,
                   help="Comma-separated symbols for multi-asset report (e.g. BTC/USDT,ETH/USDT)")
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
    strat_text = strategy_path.read_text().lower()
    auto_funding = "funding" in strat_text
    auto_basis = "basis" in strat_text
    auto_timeframes = "d1_" in strat_text or "h4_" in strat_text

    compute_signals_fn, params = load_strategy(strategy_path)
    strategy_name = strategy_path.parent.name.replace("_", " ").title()

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else [args.symbol]

    if len(symbols) == 1:
        # Single-asset mode (original behaviour)
        print(f"Loading {symbols[0]} data...", file=sys.stderr)
        df = load_data(
            symbol=symbols[0],
            days=args.days,
            augment_funding=args.augment_funding or auto_funding,
            augment_basis=args.augment_basis or auto_basis,
            augment_timeframes=auto_timeframes,
        )
        print(f"Data: {len(df)} bars [{df.index[0].date()} -> {df.index[-1].date()}]",
              file=sys.stderr)

        portfolio = run_full_backtest(compute_signals_fn, params, df, args.init_cash)
        print_summary(portfolio, df, strategy_name=strategy_name)

        output_path = Path(args.output)
        generate_html_report(portfolio, df, output_path, strategy_name=strategy_name)
        generate_trade_log(portfolio, output_path.with_suffix(".trades.csv"))
    else:
        # Multi-asset mode: run each symbol, combine into equal-weight portfolio
        per_asset_cash = args.init_cash / len(symbols)
        all_equity = {}
        all_bh = {}

        for sym in symbols:
            short = sym.split("/")[0]
            print(f"Loading {sym}...", file=sys.stderr)
            df = load_data(
                symbol=sym,
                days=args.days,
                augment_funding=args.augment_funding or auto_funding,
                augment_basis=args.augment_basis or auto_basis,
                augment_timeframes=auto_timeframes,
            )
            print(f"  {sym}: {len(df)} bars [{df.index[0].date()} -> {df.index[-1].date()}]",
                  file=sys.stderr)

            pf = run_full_backtest(compute_signals_fn, params, df, per_asset_cash)
            eq = pf.value()
            all_equity[short] = eq
            all_bh[short] = df["close"] / df["close"].iloc[0] * per_asset_cash

            # Per-asset summary
            stats = pf.stats()
            trades = pf.trades.records_readable
            n_trades = int(stats.get("Total Closed Trades", 0))
            ret = float(stats.get("Total Return [%]", 0))
            sharpe = float(stats.get("Sharpe Ratio", 0) or 0)
            mdd = abs(float(stats.get("Max Drawdown [%]", 0)))
            wr = 0
            if not trades.empty:
                wr = (trades["PnL"] > 0).sum() / len(trades) * 100
            print(f"  {short}: {ret:+.1f}% return, Sharpe {sharpe:.3f}, "
                  f"MDD {mdd:.1f}%, {n_trades} trades, {wr:.0f}% win rate")

        # Combine equity curves (equal-weight allocation)
        combined_eq = pd.DataFrame(all_equity)
        common_idx = combined_eq.dropna().index
        combined_eq = combined_eq.loc[common_idx]
        portfolio_equity = combined_eq.sum(axis=1)

        combined_bh = pd.DataFrame(all_bh)
        combined_bh = combined_bh.loc[common_idx]
        bh_equity = combined_bh.sum(axis=1)

        # Portfolio-level stats
        pf_returns = portfolio_equity.pct_change().dropna()
        bh_returns = bh_equity.pct_change().dropna()
        total_ret = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0] - 1) * 100
        bh_ret = (bh_equity.iloc[-1] / bh_equity.iloc[0] - 1) * 100

        total_days = (common_idx[-1] - common_idx[0]).days
        years_label = f"{total_days / 365:.0f}-YEAR"
        sym_label = " + ".join(s.split("/")[0] for s in symbols)

        print()
        print("=" * 70)
        print(f"  FULL {years_label} BACKTEST — {strategy_name} ({sym_label})")
        print(f"  Period: {common_idx[0].date()} to {common_idx[-1].date()}")
        print(f"  Starting capital: ${args.init_cash:,.0f} "
              f"(${per_asset_cash:,.0f} per asset)")
        print("=" * 70)
        print(f"\n  Portfolio return:   {total_ret:+.1f}%")
        print(f"  Buy-and-hold:       {bh_ret:+.1f}%")
        print(f"  Final equity:       ${portfolio_equity.iloc[-1]:,.0f}")

        dd = (portfolio_equity / portfolio_equity.cummax() - 1)
        print(f"\n  Max Drawdown:       {abs(dd.min()) * 100:.1f}%")

        # Per-year breakdown
        print()
        for year in range(common_idx[0].year, common_idx[-1].year + 1):
            mask = portfolio_equity.index.year == year
            if mask.sum() < 100:
                continue
            yr_eq = portfolio_equity[mask]
            yr_ret = (yr_eq.iloc[-1] / yr_eq.iloc[0] - 1) * 100
            bh_yr = bh_equity[mask]
            bh_yr_ret = (bh_yr.iloc[-1] / bh_yr.iloc[0] - 1) * 100
            print(f"  {year}: portfolio {yr_ret:+.1f}%  |  buy-and-hold {bh_yr_ret:+.1f}%")
        print("=" * 70)

        # Generate quantstats report with combined portfolio
        pf_returns.index = pf_returns.index.tz_localize(None)
        bh_returns.index = bh_returns.index.tz_localize(None)
        bh_returns = bh_returns.reindex(pf_returns.index).fillna(0)

        output_path = Path(args.output)
        try:
            import quantstats as qs
            qs.reports.html(
                pf_returns,
                benchmark=bh_returns,
                title=f"{strategy_name} ({sym_label}) — Full {years_label} Backtest",
                output=str(output_path),
            )
            print(f"\n  HTML report: {output_path}")
        except Exception as e:
            log.warning("quantstats report failed: %s", e)

        # Trade logs per asset
        for sym in symbols:
            short = sym.split("/")[0]
            df_sym = load_data(
                symbol=sym, days=args.days,
                augment_funding=auto_funding,
                augment_timeframes=auto_timeframes,
            )
            pf_sym = run_full_backtest(compute_signals_fn, params, df_sym, per_asset_cash)
            trades_sym = pf_sym.trades.records_readable
            if not trades_sym.empty:
                tlog = output_path.with_suffix(f".{short.lower()}_trades.csv")
                trades_sym.to_csv(tlog, index=False)
                print(f"  Trade log ({short}): {tlog}")


if __name__ == "__main__":
    main()
