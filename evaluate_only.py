"""
evaluate_only.py — run evaluation on any strategy and print JSON result.

Used when Claude acts as the agent directly, avoiding external API calls.

Usage:
    python evaluate_only.py
    python evaluate_only.py --strategy-path runs/track_A_statistical_vol_regime/strategy.py
    python evaluate_only.py --strategy-path runs/track_C_cross_pair/strategy.py --augment-eth
    python evaluate_only.py --symbol ETH/USDT --lambda-penalty 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

import importlib.util

from evaluate import evaluate_strategy
from data import fetch_ohlcv

_ROOT = Path(__file__).parent


def load_strategy_module(strategy_path: Path):
    spec = importlib.util.spec_from_file_location("_strategy_live", strategy_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_signals, mod.PARAMS


def augment_with_eth(df):
    """Fetch ETH/USDT 1h and merge as eth_* columns aligned to BTC index."""
    eth = fetch_ohlcv("ETH/USDT", "1h", days=730)
    eth = eth.rename(columns={c: f"eth_{c}" for c in eth.columns})
    merged = df.join(eth, how="left")
    merged[list(eth.columns)] = merged[list(eth.columns)].ffill()
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--lambda-penalty", type=float, default=0.5)
    p.add_argument("--strategy-path", type=str, default=None,
                   help="Path to strategy.py (default: root strategy.py)")
    p.add_argument("--augment-eth", action="store_true",
                   help="Merge ETH/USDT data for Track C cross-pair signals")
    args = p.parse_args()

    strategy_path = Path(args.strategy_path) if args.strategy_path else _ROOT / "strategy.py"

    df = fetch_ohlcv(args.symbol, "1h", days=730)
    if args.augment_eth:
        df = augment_with_eth(df)

    compute_signals, params = load_strategy_module(strategy_path)

    result = evaluate_strategy(
        compute_signals, params,
        symbol=args.symbol,
        lambda_penalty=args.lambda_penalty,
        df=df,
    )

    output = result.to_dict()
    output["fitness"] = result.fitness
    output["summary"] = result.summary()
    output["current_params"] = params
    output["strategy_source"] = strategy_path.read_text()

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
