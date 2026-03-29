"""
track_runner.py — V2 multi-track research runner.

Creates an isolated run directory for each research track and executes the
strategy research loop within it, keeping every track's strategy.py and
experiment_log.jsonl fully independent.

Directory layout produced:
    runs/
    └── track_A_statistical_vol_regime/
        ├── strategy.py          ← track's evolving strategy (agent edits this)
        ├── strategy.py.bak      ← last-known-good backup
        └── experiment_log.jsonl ← track's experiment record

Usage:
    # Run 20 iterations of Track A (vol regime)
    python track_runner.py --track tracks/track_a_vol_regime.json --iterations 20

    # Run 20 iterations with verbose logging
    python track_runner.py --track tracks/track_b_calendar.json --iterations 20 -v

    # Reset track state and start fresh (deletes existing run dir)
    python track_runner.py --track tracks/track_a_vol_regime.json --iterations 20 --reset

    # Use a different model or lambda
    python track_runner.py --track tracks/track_c_cross_pair.json -n 20 --model claude-sonnet-4-5

Required:
    ANTHROPIC_API_KEY must be set in the environment or in .env

Notes:
    - Track C expects eth_* columns in df; the runner fetches ETH/USDT and merges them.
    - Track D expects a funding_rate column; this pipeline is not yet implemented
      (raises NotImplementedError with instructions).
    - Tracks A, B, E require no extra data beyond BTC/USDT OHLCV.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

# Auto-load .env
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from data import fetch_funding_rates, fetch_ohlcv
from research_loop import run_loop
from track_config import TrackConfig, load_track

log = logging.getLogger(__name__)

_ROOT     = Path(__file__).parent
_RUNS_DIR = _ROOT / "runs"


# ── Initial strategy templates ────────────────────────────────────────────────

# Used for Tracks A, B, C, D — a minimal pandas-only momentum strategy that:
#   - uses only standard OHLCV columns (works for all tracks)
#   - generates a few trades per window (agent has something to measure)
#   - imposes no TA-specific structure (agent rewrites freely)
_INITIAL_STRATEGY_GENERIC = '''\
"""
strategy.py — initial strategy for Track {track_id} ({description}).

This is a minimal starting point. The agent will replace it.
Signal class: {signal_class}
"""

from __future__ import annotations

import pandas as pd

PARAMS: dict = {{
    "lookback":      20,
    "exit_lookback": 10,
}}


def compute_signals(df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    """
    Minimal price-momentum starting strategy.

    Entry:  close > highest close of the past lookback bars (breakout)
    Exit:   close < lowest close of the past exit_lookback bars (breakdown)

    The agent will replace this logic entirely.
    """
    close = df["close"]

    recent_high = close.shift(1).rolling(int(params["lookback"])).max()
    recent_low  = close.shift(1).rolling(int(params["exit_lookback"])).min()

    entries = (close > recent_high).fillna(False)
    exits   = (close < recent_low).fillna(False)

    return entries, exits
'''

# Track E starts from the current V1 strategy (copy it in, don't overwrite)
_INITIAL_STRATEGY_E_SOURCE = _ROOT / "strategy.py"


# ── Data augmentation ─────────────────────────────────────────────────────────

def _augment_with_eth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch ETH/USDT 1h OHLCV and merge as eth_* columns aligned to BTC index.
    Used for Track C (cross-pair signals).
    """
    log.info("Fetching ETH/USDT 1h data for Track C cross-pair signals…")
    eth = fetch_ohlcv("ETH/USDT", "1h", days=730)
    eth = eth.rename(columns={c: f"eth_{c}" for c in eth.columns})

    merged = df.join(eth, how="left")
    merged[list(eth.columns)] = merged[list(eth.columns)].ffill()

    n_missing = merged["eth_close"].isna().sum()
    if n_missing > 0:
        log.warning("ETH data has %d missing bars after merge — filling with NaN", n_missing)

    log.info("ETH columns added: %s", list(eth.columns))
    return merged


def _augment_with_funding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch BTC perpetual funding rates and merge as a forward-filled
    funding_rate column aligned to the 1h OHLCV index.

    Funding rates are published every 8h on Binance perpetual futures.
    We forward-fill to 1h resolution so each candle carries the most
    recent funding rate — no look-ahead bias.
    """
    log.info("Fetching BTC/USDT:USDT perpetual funding rates for Track D…")
    funding = fetch_funding_rates("BTC/USDT:USDT", days=730)
    log.info("Raw funding data: %d entries [%s → %s]",
             len(funding), funding.index.min(), funding.index.max())

    merged = df.join(funding, how="left")

    merged["funding_rate"] = merged["funding_rate"].ffill()

    n_missing = merged["funding_rate"].isna().sum()
    if n_missing > 0:
        log.warning(
            "funding_rate has %d NaN rows after forward-fill "
            "(early bars before first funding entry) — filling with 0.0",
            n_missing,
        )
        merged["funding_rate"] = merged["funding_rate"].fillna(0.0)

    log.info(
        "Funding rate merged: min=%.6f  max=%.6f  mean=%.6f",
        merged["funding_rate"].min(),
        merged["funding_rate"].max(),
        merged["funding_rate"].mean(),
    )
    return merged


def _get_augment_fn(config: TrackConfig):
    """Return the df augmentation function for a given track, or None."""
    if config.track_id == "C":
        return _augment_with_eth
    if config.track_id == "D":
        return _augment_with_funding
    return None


# ── Run directory management ──────────────────────────────────────────────────

def _run_dir(config: TrackConfig) -> Path:
    """Return the canonical run directory path for a track."""
    return _RUNS_DIR / f"track_{config.track_id}_{config.signal_class}"


def _init_run_dir(config: TrackConfig, reset: bool = False) -> Path:
    """
    Ensure the track's run directory exists and contains a valid strategy.py.

    If reset=True, deletes the existing run directory first (fresh start).
    If strategy.py already exists (resuming), it is left untouched.

    Returns the run directory Path.
    """
    run_dir = _run_dir(config)

    if reset and run_dir.exists():
        log.info("--reset: removing existing run directory %s", run_dir)
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    strategy_file = run_dir / "strategy.py"
    if not strategy_file.exists():
        if config.track_id == "E":
            # Track E inherits the current V1 strategy
            shutil.copy(_INITIAL_STRATEGY_E_SOURCE, strategy_file)
            log.info("Track E: copied V1 strategy.py → %s", strategy_file)
        else:
            # All other tracks start from the generic momentum template
            content = _INITIAL_STRATEGY_GENERIC.format(
                track_id=config.track_id,
                description=config.description,
                signal_class=config.signal_class,
            )
            strategy_file.write_text(content)
            log.info(
                "Track %s: wrote initial strategy template → %s",
                config.track_id, strategy_file,
            )
    else:
        log.info("Resuming: existing strategy.py found at %s", strategy_file)

    return run_dir


# ── Main runner ───────────────────────────────────────────────────────────────

def run_track(
    config: TrackConfig,
    n_iterations: int,
    reset: bool = False,
    symbol: str = "BTC/USDT",
    lambda_penalty: float = 0.5,
    model: str = "claude-opus-4-6",
    history_window: int = 15,
    target_fitness: float | None = None,
    plateau_window: int = 5,
    plateau_threshold: float = 0.005,
    agent_mode: str = "api",
) -> None:
    """
    Run N research iterations for a single V2 track in an isolated directory.

    Args:
        config:            Track configuration loaded from a JSON file.
        n_iterations:      Number of research iterations to run.
        reset:             If True, delete and recreate the run directory first.
        symbol:            Market symbol for evaluation.
        lambda_penalty:    Fitness variance penalty λ.
        model:             Anthropic model to use.
        history_window:    How many past experiments to pass to the agent.
        target_fitness:    Stop early when fitness reaches this value.
        plateau_window:    Number of recent accepted iterations to check for plateau.
        plateau_threshold: Minimum Δ fitness per accepted iteration to avoid plateau.
        agent_mode:        "api" or "local" — passed through to run_loop().
    """
    run_dir = _init_run_dir(config, reset=reset)

    strategy_path = run_dir / "strategy.py"
    backup_path   = run_dir / "strategy.py.bak"
    log_path      = run_dir / "experiment_log.jsonl"
    df_augment_fn = _get_augment_fn(config)

    log.info(
        "Track %s (%s) — %d iterations [agent_mode=%s]",
        config.track_id, config.signal_class, n_iterations, agent_mode,
    )
    log.info("Run dir: %s", run_dir)

    run_loop(
        n_iterations=n_iterations,
        symbol=symbol,
        lambda_penalty=lambda_penalty,
        model=model,
        history_window=history_window,
        target_fitness=target_fitness,
        plateau_window=plateau_window,
        plateau_threshold=plateau_threshold,
        track_config=config,
        strategy_path=strategy_path,
        backup_path=backup_path,
        log_path=log_path,
        df_augment_fn=df_augment_fn,
        agent_mode=agent_mode,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="V2 track runner — runs a research track in an isolated directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--track", "-t", type=str, required=True,
        metavar="PATH",
        help="Path to track JSON config file (e.g. tracks/track_a_vol_regime.json)",
    )
    p.add_argument(
        "--iterations", "-n", type=int, default=20,
        help="Number of research iterations to run",
    )
    p.add_argument(
        "--reset", action="store_true",
        help="Delete and recreate the run directory (fresh start, discards history)",
    )
    p.add_argument(
        "--symbol", "-s", type=str, default="BTC/USDT",
        help="Market symbol for evaluation",
    )
    p.add_argument(
        "--lambda-penalty", type=float, default=0.5,
        help="Variance penalty λ in the fitness function",
    )
    p.add_argument(
        "--model", type=str, default="claude-opus-4-6",
        help="Anthropic model string for the LLM agent",
    )
    p.add_argument(
        "--history", type=int, default=15,
        help="Number of past experiments to include in agent context",
    )
    p.add_argument(
        "--target-fitness", type=float, default=None,
        help="Stop as soon as fitness reaches this value",
    )
    p.add_argument(
        "--plateau-window", type=int, default=5,
        help="Stop if the last N accepted iterations all improved by less than "
             "--plateau-threshold",
    )
    p.add_argument(
        "--plateau-threshold", type=float, default=0.005,
        help="Minimum meaningful Δ fitness per accepted iteration",
    )
    p.add_argument(
        "--agent-mode", type=str, default="api",
        choices=["api", "local"],
        help="'api' calls the Anthropic API; 'local' uses file-based "
             "request/response for an external agent (e.g. Claude in Cursor)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    config = load_track(args.track)

    run_track(
        config=config,
        n_iterations=args.iterations,
        reset=args.reset,
        symbol=args.symbol,
        lambda_penalty=args.lambda_penalty,
        model=args.model,
        history_window=args.history,
        target_fitness=args.target_fitness,
        plateau_window=args.plateau_window,
        plateau_threshold=args.plateau_threshold,
        agent_mode=args.agent_mode,
    )


if __name__ == "__main__":
    main()
