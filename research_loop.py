"""
research_loop.py — autonomous strategy discovery orchestrator.

Implements the self-improving research loop from the project brief:

  1. Pre-fetch 2 years of market data (cached locally)
  2. Evaluate the current strategy.py → baseline fitness
  3. Load recent experiment history
  4. Call the LLM agent → StrategySpec (a proposed modification)
  5. Apply the modification to strategy.py
  6. Re-evaluate → new fitness
  7. Accept if improved, revert to backup if not
  8. Append full record to experiment_log.jsonl
  9. Repeat until a stopping condition is met

Stopping conditions (whichever fires first):
  --iterations N         Hard budget: stop after N iterations (default 10)
  --target-fitness F     Stop as soon as fitness ≥ F (default: disabled)
  --plateau-window W     Stop if the last W *accepted* iterations all improved
                         fitness by less than --plateau-threshold (default 5)
  --plateau-threshold T  Minimum meaningful improvement per accepted iteration
                         (default 0.005)

Usage:
    python research_loop.py                        # 10 iterations on BTC/USDT
    python research_loop.py --iterations 20
    python research_loop.py --iterations 100 --target-fitness 0.7
    python research_loop.py --plateau-window 5 --plateau-threshold 0.01
    python research_loop.py --symbol ETH/USDT --iterations 5
    python research_loop.py --verbose              # DEBUG logging

V2: run_loop() accepts optional track_config, path overrides, and df_augment_fn
    so that track_runner.py can drive isolated per-track research sessions without
    touching the root strategy.py or experiment_log.jsonl.

Required:
    ANTHROPIC_API_KEY environment variable must be set (for --agent-mode api).
    Not required for --agent-mode local.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Auto-load .env from the project directory so ANTHROPIC_API_KEY is always set
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from agent import StrategySpec, propose_modification, propose_modification_local
from data import fetch_ohlcv
from evaluate import FitnessResult, evaluate_strategy
from track_config import TrackConfig

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT          = Path(__file__).parent
STRATEGY_PATH  = _ROOT / "strategy.py"
BACKUP_PATH    = _ROOT / "strategy.py.bak"
LOG_PATH       = _ROOT / "experiment_log.jsonl"


# ── Strategy file utilities ───────────────────────────────────────────────────

def load_strategy_module(strategy_path: Path | None = None):
    """
    Import strategy.py as a fresh module (bypasses sys.modules cache).

    Args:
        strategy_path: Path to strategy.py. Defaults to the root STRATEGY_PATH.

    Returns:
        (compute_signals, PARAMS) — the strategy function and its parameter dict.
    """
    path = strategy_path or STRATEGY_PATH
    spec = importlib.util.spec_from_file_location("_strategy_live", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_signals, mod.PARAMS


def apply_spec_to_source(spec: StrategySpec, current_source: str) -> str:
    """
    Apply a StrategySpec to strategy.py source, returning the modified source.

    Always replaces the PARAMS dict. For structural changes, also replaces the
    compute_signals function body (everything from 'def compute_signals(' onward).

    Args:
        spec:           Proposed modification from the agent.
        current_source: Current content of strategy.py as a string.

    Returns:
        New strategy.py source string.

    Raises:
        ValueError: if expected markers are not found in the source.
    """
    source = _replace_params(current_source, spec.params)

    if spec.change_type == "structural" and spec.new_signals_code:
        source = _replace_compute_signals(source, spec.new_signals_code)

    return source


def _replace_params(source: str, new_params: dict[str, float]) -> str:
    """Replace the PARAMS dict block in source with new_params values."""
    lines = source.split("\n")

    # Locate the PARAMS dict start line
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^PARAMS\s*:\s*dict\s*=\s*\{", line):
            start = i
            break
    if start is None:
        raise ValueError("Could not find 'PARAMS: dict = {' in strategy.py")

    # Find the matching closing brace (brace-depth tracking)
    depth = 0
    end = start
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0:
            end = i
            break

    # Build the new PARAMS block
    block = ["PARAMS: dict = {"]
    for key, val in new_params.items():
        # Render whole numbers as ints for readability
        rendered = int(val) if isinstance(val, float) and val.is_integer() else val
        block.append(f'    "{key}": {rendered},')
    block.append("}")

    return "\n".join(lines[:start] + block + lines[end + 1 :])


def _replace_compute_signals(source: str, new_fn_source: str) -> str:
    """Replace the compute_signals function with new_fn_source."""
    lines = source.split("\n")

    # Locate the function definition
    fn_start = None
    for i, line in enumerate(lines):
        if re.match(r"^def compute_signals\s*\(", line):
            fn_start = i
            break
    if fn_start is None:
        raise ValueError("Could not find 'def compute_signals(' in strategy.py")

    # Keep module header (everything before the function definition)
    header = lines[:fn_start]

    fn_lines = new_fn_source.strip().split("\n")

    return "\n".join(header + fn_lines + [""])


# ── Experiment log ────────────────────────────────────────────────────────────

def load_history(n: int = 20, log_path: Path | None = None) -> list[dict]:
    """Load the last n entries from experiment_log.jsonl."""
    path = log_path or LOG_PATH
    if not path.exists():
        return []
    entries: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("Skipping malformed log line: %s", line[:80])
    return entries[-n:]


def append_log(entry: dict, log_path: Path | None = None) -> None:
    """Append one experiment record to experiment_log.jsonl."""
    path = log_path or LOG_PATH
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _next_iteration_number(log_path: Path | None = None) -> int:
    """Return the iteration number for the next experiment (1-indexed, no gaps)."""
    history = load_history(n=999_999, log_path=log_path)
    if not history:
        return 1
    return max(e.get("iteration", 0) for e in history) + 1


def _build_log_entry(
    iteration: int,
    accepted: bool,
    fitness_before: float,
    fitness_after: float,
    spec: StrategySpec,
    fitness_result: FitnessResult,
    note: str = "",
    track_config: TrackConfig | None = None,
) -> dict:
    entry: dict = {
        "iteration":      iteration,
        "timestamp":      datetime.now(tz=timezone.utc).isoformat(),
        "accepted":       accepted,
        "fitness_before": round(fitness_before, 6),
        "fitness_after":  round(fitness_after, 6),
        "delta":          round(fitness_after - fitness_before, 6),
        "change_type":    spec.change_type,
        "rationale":      spec.rationale,
        "params":         spec.params,
        "has_new_code":   spec.new_signals_code is not None,
        "fitness_result": fitness_result.to_dict(),
    }
    if track_config is not None:
        entry["track_id"] = track_config.track_id
        entry["signal_class"] = track_config.signal_class
    if note:
        entry["note"] = note
    return entry


# ── Research loop ─────────────────────────────────────────────────────────────

def run_loop(
    n_iterations: int,
    symbol: str = "BTC/USDT",
    lambda_penalty: float = 0.5,
    model: str = "claude-opus-4-6",
    history_window: int = 15,
    target_fitness: float | None = None,
    plateau_window: int = 5,
    plateau_threshold: float = 0.005,
    # V2 additions — all optional, default to V1 behaviour
    track_config: TrackConfig | None = None,
    strategy_path: Path | None = None,
    backup_path: Path | None = None,
    log_path: Path | None = None,
    df_augment_fn=None,
    agent_mode: str = "api",
) -> None:
    """
    Run the autonomous strategy research loop.

    Args:
        n_iterations:      Hard budget — stop after this many iterations.
        symbol:            Market symbol for evaluation (e.g. "BTC/USDT").
        lambda_penalty:    λ in fitness = mean(sharpe) − λ × std(sharpe).
        model:             Anthropic model for the LLM agent.
        history_window:    How many past experiments to include in agent context.
        target_fitness:    Stop early if fitness reaches this value (None = disabled).
        plateau_window:    Plateau window: number of recent *accepted* iterations
                           to inspect for meaningful improvement.
        plateau_threshold: Minimum Δ fitness per accepted iteration to count as
                           meaningful progress. If all recent accepted iterations
                           are below this, the loop stops.
        track_config:      V2 TrackConfig — parameterises agent system prompt per
                           signal class. None = V1 TA baseline behaviour.
        strategy_path:     Path to strategy.py for this run. Defaults to root
                           STRATEGY_PATH (for V1 backward compatibility).
        backup_path:       Path for strategy backup. Defaults to root BACKUP_PATH.
        log_path:          Path to experiment_log.jsonl. Defaults to root LOG_PATH.
        df_augment_fn:     Optional callable(df) -> df that merges extra columns
                           into the OHLCV dataframe before evaluation (e.g. ETH
                           data for Track C, funding rates for Track D).
        agent_mode:        "api" (default) calls the Anthropic API; "local" uses
                           file-based request/response for an external agent.
    """
    _strategy_path = strategy_path or STRATEGY_PATH
    _backup_path   = backup_path   or BACKUP_PATH
    _log_path      = log_path      or LOG_PATH

    track_label = (
        f"Track {track_config.track_id} — {track_config.description}"
        if track_config else "V1 (TA baseline)"
    )
    _print_banner(f"Research loop — {n_iterations} iterations on {symbol}  [{track_label}]")
    log.info("Agent mode: %s", agent_mode)
    if agent_mode == "api":
        log.info("Model: %s   λ=%.2f   history_window=%d", model, lambda_penalty, history_window)
    else:
        log.info("λ=%.2f   history_window=%d", lambda_penalty, history_window)
    log.info("Strategy : %s", _strategy_path)
    log.info("Log      : %s", _log_path)
    if target_fitness is not None:
        log.info("Target fitness : %+.4f", target_fitness)
    log.info("Plateau stop   : window=%d  threshold=%.4f", plateau_window, plateau_threshold)

    # ── pre-fetch data ────────────────────────────────────────────────────────
    log.info("\nPre-fetching 2 years of %s 1h candles…", symbol)
    df = fetch_ohlcv(symbol, "1h", days=730)
    log.info(
        "Data: %d rows  [%s → %s]",
        len(df), df.index[0].date(), df.index[-1].date(),
    )

    if df_augment_fn is not None:
        log.info("Augmenting df with extra columns for %s…", track_label)
        df = df_augment_fn(df)
        log.info("df columns after augment: %s", list(df.columns))

    # ── baseline evaluation ───────────────────────────────────────────────────
    log.info("\n── Baseline evaluation ─────────────────────────────────────────")
    compute_signals, params = load_strategy_module(strategy_path=_strategy_path)
    baseline = evaluate_strategy(
        compute_signals, params,
        symbol=symbol, lambda_penalty=lambda_penalty, df=df,
    )
    print(baseline.summary())
    log.info("Baseline fitness: %+.4f\n", baseline.fitness)

    current_fitness = baseline.fitness
    start_iter = _next_iteration_number(log_path=_log_path)

    # ── iteration loop ────────────────────────────────────────────────────────
    for i in range(start_iter, start_iter + n_iterations):
        _print_banner(f"Iteration {i}  (fitness so far: {current_fitness:+.4f})", width=50)

        history = load_history(n=history_window, log_path=_log_path)
        strategy_source = _strategy_path.read_text()

        # ── re-evaluate current strategy (for per-window agent context) ───────
        try:
            compute_signals, params = load_strategy_module(strategy_path=_strategy_path)
            current_result = evaluate_strategy(
                compute_signals, params,
                symbol=symbol, lambda_penalty=lambda_penalty, df=df,
            )
            current_fitness = current_result.fitness
        except Exception as exc:
            log.error("Could not evaluate current strategy: %s — skipping iteration", exc)
            continue

        # ── agent proposal ────────────────────────────────────────────────────
        try:
            if agent_mode == "local":
                run_dir = _strategy_path.parent
                spec = propose_modification_local(
                    strategy_source=strategy_source,
                    current_fitness=current_fitness,
                    fitness_summary=current_result.summary(),
                    experiment_history=history,
                    run_dir=run_dir,
                    track_config=track_config,
                )
            else:
                spec = propose_modification(
                    strategy_source=strategy_source,
                    current_fitness=current_fitness,
                    fitness_summary=current_result.summary(),
                    experiment_history=history,
                    model=model,
                    track_config=track_config,
                )
        except Exception as exc:
            log.error("Agent error in iteration %d: %s — skipping", i, exc)
            continue

        log.info("[%s] %s", spec.change_type.upper(), spec.rationale)

        # ── apply modification ────────────────────────────────────────────────
        shutil.copy(_strategy_path, _backup_path)

        try:
            new_source = apply_spec_to_source(spec, strategy_source)
        except Exception as exc:
            log.error("Failed to apply StrategySpec: %s — skipping", exc)
            shutil.copy(_backup_path, _strategy_path)
            continue

        _strategy_path.write_text(new_source)

        # ── evaluate modified strategy ────────────────────────────────────────
        try:
            new_signals, new_params = load_strategy_module(strategy_path=_strategy_path)
            new_result = evaluate_strategy(
                new_signals, new_params,
                symbol=symbol, lambda_penalty=lambda_penalty, df=df,
            )
            new_fitness = new_result.fitness
        except Exception as exc:
            log.error("Evaluation failed after modification: %s — reverting", exc)
            shutil.copy(_backup_path, _strategy_path)
            append_log(
                _build_log_entry(
                    iteration=i, accepted=False,
                    fitness_before=current_fitness, fitness_after=current_fitness,
                    spec=spec, fitness_result=current_result,
                    note=f"evaluation_error: {exc}",
                    track_config=track_config,
                ),
                log_path=_log_path,
            )
            continue

        # ── accept or revert ──────────────────────────────────────────────────
        fitness_before = current_fitness
        accepted = new_fitness > current_fitness
        delta = new_fitness - fitness_before

        if accepted:
            current_fitness = new_fitness
            log.info(
                "✓ ACCEPTED   %+.4f → %+.4f  (Δ=%+.4f)",
                fitness_before, new_fitness, delta,
            )
            print(new_result.summary())
        else:
            shutil.copy(_backup_path, _strategy_path)
            log.info(
                "✗ REJECTED   %+.4f → %+.4f  (Δ=%+.4f) — reverted",
                fitness_before, new_fitness, delta,
            )

        append_log(
            _build_log_entry(
                iteration=i, accepted=accepted,
                fitness_before=fitness_before, fitness_after=new_fitness,
                spec=spec, fitness_result=new_result,
                track_config=track_config,
            ),
            log_path=_log_path,
        )

        # ── stopping conditions ───────────────────────────────────────────────

        # 1. Target fitness reached
        if target_fitness is not None and current_fitness >= target_fitness:
            log.info(
                "Target fitness %.4f reached (current: %+.4f) — stopping.",
                target_fitness, current_fitness,
            )
            break

        # 2. Plateau: last `plateau_window` accepted iterations all tiny
        all_history = load_history(n=999_999, log_path=_log_path)
        accepted_deltas = [
            e["delta"] for e in all_history
            if e.get("accepted") and "delta" in e
        ]
        if len(accepted_deltas) >= plateau_window:
            recent = accepted_deltas[-plateau_window:]
            if all(abs(d) < plateau_threshold for d in recent):
                log.info(
                    "Plateau detected: last %d accepted iterations all had "
                    "|delta fitness| < %.4f — stopping.",
                    plateau_window, plateau_threshold,
                )
                break

    # ── summary ───────────────────────────────────────────────────────────────
    _print_banner("Research loop complete")
    history = load_history(n=n_iterations, log_path=_log_path)
    accepted_count = sum(1 for e in history if e.get("accepted"))
    log.info("Iterations run : %d", n_iterations)
    log.info("Accepted       : %d  (%.0f%%)", accepted_count, 100 * accepted_count / max(n_iterations, 1))
    log.info("Final fitness  : %+.4f", current_fitness)
    log.info("Log            : %s", _log_path)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _print_banner(text: str, width: int = 60) -> None:
    log.info("")
    log.info("═" * width)
    log.info("  %s", text)
    log.info("═" * width)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Autonomous trading strategy research loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--iterations", "-n", type=int, default=10,
        help="Number of research iterations to run",
    )
    p.add_argument(
        "--symbol", "-s", type=str, default="BTC/USDT",
        help="Market symbol (e.g. BTC/USDT, ETH/USDT)",
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
        help="Stop as soon as fitness reaches this value (default: disabled)",
    )
    p.add_argument(
        "--plateau-window", type=int, default=5,
        help="Stop if the last N accepted iterations all improved by less than "
             "--plateau-threshold (default: 5)",
    )
    p.add_argument(
        "--plateau-threshold", type=float, default=0.005,
        help="Minimum meaningful Δ fitness per accepted iteration (default: 0.005)",
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

    run_loop(
        n_iterations=args.iterations,
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
