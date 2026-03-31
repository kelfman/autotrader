"""
optimize_params.py — Bayesian parameter optimization via Optuna TPE.

Replaces LLM-driven parametric search for synthesis strategies with a
surrogate-model optimizer that systematically maps the parameter space.
200 Optuna trials evaluates the equivalent of 200 LLM iterations in ~15 min.

Modes:
  Normal:   maximize fitness (may find fragile spikes)
  Robust:   maximize mean fitness under random ±10% perturbations (finds basins)
  Ensemble: average top-K trials from an existing study (free robustness)

Usage:
    # Stage 1: D+A joint optimization (200 trials)
    python optimize_params.py --strategy runs/synthesis_D_A/strategy.py \
        --preset d_a --trials 200 --augment-funding

    # Robust optimization (perturbation-aware, finds stable basins)
    python optimize_params.py --strategy runs/synthesis_D_A/strategy.py \
        --preset d_a --trials 200 --augment-funding --robust

    # Ensemble average of top-20 trials from existing study
    python optimize_params.py --strategy runs/synthesis_D_A/strategy.py \
        --preset d_a --augment-funding --resume --ensemble 20 \
        --stability-check --walkforward

    # Evaluate best result with stability + walk-forward
    python optimize_params.py --strategy runs/synthesis_D_A/strategy.py \
        --preset d_a --augment-funding --trials 0 --resume \
        --stability-check --walkforward
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pandas as pd

from backtest import run_backtest
from data import augment_with_timeframes, fetch_funding_rates, fetch_ohlcv
from evaluate import FitnessResult, evaluate_strategy

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent


# ── Search space presets ──────────────────────────────────────────────────────

def suggest_d_a(trial: optuna.Trial) -> dict:
    """D+A synthesis search space (brief §5.15.3)."""
    return {
        "sma_period":     trial.suggest_int("sma_period", 96, 384, step=8),
        "fr_pct_window":  trial.suggest_int("fr_pct_window", 360, 1440, step=24),
        "fr_entry_pct":   trial.suggest_float("fr_entry_pct", 0.30, 0.80),
        "fr_exit_pct":    trial.suggest_float("fr_exit_pct", 0.70, 0.95),
        "exit_lookback":  trial.suggest_int("exit_lookback", 20, 72),
        "vol_lookback":   trial.suggest_int("vol_lookback", 12, 72),
        "vol_pct_window": trial.suggest_int("vol_pct_window", 720, 2160, step=24),
        "vol_entry_pct":  trial.suggest_float("vol_entry_pct", 0.40, 0.85),
        "vol_exit_pct":   trial.suggest_float("vol_exit_pct", 0.55, 0.90),
    }


def suggest_d_a_b_c(trial: optuna.Trial) -> dict:
    """D+A+B+C synthesis with boolean switches for marginal signals."""
    params = suggest_d_a(trial)

    params["use_session_filter"] = trial.suggest_categorical(
        "use_session_filter", [True, False]
    )
    if params["use_session_filter"]:
        params["session_start_hour"] = trial.suggest_int("session_start_hour", 0, 23)
        params["session_end_hour"] = trial.suggest_int("session_end_hour", 0, 23)

    params["use_ratio_filter"] = trial.suggest_categorical(
        "use_ratio_filter", [True, False]
    )
    if params["use_ratio_filter"]:
        params["ratio_lookback"] = trial.suggest_int("ratio_lookback", 24, 360, step=24)
        params["ratio_z_entry"] = trial.suggest_float("ratio_z_entry", -2.0, 0.0)

    return params


def suggest_basis(trial: optuna.Trial) -> dict:
    """Track G basis spread search space."""
    return {
        "sma_period":     trial.suggest_int("sma_period", 96, 384, step=8),
        "z_window":       trial.suggest_int("z_window", 168, 1440, step=24),
        "z_entry":        trial.suggest_float("z_entry", -2.5, -0.3),
        "z_exit":         trial.suggest_float("z_exit", 0.5, 2.5),
        "exit_lookback":  trial.suggest_int("exit_lookback", 20, 72),
    }


def suggest_d_a_sized(trial: optuna.Trial) -> dict:
    """V3 D+A with continuous position sizing (brief §6.5.2)."""
    return {
        # Trend filter
        "sma_period":       trial.suggest_int("sma_period", 96, 384, step=8),
        # Funding rate
        "fr_pct_window":    trial.suggest_int("fr_pct_window", 360, 1440, step=24),
        "fr_entry_pct":     trial.suggest_float("fr_entry_pct", 0.30, 0.80),
        "fr_exit_pct":      trial.suggest_float("fr_exit_pct", 0.70, 0.95),
        # Exit
        "exit_lookback":    trial.suggest_int("exit_lookback", 20, 72),
        # Vol regime (for sizing)
        "vol_lookback":     trial.suggest_int("vol_lookback", 12, 72),
        "vol_pct_window":   trial.suggest_int("vol_pct_window", 720, 2160, step=24),
        # Position sizing params
        "vol_size_floor":   trial.suggest_float("vol_size_floor", 0.05, 0.50),
        "vol_size_ceiling": trial.suggest_float("vol_size_ceiling", 0.50, 1.0),
        "vol_size_midpoint": trial.suggest_float("vol_size_midpoint", 0.30, 0.70),
        "fr_size_weight":   trial.suggest_float("fr_size_weight", 0.0, 1.0),
    }


def suggest_d_a_mtf(trial: optuna.Trial) -> dict:
    """V3 D+A with multi-timeframe signals (brief §6.5.1)."""
    return {
        # Daily trend filter (replaces 1h SMA)
        "d1_sma_period":    trial.suggest_int("d1_sma_period", 20, 200, step=10),
        # 4h confirmation
        "h4_sma_period":    trial.suggest_int("h4_sma_period", 20, 200, step=10),
        "use_h4_confirmation": trial.suggest_categorical("use_h4_confirmation", [True, False]),
        # Funding rate
        "fr_pct_window":    trial.suggest_int("fr_pct_window", 360, 1440, step=24),
        "fr_entry_pct":     trial.suggest_float("fr_entry_pct", 0.30, 0.80),
        "fr_exit_pct":      trial.suggest_float("fr_exit_pct", 0.70, 0.95),
        # Exit
        "exit_lookback":    trial.suggest_int("exit_lookback", 20, 72),
        # Vol regime gate
        "vol_lookback":     trial.suggest_int("vol_lookback", 12, 72),
        "vol_pct_window":   trial.suggest_int("vol_pct_window", 720, 2160, step=24),
        "vol_entry_pct":    trial.suggest_float("vol_entry_pct", 0.40, 0.85),
        "vol_exit_pct":     trial.suggest_float("vol_exit_pct", 0.55, 0.90),
        # MTF-specific exits
        "use_d1_vol_exit":  trial.suggest_categorical("use_d1_vol_exit", [True, False]),
        "d1_vol_exit_pct":  trial.suggest_float("d1_vol_exit_pct", 0.60, 0.95),
    }


PRESETS: dict[str, Callable] = {
    "d_a": suggest_d_a,
    "d_a_b_c": suggest_d_a_b_c,
    "basis": suggest_basis,
    "d_a_sized": suggest_d_a_sized,
    "d_a_mtf": suggest_d_a_mtf,
}


# ── Strategy loading ──────────────────────────────────────────────────────────

def load_strategy(path: Path):
    """Load compute_signals function from a strategy module."""
    spec = importlib.util.spec_from_file_location("_opt_strategy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_signals, mod.PARAMS


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(
    symbol: str = "BTC/USDT",
    days: int = 730,
    augment_funding: bool = False,
    augment_eth: bool = False,
    augment_basis: bool = False,
    augment_timeframes: bool = False,
) -> pd.DataFrame:
    """Load and augment OHLCV data once for all trials."""
    df = fetch_ohlcv(symbol, "1h", days=days)

    if augment_funding:
        funding = fetch_funding_rates("BTC/USDT:USDT", days=days)
        df = df.join(funding, how="left")
        df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)

    if augment_eth:
        eth = fetch_ohlcv("ETH/USDT", "1h", days=days)
        eth = eth.rename(columns={c: f"eth_{c}" for c in eth.columns})
        df = df.join(eth, how="left")
        for col in eth.columns:
            df[col] = df[col].ffill()

    if augment_basis:
        from data import fetch_perp_ohlcv
        perp = fetch_perp_ohlcv("BTC/USDT:USDT", timeframe="1h", days=days)
        perp_close = perp[["close"]].rename(columns={"close": "perp_close"})
        df = df.join(perp_close, how="left")
        df["perp_close"] = df["perp_close"].ffill().fillna(df["close"])
        df["basis"] = df["perp_close"] - df["close"]
        df["basis_pct"] = df["basis"] / df["close"]

    if augment_timeframes:
        df = augment_with_timeframes(df)

    return df


# ── Objective ─────────────────────────────────────────────────────────────────

def make_objective(
    compute_signals_fn: Callable,
    df: pd.DataFrame,
    suggest_fn: Callable,
    lambda_penalty: float = 0.5,
    slippage: float = 0.0005,
) -> Callable:
    """Factory for Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        try:
            result = evaluate_strategy(
                compute_signals_fn,
                params,
                df=df,
                lambda_penalty=lambda_penalty,
                slippage=slippage,
            )
            trial.set_user_attr("mean_sharpe", result.mean_sharpe)
            trial.set_user_attr("std_sharpe", result.std_sharpe)
            per_window = [
                {"sharpe": r.sharpe, "trades": r.n_trades, "return_pct": r.total_return_pct}
                for r in result.window_results
            ]
            trial.set_user_attr("windows", per_window)
            return result.fitness
        except Exception as e:
            log.warning("Trial %d failed: %s", trial.number, e)
            return -10.0

    return objective


# ── Robust objective ──────────────────────────────────────────────────────────

def make_robust_objective(
    compute_signals_fn: Callable,
    df: pd.DataFrame,
    suggest_fn: Callable,
    lambda_penalty: float = 0.5,
    n_perturbations: int = 3,
    perturbation_pct: float = 0.10,
    slippage: float = 0.0005,
) -> Callable:
    """
    Objective that evaluates base params AND random perturbations.
    Returns mean fitness across all evaluations. Forces Optuna to find
    broad basins rather than narrow spikes.

    Each trial costs (1 + n_perturbations) evaluations. With n_perturbations=3,
    that's ~1.6s/trial instead of 0.4s.
    """

    def objective(trial: optuna.Trial) -> float:
        base_params = suggest_fn(trial)

        try:
            base_result = evaluate_strategy(
                compute_signals_fn, base_params, df=df,
                lambda_penalty=lambda_penalty,
                slippage=slippage,
            )
            fitnesses = [base_result.fitness]
        except Exception as e:
            log.warning("Trial %d base eval failed: %s", trial.number, e)
            return -10.0

        rng = np.random.RandomState(trial.number * 1000 + 7)
        for _ in range(n_perturbations):
            perturbed = {}
            for name, value in base_params.items():
                if isinstance(value, bool) or isinstance(value, str):
                    perturbed[name] = value
                elif isinstance(value, int):
                    scale = rng.uniform(
                        1.0 - perturbation_pct, 1.0 + perturbation_pct
                    )
                    perturbed[name] = max(1, round(value * scale))
                elif isinstance(value, float):
                    scale = rng.uniform(
                        1.0 - perturbation_pct, 1.0 + perturbation_pct
                    )
                    perturbed[name] = value * scale
                else:
                    perturbed[name] = value

            try:
                result = evaluate_strategy(
                    compute_signals_fn, perturbed, df=df,
                    lambda_penalty=lambda_penalty,
                    slippage=slippage,
                )
                fitnesses.append(result.fitness)
            except Exception:
                fitnesses.append(-10.0)

        robust_fitness = float(np.mean(fitnesses))

        trial.set_user_attr("base_fitness", base_result.fitness)
        trial.set_user_attr("robust_fitness", robust_fitness)
        trial.set_user_attr("min_perturbed", float(np.min(fitnesses)))
        trial.set_user_attr("mean_sharpe", base_result.mean_sharpe)
        trial.set_user_attr("std_sharpe", base_result.std_sharpe)
        per_window = [
            {"sharpe": r.sharpe, "trades": r.n_trades, "return_pct": r.total_return_pct}
            for r in base_result.window_results
        ]
        trial.set_user_attr("windows", per_window)

        return robust_fitness

    return objective


# ── Ensemble averaging ────────────────────────────────────────────────────────

def ensemble_average(
    study: optuna.Study,
    top_k: int = 20,
) -> tuple[dict, dict, list]:
    """
    Average parameters from top-K trials to find a robust basin center.

    Returns (averaged_params, param_spreads, top_trials).
    param_spreads has {name: {mean, std, min, max}} for each numeric param.
    """
    completed = [t for t in study.trials if t.value is not None and t.value > -5.0]
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_k]

    if not top_trials:
        raise ValueError("No valid trials to average")

    param_names = list(top_trials[0].params.keys())
    averaged: dict = {}
    spreads: dict = {}

    for name in param_names:
        values = [t.params[name] for t in top_trials]
        if isinstance(values[0], bool):
            averaged[name] = sum(values) > len(values) / 2
        elif isinstance(values[0], int):
            avg = sum(values) / len(values)
            averaged[name] = round(avg)
            spreads[name] = {
                "mean": avg, "std": float(np.std(values)),
                "min": min(values), "max": max(values),
            }
        elif isinstance(values[0], float):
            avg = sum(values) / len(values)
            averaged[name] = avg
            spreads[name] = {
                "mean": avg, "std": float(np.std(values)),
                "min": min(values), "max": max(values),
            }
        else:
            averaged[name] = values[0]

    return averaged, spreads, top_trials


def print_ensemble_report(
    averaged_params: dict,
    spreads: dict,
    fitness: float,
    single_best_fitness: float,
    top_trials: list,
    stability: dict | None = None,
) -> None:
    """Print formatted ensemble averaging results."""
    print()
    print("=" * 90)
    print("  ENSEMBLE AVERAGING REPORT")
    print(
        f"  Top-{len(top_trials)} trials averaged  |  "
        f"Fitness range: {top_trials[-1].value:+.4f} to {top_trials[0].value:+.4f}"
    )
    print("=" * 90)

    print(f"\n  Ensemble fitness:      {fitness:+.4f}")
    print(f"  Single-best fitness:   {single_best_fitness:+.4f}")
    delta = fitness - single_best_fitness
    print(f"  Delta:                 {delta:+.4f} ({delta / abs(single_best_fitness) * 100:+.1f}%)")

    print(f"\n  Averaged parameters (with top-{len(top_trials)} spread):")
    print(
        f"    {'Parameter':<20}  {'Ensemble':>10}  {'Single-Best':>12}  "
        f"{'Spread (std)':>12}  {'Range':>20}"
    )
    print(f"    {'─' * 20}  {'─' * 10}  {'─' * 12}  {'─' * 12}  {'─' * 20}")

    single_best_params = top_trials[0].params
    for name in sorted(averaged_params.keys()):
        val = averaged_params[name]
        sb_val = single_best_params.get(name, "?")
        sp = spreads.get(name)
        if isinstance(val, float):
            val_str = f"{val:.4f}"
            sb_str = f"{sb_val:.4f}" if isinstance(sb_val, float) else str(sb_val)
        else:
            val_str = str(val)
            sb_str = str(sb_val)

        if sp:
            std_str = f"{sp['std']:.2f}"
            range_str = f"[{sp['min']:.2f}, {sp['max']:.2f}]"
        else:
            std_str = "—"
            range_str = "—"

        print(f"    {name:<20}  {val_str:>10}  {sb_str:>12}  {std_str:>12}  {range_str:>20}")

    if stability is not None:
        verdict = "STABLE" if stability["stable"] else "FRAGILE"
        print(
            f"\n  Stability check (+/-{stability['perturbation_pct']:.0f}% perturbation):"
        )
        print(
            f"    Verdict: {verdict}  "
            f"(worst drop: {stability['worst_drop_pct']:.1f}%)"
        )
        if stability["perturbations"]:
            print(f"\n    Parameter sensitivity:")
            for p in stability["perturbations"][:5]:
                print(
                    f"      {p['param']:<20}  {p['direction']:>5}  "
                    f"fitness={p['fitness']:+.4f}  "
                    f"drop={p['fitness_drop_pct']:+.1f}%"
                )

    print()
    print("=" * 90)


# ── Stability check ──────────────────────────────────────────────────────────

def stability_check(
    compute_signals_fn: Callable,
    df: pd.DataFrame,
    best_params: dict,
    perturbation: float = 0.10,
    lambda_penalty: float = 0.5,
) -> dict:
    """
    Perturb each numeric parameter by ±perturbation and re-evaluate.
    If worst-case fitness drops by >20%, the optimum is fragile.
    """
    base_result = evaluate_strategy(
        compute_signals_fn, best_params, df=df, lambda_penalty=lambda_penalty
    )
    base_fitness = base_result.fitness

    perturbations = []
    for name, value in best_params.items():
        if isinstance(value, bool) or isinstance(value, str):
            continue
        for direction in [-1, +1]:
            perturbed = dict(best_params)
            if isinstance(value, int):
                delta = max(1, int(abs(value) * perturbation))
                perturbed[name] = value + direction * delta
            elif isinstance(value, float):
                perturbed[name] = value + direction * abs(value) * perturbation
            else:
                continue

            try:
                result = evaluate_strategy(
                    compute_signals_fn, perturbed, df=df,
                    lambda_penalty=lambda_penalty,
                )
                drop = (
                    (base_fitness - result.fitness) / abs(base_fitness) * 100
                    if base_fitness != 0
                    else 0
                )
                perturbations.append({
                    "param": name,
                    "direction": f"{'+' if direction > 0 else '-'}{perturbation * 100:.0f}%",
                    "original": value,
                    "perturbed": perturbed[name],
                    "fitness": result.fitness,
                    "fitness_drop_pct": drop,
                })
            except Exception:
                perturbations.append({
                    "param": name,
                    "direction": f"{'+' if direction > 0 else '-'}{perturbation * 100:.0f}%",
                    "original": value,
                    "perturbed": perturbed[name],
                    "fitness": -10.0,
                    "fitness_drop_pct": 100.0,
                })

    worst_drop = max(p["fitness_drop_pct"] for p in perturbations) if perturbations else 0
    stable = worst_drop < 20.0

    return {
        "base_fitness": base_fitness,
        "perturbation_pct": perturbation * 100,
        "worst_drop_pct": worst_drop,
        "stable": stable,
        "perturbations": sorted(perturbations, key=lambda p: -p["fitness_drop_pct"]),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(study: optuna.Study, stability: dict | None = None) -> None:
    """Print formatted optimization results."""
    best = study.best_trial

    print()
    print("=" * 90)
    print("  BAYESIAN OPTIMIZATION REPORT")
    print(
        f"  Study: {study.study_name}  |  Trials: {len(study.trials)}  "
        f"|  Best: #{best.number}"
    )
    print("=" * 90)

    print(f"\n  Best fitness: {best.value:+.4f}")
    print(f"\n  Best parameters:")
    for name, value in sorted(best.params.items()):
        if isinstance(value, float):
            print(f"    {name:<20} = {value:.4f}")
        else:
            print(f"    {name:<20} = {value}")

    # Per-window breakdown for best trial
    windows = best.user_attrs.get("windows")
    if windows:
        print(f"\n  Per-window breakdown (best trial):")
        print(f"    {'Window':>6}  {'Sharpe':>8}  {'Return':>8}  {'Trades':>6}")
        print(f"    {'------':>6}  {'------':>8}  {'------':>8}  {'------':>6}")
        for i, w in enumerate(windows, 1):
            print(
                f"    W{i:>4}  {w['sharpe']:>+8.3f}  {w['return_pct']:>+7.1f}%  "
                f"{w['trades']:>6}"
            )

    # Parameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"\n  Parameter importance:")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
            bar = "#" * int(imp * 40)
            print(f"    {name:<20}  {imp:.3f}  {bar}")
    except Exception:
        print("\n  (Parameter importance unavailable — need more completed trials)")

    # Top 5 trials
    completed = [t for t in study.trials if t.value is not None]
    trials_sorted = sorted(completed, key=lambda t: t.value, reverse=True)
    print(f"\n  Top 5 trials:")
    print(f"    {'#':>5}  {'Fitness':>9}  {'Mean Sharpe':>12}  {'Std':>6}  Key params")
    print(f"    {'-----':>5}  {'---------':>9}  {'----------':>12}  {'---':>6}  {'----------'}")
    for t in trials_sorted[:5]:
        ms = t.user_attrs.get("mean_sharpe", "?")
        ss = t.user_attrs.get("std_sharpe", "?")
        ms_str = f"{ms:+.3f}" if isinstance(ms, (int, float)) else ms
        ss_str = f"{ss:.3f}" if isinstance(ss, (int, float)) else ss
        print(f"    {t.number:>5}  {t.value:>+9.4f}  {ms_str:>12}  {ss_str:>6}")

    # Stability check
    if stability is not None:
        print(
            f"\n  Stability check "
            f"(+/-{stability['perturbation_pct']:.0f}% perturbation):"
        )
        verdict = "STABLE" if stability["stable"] else "FRAGILE"
        print(
            f"    Verdict: {verdict}  "
            f"(worst drop: {stability['worst_drop_pct']:.1f}%)"
        )
        if not stability["stable"]:
            print(f"\n    Most sensitive parameters:")
            for p in stability["perturbations"][:5]:
                print(
                    f"      {p['param']:<20}  {p['direction']:>5}  "
                    f"fitness={p['fitness']:+.4f}  "
                    f"drop={p['fitness_drop_pct']:+.1f}%"
                )

    print()
    print("=" * 90)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bayesian parameter optimization via Optuna TPE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--strategy", type=str, required=True,
        help="Path to strategy.py to optimize",
    )
    p.add_argument(
        "--preset", type=str, choices=list(PRESETS.keys()), required=True,
        help="Search space preset",
    )
    p.add_argument(
        "--trials", "-n", type=int, default=200,
        help="Number of optimization trials",
    )
    p.add_argument("--study-name", type=str, default=None,
                   help="Optuna study name (defaults to preset name)")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--lambda-penalty", type=float, default=0.5)
    p.add_argument("--augment-funding", action="store_true",
                   help="Merge funding rate data (required for D+A)")
    p.add_argument("--augment-eth", action="store_true",
                   help="Merge ETH/USDT data (required for Stage 2 ratio filter)")
    p.add_argument("--augment-basis", action="store_true",
                   help="Merge perp-spot basis data (required for Track G)")
    p.add_argument("--augment-timeframes", action="store_true",
                   help="Add 4h and 1d resampled columns (required for V3 MTF)")
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing study from SQLite storage")
    p.add_argument("--stability-check", action="store_true",
                   help="Run stability check on the best result")
    p.add_argument("--walkforward", action="store_true",
                   help="Run walk-forward validation on the best result")
    p.add_argument("--write-params", type=str, default=None,
                   help="Write best params to this JSON file")
    p.add_argument("--json", action="store_true",
                   help="Output results as JSON")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    robust = p.add_argument_group("robust optimization")
    robust.add_argument("--robust", action="store_true",
                        help="Use perturbation-aware objective (finds stable basins)")
    robust.add_argument("--n-perturbations", type=int, default=3,
                        help="Random perturbations per trial in robust mode")
    robust.add_argument("--perturbation-pct", type=float, default=0.10,
                        help="Perturbation magnitude (0.10 = +/-10%%)")

    ens = p.add_argument_group("ensemble averaging")
    ens.add_argument("--ensemble", type=int, default=0, metavar="K",
                     help="Average top-K trials from an existing study (requires --resume)")
    return p


def _run_ensemble(args, compute_signals_fn, df, study) -> None:
    """Handle --ensemble mode: average top-K trials and evaluate."""
    top_k = args.ensemble
    print(f"Computing ensemble average of top-{top_k} trials...", file=sys.stderr)

    averaged_params, spreads, top_trials = ensemble_average(study, top_k=top_k)
    single_best_fitness = study.best_value

    print("Evaluating ensemble params...", file=sys.stderr)
    result = evaluate_strategy(
        compute_signals_fn, averaged_params, df=df,
        lambda_penalty=args.lambda_penalty,
    )

    stab = None
    if args.stability_check:
        print("Running stability check on ensemble...", file=sys.stderr)
        stab = stability_check(
            compute_signals_fn, df, averaged_params,
            lambda_penalty=args.lambda_penalty,
        )

    wf_results = None
    if args.walkforward:
        print("Running walk-forward on ensemble...", file=sys.stderr)
        from validate_walkforward import run_walkforward
        wf_results = run_walkforward(compute_signals_fn, averaged_params, df)

    out_path = Path(args.strategy).parent / f"optuna_{study.study_name}_ensemble{top_k}.json"
    out_path.write_text(json.dumps(averaged_params, indent=2, default=str) + "\n")
    print(f"Ensemble params written to {out_path}", file=sys.stderr)

    if args.json:
        output = {
            "mode": "ensemble",
            "top_k": top_k,
            "ensemble_fitness": result.fitness,
            "single_best_fitness": single_best_fitness,
            "ensemble_params": averaged_params,
            "param_spreads": spreads,
        }
        if stab:
            output["stability"] = {
                "base_fitness": stab["base_fitness"],
                "worst_drop_pct": stab["worst_drop_pct"],
                "stable": stab["stable"],
            }
        if wf_results:
            output["walkforward"] = wf_results
        print(json.dumps(output, indent=2, default=str))
    else:
        print_ensemble_report(
            averaged_params, spreads, result.fitness,
            single_best_fitness, top_trials, stab,
        )
        if wf_results:
            from validate_walkforward import _print_report as wf_print
            wf_print(wf_results, averaged_params, "ensemble")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if not args.verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    strategy_path = Path(args.strategy)
    run_dir = strategy_path.parent
    compute_signals_fn, _ = load_strategy(strategy_path)

    study_suffix = "_robust" if args.robust else ""
    study_name = args.study_name or f"synthesis_{args.preset}{study_suffix}"
    db_path = run_dir / f"optuna_{study_name}.db"
    storage = f"sqlite:///{db_path}"
    suggest_fn = PRESETS[args.preset]

    print(f"Loading data ({args.symbol})...", file=sys.stderr)
    df = load_data(
        symbol=args.symbol,
        augment_funding=args.augment_funding,
        augment_eth=args.augment_eth,
        augment_basis=getattr(args, "augment_basis", False),
        augment_timeframes=getattr(args, "augment_timeframes", False),
    )
    print(
        f"Data loaded: {len(df)} bars "
        f"[{df.index[0].date()} -> {df.index[-1].date()}]",
        file=sys.stderr,
    )

    # For ensemble mode, load the BASE study (not robust), regardless of --robust
    if args.ensemble > 0:
        base_study_name = args.study_name or f"synthesis_{args.preset}"
        base_db = run_dir / f"optuna_{base_study_name}.db"
        base_storage = f"sqlite:///{base_db}"
        study = optuna.load_study(study_name=base_study_name, storage=base_storage)
        print(
            f"Loaded study '{base_study_name}' ({len(study.trials)} trials)",
            file=sys.stderr,
        )
        _run_ensemble(args, compute_signals_fn, df, study)
        return

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=args.resume,
    )

    existing = len(study.trials)
    remaining = max(0, args.trials - existing) if args.resume else args.trials

    if remaining > 0:
        mode_label = "robust" if args.robust else "normal"
        print(
            f"Running {remaining} trials "
            f"(preset={args.preset}, mode={mode_label}, existing={existing})...",
            file=sys.stderr,
        )

        if args.robust:
            objective = make_robust_objective(
                compute_signals_fn, df, suggest_fn, args.lambda_penalty,
                n_perturbations=args.n_perturbations,
                perturbation_pct=args.perturbation_pct,
            )
        else:
            objective = make_objective(
                compute_signals_fn, df, suggest_fn, args.lambda_penalty
            )

        t0 = time.time()
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
        elapsed = time.time() - t0

        print(
            f"Optimization complete: {remaining} trials in {elapsed:.0f}s "
            f"({elapsed / remaining:.1f}s/trial)",
            file=sys.stderr,
        )

    if not study.trials:
        print("No trials to report.", file=sys.stderr)
        return

    # For robust studies, the "best" params are the ones with best robust fitness.
    # Evaluate base fitness separately for the report.
    best_params = study.best_params

    # Stability check
    stab = None
    if args.stability_check:
        print("Running stability check...", file=sys.stderr)
        stab = stability_check(
            compute_signals_fn, df, best_params,
            lambda_penalty=args.lambda_penalty,
        )

    # Walk-forward
    wf_results = None
    if args.walkforward:
        print("Running walk-forward validation...", file=sys.stderr)
        from validate_walkforward import run_walkforward
        wf_results = run_walkforward(compute_signals_fn, best_params, df)

    # Write best params
    out_path = Path(args.write_params) if args.write_params else None
    if out_path:
        out_path.write_text(json.dumps(best_params, indent=2) + "\n")
        print(f"Best params written to {out_path}", file=sys.stderr)

    default_out = run_dir / f"optuna_{study_name}_best.json"
    default_out.write_text(json.dumps(best_params, indent=2) + "\n")

    # Output
    if args.json:
        output = {
            "study_name": study_name,
            "mode": "robust" if args.robust else "normal",
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_fitness": study.best_value,
            "best_params": best_params,
        }
        if args.robust:
            bt = study.best_trial
            output["base_fitness"] = bt.user_attrs.get("base_fitness")
            output["min_perturbed"] = bt.user_attrs.get("min_perturbed")
        if stab:
            output["stability"] = {
                "base_fitness": stab["base_fitness"],
                "worst_drop_pct": stab["worst_drop_pct"],
                "stable": stab["stable"],
            }
        if wf_results:
            output["walkforward"] = wf_results
        print(json.dumps(output, indent=2, default=str))
    else:
        print_report(study, stab)
        if wf_results:
            from validate_walkforward import _print_report as wf_print
            wf_print(wf_results, best_params, str(strategy_path))


if __name__ == "__main__":
    main()
