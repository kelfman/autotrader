"""
agent.py — LLM research agent for autonomous strategy discovery.

The agent reads:
  - Current strategy.py source
  - Current fitness result (per-window breakdown)
  - Recent experiment history (from experiment_log.jsonl)

And outputs a StrategySpec: a structured modification proposal that the
research loop applies to strategy.py, re-evaluates, and accepts or rejects.

Two agent modes:
  "api"   — calls the Anthropic API with tool_use (original, requires API key)
  "local" — writes context to agent_request.json and polls for agent_response.json,
             allowing an external agent (e.g. Claude in Cursor) to provide proposals
             without API calls.

V2: pass a TrackConfig to parameterise the system prompt per research track.
    If no TrackConfig is supplied, the original V1 TA-baseline behaviour is used.

Usage (standalone smoke-test):
    python agent.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from track_config import TrackConfig

log = logging.getLogger(__name__)


# ── StrategySpec ──────────────────────────────────────────────────────────────

class StrategySpec(BaseModel):
    """
    Structured modification proposal output by the LLM agent.

    The research loop applies this to strategy.py, re-evaluates, and keeps
    the change only if fitness improves.

    Fields:
        change_type:      "parametric" (PARAMS values only) or "structural" (new logic).
        rationale:        Brief explanation of what changed and why.
        params:           Complete replacement for the PARAMS dict.
                          ALL existing keys must be present; new keys are allowed.
                          Values must be numeric.
        new_signals_code: Full source of the new compute_signals() function,
                          including the def line and its docstring.
                          Must be None for parametric changes.
                          Must be valid Python using the track's allowed libraries.
    """

    change_type: Literal["parametric", "structural"]
    rationale: str = Field(description="1-3 sentences explaining what changed and why")
    params: dict[str, float] = Field(
        description="Complete replacement for PARAMS. Include ALL keys, even unchanged ones."
    )
    new_signals_code: str | None = Field(
        default=None,
        description=(
            "Full compute_signals() function source including the def line. "
            "Required for structural changes; None for parametric changes."
        ),
    )


# ── V1 system prompt (used when no TrackConfig is supplied) ───────────────────

_V1_SYSTEM_PROMPT = """\
You are an algorithmic trading strategy researcher specialising in crypto markets \
(BTC/ETH, 1-hour candles, 24/7 trading).

YOUR GOAL
Propose ONE modification to a trading strategy that improves its regime-robustness \
fitness score. Each modification is evaluated, then accepted or rejected based on \
whether fitness actually improves.

FITNESS FUNCTION
  fitness = mean(sharpe_across_5_windows) − 0.5 × std(sharpe_across_5_windows)

  The 5 windows are non-overlapping 3-month slices distributed evenly over 2 years \
of BTC/USDT hourly data (covering 2020-crash, 2021-bull, 2022-bear, 2023-24-recovery).

  A consistent Sharpe of 1.4 across all windows beats 2.5 in one and 0.2 in others.
  The variance penalty is the key constraint — target consistency, not peak performance.

TECHNICAL RULES
  - Use the `ta` library for indicators. Do NOT use pandas_ta (Python 3.10+ incompatibility).
  - ta indicator examples:
      ta.trend.ema_indicator(close, window=N)            → EMA Series
      ta.momentum.rsi(close, window=N)                   → RSI Series
      ta.volatility.bollinger_hband(close, window=N, window_dev=D)
      ta.volatility.bollinger_lband(close, window=N, window_dev=D)
      ta.volatility.bollinger_mavg(close, window=N)
      ta.volatility.average_true_range(high, low, close, window=N)
      ta.trend.macd(close, window_slow=26, window_fast=12, window_sign=9)
      ta.trend.macd_signal(close, window_slow=26, window_fast=12, window_sign=9)
      ta.trend.macd_diff(close, window_slow=26, window_fast=12, window_sign=9)
      ta.volume.on_balance_volume(close, volume)
      ta.trend.adx(high, low, close, window=N)           → ADX Series
      ta.momentum.stoch(high, low, close, window=N, smooth_window=M) → stoch %K
  - df columns available: open, high, low, close, volume (UTC DatetimeIndex)
  - compute_signals MUST return (entries, exits) as boolean pandas Series
  - Fill NaN with False (.fillna(False)) before returning
  - Long-only for V1 — no shorting, no position sizing changes
  - Only import libraries already in the project stack

EXPLORATION STRATEGY
  - Study the per-window breakdown: which regimes does the strategy fail in?
  - If std_sharpe is high: prioritise reducing variance (more conservative filters)
  - If mean_sharpe is low across all windows: try a structurally different approach
  - Avoid repeating recently rejected configurations
  - Alternate between parametric tuning and structural changes to explore broadly
  - When proposing structural changes, ensure the new logic is internally consistent \
(e.g., do not combine RSI oversold entry with EMA uptrend — they are contradictory)"""


# ── V1 tool definition (used when no TrackConfig is supplied) ─────────────────

_V1_PROPOSE_TOOL: dict = {
    "name": "propose_strategy_modification",
    "description": (
        "Propose a single, targeted modification to the trading strategy. "
        "Output the complete new PARAMS dict and, for structural changes, "
        "the complete new compute_signals function."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "change_type": {
                "type": "string",
                "enum": ["parametric", "structural"],
                "description": (
                    "'parametric' if only changing PARAMS values. "
                    "'structural' if rewriting compute_signals logic."
                ),
            },
            "rationale": {
                "type": "string",
                "description": (
                    "1-3 sentences: what you changed, why you expect it to "
                    "improve regime-robust fitness, and what failure mode you are addressing."
                ),
            },
            "params": {
                "type": "object",
                "description": (
                    "Complete new PARAMS dict. Include ALL keys, even unchanged ones. "
                    "Values must be numbers (int or float)."
                ),
                "additionalProperties": {"type": "number"},
            },
            "new_signals_code": {
                "type": ["string", "null"],
                "description": (
                    "Full compute_signals() source including the def line. "
                    "Must use the `ta` library (not pandas_ta). "
                    "Must return (entries, exits) as boolean Series. "
                    "Fill NaN with False before returning. "
                    "Leave null for parametric-only changes."
                ),
            },
        },
        "required": ["change_type", "rationale", "params"],
    },
}


# ── V2 prompt and tool builders ───────────────────────────────────────────────

def _build_system_prompt(
    config: TrackConfig,
    iteration_number: int = 0,
) -> str:
    """
    Build a track-specific system prompt from a TrackConfig.

    When iteration_number > 0 and ≤ exploration_phase_iterations, the
    exploration phase instructions replace the general exploration strategy
    section. Anti-convergence constraints are always injected when present.
    """
    libs_str = ", ".join(f"`{lib}`" for lib in config.allowed_libraries)

    std_cols = "open, high, low, close, volume (UTC DatetimeIndex)"
    if config.df_columns_extra:
        extra_str = ", ".join(config.df_columns_extra)
        cols_str = f"{std_cols}; extra columns provided by this track: {extra_str}"
    else:
        cols_str = std_cols

    hints_str = "\n".join(f"  - {hint}" for hint in config.exploration_hints)

    in_exploration_phase = (
        config.exploration_phase_iterations > 0
        and 0 < iteration_number <= config.exploration_phase_iterations
    )

    # Build the exploration/strategy section depending on phase
    if in_exploration_phase and config.exploration_phase_instructions:
        exploration_section = f"""\
EXPLORATION PHASE (iteration {iteration_number} of {config.exploration_phase_iterations})
{config.exploration_phase_instructions}"""
    else:
        exploration_section = f"""\
EXPLORATION HINTS FOR THIS TRACK
{hints_str}

GENERAL EXPLORATION STRATEGY
  - Study the per-window breakdown: which regimes does the strategy fail in?
  - If std_sharpe is high: prioritise reducing variance (more conservative filters)
  - If mean_sharpe is low across all windows: try a structurally different approach
  - Avoid repeating recently rejected configurations
  - Alternate between parametric tuning and structural changes to explore broadly
  - When proposing structural changes, ensure the new logic is internally consistent"""

    # Anti-convergence constraints (Layer 1, §5.12.3)
    constraints_section = ""
    if config.primary_signal_requirement:
        forbidden_str = "\n".join(
            f"  - {p}" for p in config.forbidden_entry_patterns
        )
        constraints_section = f"""

SIGNAL CLASS REQUIREMENTS (MANDATORY)
{config.primary_signal_requirement}

FORBIDDEN PATTERNS — DO NOT USE:
{forbidden_str}

PARAMETER BUDGET: Maximum {config.max_params} parameters in PARAMS."""

    return f"""\
You are an algorithmic trading strategy researcher specialising in crypto markets \
(BTC/ETH, 1-hour candles, 24/7 trading). You are working on research Track {config.track_id}: \
{config.description}.

YOUR GOAL
Propose ONE modification to a trading strategy that improves its regime-robustness \
fitness score. Each modification is evaluated, then accepted or rejected based on \
whether fitness actually improves.

FITNESS FUNCTION
  fitness = mean(sharpe_across_5_windows) − 0.5 × std(sharpe_across_5_windows)

  The 5 windows are non-overlapping 3-month slices over 2 years of BTC/USDT hourly \
data, covering diverse market regimes: bull runs, bear phases, choppy/sideways markets, \
and recovery periods.

  A consistent Sharpe of 1.4 across all windows beats 2.5 in one and 0.2 in others.
  The variance penalty is the key constraint — target consistency, not peak performance.
  Windows with fewer than 3 trades are scored as Sharpe = 0. Avoid zeroing out windows.

  TA BASELINE (Track E): The EMA-crossover TA strategy achieved fitness = \
{config.ta_baseline_fitness:+.4f} after 55 iterations. Your track must beat this \
to be considered an improvement over TA.

SIGNAL CLASS: {config.signal_class.upper().replace("_", " ")}
{config.signal_class_brief}

REDUCIBILITY CHECK
{config.reducibility_note}

TECHNICAL RULES
  - Allowed libraries: {libs_str}. Do NOT use pandas_ta (Python 3.10+ incompatibility).
  - df columns available: {cols_str}
  - compute_signals MUST return (entries, exits) as boolean pandas Series
  - Fill NaN with False (.fillna(False)) before returning
  - Long-only — no shorting, no position sizing changes
  - Only import from the explicitly allowed libraries listed above

{exploration_section}{constraints_section}"""


def _build_propose_tool(config: TrackConfig) -> dict:
    """Build the tool definition with track-specific library constraints."""
    libs_str = ", ".join(config.allowed_libraries)
    return {
        "name": "propose_strategy_modification",
        "description": (
            "Propose a single, targeted modification to the trading strategy. "
            "Output the complete new PARAMS dict and, for structural changes, "
            "the complete new compute_signals function."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "change_type": {
                    "type": "string",
                    "enum": ["parametric", "structural"],
                    "description": (
                        "'parametric' if only changing PARAMS values. "
                        "'structural' if rewriting compute_signals logic."
                    ),
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "1-3 sentences: what you changed, why you expect it to "
                        "improve regime-robust fitness, and what failure mode you are addressing."
                    ),
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Complete new PARAMS dict. Include ALL keys, even unchanged ones. "
                        "Values must be numbers (int or float)."
                    ),
                    "additionalProperties": {"type": "number"},
                },
                "new_signals_code": {
                    "type": ["string", "null"],
                    "description": (
                        "Full compute_signals() source including the def line. "
                        f"Must only use libraries: {libs_str}. "
                        "Must return (entries, exits) as boolean Series. "
                        "Fill NaN with False before returning. "
                        "Leave null for parametric-only changes."
                    ),
                },
            },
            "required": ["change_type", "rationale", "params"],
        },
    }


# ── Agent function ────────────────────────────────────────────────────────────

def propose_modification(
    strategy_source: str,
    current_fitness: float,
    fitness_summary: str,
    experiment_history: list[dict],
    model: str = "claude-opus-4-6",
    track_config: TrackConfig | None = None,
    iteration_number: int = 0,
    course_correction: str | None = None,
) -> StrategySpec:
    """
    Call the LLM agent and return a StrategySpec modification proposal.

    Args:
        strategy_source:    Full text of the current strategy.py.
        current_fitness:    Scalar fitness score of the current strategy.
        fitness_summary:    Human-readable per-window breakdown (FitnessResult.summary()).
        experiment_history: Recent experiment dicts from experiment_log.jsonl.
        model:              Anthropic model to use.
        track_config:       V2 track configuration. If None, uses V1 TA-baseline
                            system prompt (backward compatible).
        iteration_number:   Current iteration number (used for exploration phase).
        course_correction:  PI review feedback to inject (§5.12.5). If set,
                            appended as a mandatory course-correction section.

    Returns:
        StrategySpec describing the proposed modification.

    Raises:
        RuntimeError: if ANTHROPIC_API_KEY is unset or the API returns no tool call.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)

    if track_config is not None:
        system_prompt = _build_system_prompt(track_config, iteration_number=iteration_number)
        propose_tool = _build_propose_tool(track_config)
        track_label = f"Track {track_config.track_id} ({track_config.signal_class})"
    else:
        system_prompt = _V1_SYSTEM_PROMPT
        propose_tool = _V1_PROPOSE_TOOL
        track_label = "V1 (TA baseline)"

    history_str = _format_history(experiment_history)

    user_message = f"""\
## Current Strategy (strategy.py)

```python
{strategy_source}
```

## Current Performance

Fitness score: {current_fitness:+.4f}

{fitness_summary}

## Recent Experiment History (chronological — most recent last)

{history_str}

---

Propose ONE modification using the propose_strategy_modification tool.

Think step-by-step:
1. Which regime windows are weakest, and why might that be?
2. What is the highest-leverage change — parameter tweak or logic rewrite?
3. What would you predict for the new per-window Sharpe distribution?

Your goal is to reduce fitness variance across regimes while maintaining or \
improving mean Sharpe."""

    if course_correction:
        user_message += f"""

## PI Review — Course Correction (mandatory)

{course_correction}

You MUST address this feedback in your next proposal. Ignoring PI direction \
will result in the proposal being rejected regardless of fitness."""

    log.info("Calling %s [%s] for modification proposal…", model, track_label)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        tools=[propose_tool],
        tool_choice={"type": "tool", "name": "propose_strategy_modification"},
        messages=[{"role": "user", "content": user_message}],
    )

    tool_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_block is None:
        raise RuntimeError(
            "Agent returned no tool_use block — unexpected API response format. "
            f"Stop reason: {response.stop_reason}"
        )

    spec = StrategySpec(**tool_block.input)
    log.info(
        "Agent proposed [%s]: %s",
        spec.change_type,
        spec.rationale[:120],
    )
    return spec


# ── Local agent (file-based) ──────────────────────────────────────────────────

_LOCAL_POLL_INTERVAL = 2.0  # seconds between checks for response file


def propose_modification_local(
    strategy_source: str,
    current_fitness: float,
    fitness_summary: str,
    experiment_history: list[dict],
    run_dir: Path,
    track_config: TrackConfig | None = None,
) -> StrategySpec:
    """
    File-based agent: write context to agent_request.json, wait for agent_response.json.

    This allows an external agent (e.g. Claude in Cursor) to act as the research
    agent without any API calls. The research loop pauses until the response file
    appears.

    Request file (written by this function):
        {run_dir}/agent_request.json — contains strategy source, fitness, history.

    Response file (written by the external agent):
        {run_dir}/agent_response.json — a StrategySpec JSON object:
        {
            "change_type": "parametric" | "structural",
            "rationale": "...",
            "params": {"key": value, ...},
            "new_signals_code": "..." | null
        }

    Both files are deleted after the response is read.
    """
    request_path = run_dir / "agent_request.json"
    response_path = run_dir / "agent_response.json"

    track_label = (
        f"Track {track_config.track_id} ({track_config.signal_class})"
        if track_config else "V1 (TA baseline)"
    )

    history_str = _format_history(experiment_history)

    request = {
        "track": track_label,
        "current_fitness": round(current_fitness, 6),
        "fitness_summary": fitness_summary,
        "strategy_source": strategy_source,
        "experiment_history": experiment_history,
        "history_formatted": history_str,
    }
    if track_config is not None:
        request["track_config"] = {
            "track_id": track_config.track_id,
            "signal_class": track_config.signal_class,
            "description": track_config.description,
            "allowed_libraries": track_config.allowed_libraries,
            "exploration_hints": track_config.exploration_hints,
        }

    request_path.write_text(json.dumps(request, indent=2, default=str))

    # Remove stale response file if it exists
    if response_path.exists():
        response_path.unlink()

    log.info(
        "LOCAL AGENT MODE [%s] — waiting for proposal…",
        track_label,
    )
    print(
        f"\n{'=' * 60}\n"
        f"  LOCAL AGENT: waiting for proposal\n"
        f"  Request:  {request_path}\n"
        f"  Respond:  {response_path}\n"
        f"{'=' * 60}\n"
    )

    while not response_path.exists():
        time.sleep(_LOCAL_POLL_INTERVAL)

    raw = response_path.read_text().strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
        raise RuntimeError(f"Invalid JSON in agent_response.json: {exc}") from exc

    try:
        spec = StrategySpec(**data)
    except Exception as exc:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"agent_response.json does not match StrategySpec schema: {exc}"
        ) from exc

    request_path.unlink(missing_ok=True)
    response_path.unlink(missing_ok=True)

    log.info(
        "Agent proposed [%s]: %s",
        spec.change_type,
        spec.rationale[:120],
    )
    return spec


# ── History formatter ─────────────────────────────────────────────────────────

def _format_history(history: list[dict]) -> str:
    """Format experiment history as a compact readable table."""
    if not history:
        return "(no experiments yet — this is the first iteration)"

    lines = [
        f"{'#':>3}  {'Type':<12}  {'Before':>7}  {'After':>7}  {'Δ':>7}  {'OK':>3}  Rationale",
        "─" * 90,
    ]
    for entry in history:
        before = entry.get("fitness_before", 0.0)
        after = entry.get("fitness_after", 0.0)
        delta = after - before
        ok = "✓" if entry.get("accepted") else "✗"
        ctype = str(entry.get("change_type", "?"))[:12]
        rationale = str(entry.get("rationale", ""))[:55]
        n = entry.get("iteration", "?")
        lines.append(
            f"{n:>3}  {ctype:<12}  {before:>+7.4f}  {after:>+7.4f}  {delta:>+7.4f}  {ok:>3}  {rationale}"
        )

    return "\n".join(lines)


# ── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    strategy_path = Path(__file__).parent / "strategy.py"
    if not strategy_path.exists():
        print("strategy.py not found — run from the Autotrader directory", file=sys.stderr)
        sys.exit(1)

    strategy_source = strategy_path.read_text()

    # Synthetic fitness summary to test the agent without running backtests
    fitness_summary = (
        "── Fitness: +0.2300  (mean_sharpe=+0.480, std=0.500, λ=0.5)\n"
        "   Symbol: BTC/USDT  Windows: 5\n"
        "   ✓ W1: [2024-03-01 → 2024-05-31] Sharpe=+0.810  Return=+8.2%  MDD=12.1%  Trades=14  WinRate=57%\n"
        "   ✓ W2: [2024-06-14 → 2024-09-13] Sharpe=+0.320  Return=+3.1%  MDD=9.8%   Trades=11  WinRate=45%\n"
        "   ✓ W3: [2024-09-28 → 2024-12-28] Sharpe=+0.610  Return=+6.0%  MDD=8.3%   Trades=16  WinRate=62%\n"
        "   ✓ W4: [2025-01-11 → 2025-04-12] Sharpe=+0.520  Return=+5.0%  MDD=11.2%  Trades=12  WinRate=50%\n"
        "   ⚠ W5: [2025-04-27 → 2025-07-27] Sharpe=-0.360  Return=-3.5%  MDD=18.4%  Trades=8   WinRate=37%"
    )

    # Test V1 path (no track config)
    print("\n── V1 path (no TrackConfig) ───────────────────────────────────────")
    spec = propose_modification(
        strategy_source=strategy_source,
        current_fitness=0.23,
        fitness_summary=fitness_summary,
        experiment_history=[],
    )
    print(f"  change_type: {spec.change_type}")
    print(f"  rationale:   {spec.rationale}")

    # Test V2 path (with Track A config)
    from track_config import load_track
    track_a = load_track(Path(__file__).parent / "tracks" / "track_a_vol_regime.json")

    print("\n── V2 path (Track A — vol regime) ────────────────────────────────")
    spec_a = propose_modification(
        strategy_source=strategy_source,
        current_fitness=0.23,
        fitness_summary=fitness_summary,
        experiment_history=[],
        track_config=track_a,
    )
    print(f"  change_type: {spec_a.change_type}")
    print(f"  rationale:   {spec_a.rationale}")
    if spec_a.new_signals_code:
        print(f"\n  new_signals_code:\n{spec_a.new_signals_code}")
    else:
        print("  new_signals_code: null (parametric only)")
