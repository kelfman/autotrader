"""
agent.py — LLM research agent for autonomous strategy discovery.

The agent reads:
  - Current strategy.py source
  - Current fitness result (per-window breakdown)
  - Recent experiment history (from experiment_log.jsonl)

And outputs a StrategySpec: a structured modification proposal that the
research loop applies to strategy.py, re-evaluates, and accepts or rejects.

The agent uses the Anthropic API with tool_use to guarantee a structured
response — no free-form text parsing required.

ANTHROPIC_API_KEY must be set in the environment.

Usage (standalone smoke-test):
    python agent.py
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

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
                          Must be valid Python using the `ta` library.
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


# ── Anthropic tool definition (mirrors StrategySpec) ─────────────────────────

_PROPOSE_TOOL: dict = {
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


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
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


# ── Agent function ────────────────────────────────────────────────────────────

def propose_modification(
    strategy_source: str,
    current_fitness: float,
    fitness_summary: str,
    experiment_history: list[dict],
    model: str = "claude-opus-4-6",
) -> StrategySpec:
    """
    Call the LLM agent and return a StrategySpec modification proposal.

    Args:
        strategy_source:    Full text of the current strategy.py.
        current_fitness:    Scalar fitness score of the current strategy.
        fitness_summary:    Human-readable per-window breakdown (FitnessResult.summary()).
        experiment_history: Recent experiment dicts from experiment_log.jsonl.
        model:              Anthropic model to use.

    Returns:
        StrategySpec describing the proposed modification.

    Raises:
        RuntimeError: if ANTHROPIC_API_KEY is unset or the API returns no tool call.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)

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

    log.info("Calling %s for modification proposal…", model)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        tools=[_PROPOSE_TOOL],
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

    spec = propose_modification(
        strategy_source=strategy_source,
        current_fitness=0.23,
        fitness_summary=fitness_summary,
        experiment_history=[],
    )

    print("\n── Proposed StrategySpec ──────────────────────────────────────────")
    print(f"  change_type: {spec.change_type}")
    print(f"  rationale:   {spec.rationale}")
    print(f"  params:      {json.dumps(spec.params, indent=16)}")
    if spec.new_signals_code:
        print(f"\n  new_signals_code:\n{spec.new_signals_code}")
    else:
        print("  new_signals_code: null (parametric only)")
