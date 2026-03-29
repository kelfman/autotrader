"""
oversight.py — Research oversight layer for V2 multi-track strategy discovery.

Three layers of oversight (§5.12):
  Layer 1: Anti-convergence constraints (prompt-level, handled by agent.py)
  Layer 2: Automated compliance checks (this module, run every iteration)
  Layer 3: LLM direction reviews (this module, run every N iterations)

Layer 2 checks:
  1. Signal class fidelity — does the entry condition use the track's signal?
  2. Parameter count — is PARAMS within budget?
  3. Archetype detection — does the strategy match a forbidden pattern?
  4. Structural stagnation — has the agent been parameter-tuning too long?
  5. Cross-track convergence — is this strategy converging on another track's?

Layer 3 direction reviews:
  An LLM call (separate from the research agent) that assesses whether the
  agent is exploring the right search space. Returns a DirectionReview with
  recommended actions: continue, force_structural, reset_to_seed, extend_budget,
  or kill_track.

Usage:
    from oversight import run_compliance_checks, run_direction_review

    flags = run_compliance_checks(strategy_source, config, log_entries, other_strategies)
    review = run_direction_review(strategy_source, config, log_entries, flags, model)
"""

from __future__ import annotations

import ast
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field as PydField

from track_config import TrackConfig

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent
_RUNS_DIR = _ROOT / "runs"


# ── Layer 2: Compliance check data structures ─────────────────────────────────

@dataclass
class ComplianceFlag:
    check_name: str
    passed: bool
    severity: str  # "info" | "warning" | "violation"
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


# ── Layer 2: Individual checks ────────────────────────────────────────────────

def _extract_entry_block(source: str) -> str:
    """
    Extract the source lines that contribute to the `entries` variable.

    Heuristic: find the line(s) where `entries` is assigned (entries = ...)
    and return the full assignment including any continuation lines.
    """
    lines = source.split("\n")
    entry_lines: list[str] = []
    capturing = False
    paren_depth = 0

    for line in lines:
        stripped = line.strip()

        if not capturing and re.match(r"entries\s*=", stripped):
            capturing = True
            paren_depth = 0

        if capturing:
            entry_lines.append(stripped)
            paren_depth += stripped.count("(") - stripped.count(")")
            paren_depth += stripped.count("[") - stripped.count("]")
            if paren_depth <= 0 and not stripped.endswith("\\"):
                break

    return "\n".join(entry_lines)


def _check_signal_fidelity(source: str, config: TrackConfig) -> ComplianceFlag:
    """Check 1: Does the entry condition reference the track's signal class?"""
    entry_block = _extract_entry_block(source)

    if not entry_block:
        return ComplianceFlag(
            check_name="signal_fidelity",
            passed=False,
            severity="warning",
            message="Could not find `entries = ...` in strategy source.",
        )

    # Tracks with extra df columns: check if those columns appear in entry logic
    if config.df_columns_extra:
        col_found = any(col in entry_block for col in config.df_columns_extra)
        # Also check derived variables (ratio, funding, eth)
        derived_patterns = {
            "C": ["ratio", "eth_", "relative_strength", "rs_"],
            "D": ["funding", "fr_"],
        }
        track_patterns = derived_patterns.get(config.track_id, [])
        derived_found = any(pat in entry_block for pat in track_patterns)

        if col_found or derived_found:
            return ComplianceFlag(
                check_name="signal_fidelity",
                passed=True,
                severity="info",
                message=f"Entry condition references track-specific signal columns.",
            )
        else:
            return ComplianceFlag(
                check_name="signal_fidelity",
                passed=False,
                severity="warning",
                message=(
                    f"Entry condition does not reference {config.df_columns_extra} "
                    f"or derived signals — Track {config.track_id} signal class "
                    f"may not be the primary entry driver."
                ),
            )

    # Tracks without extra columns (A, B): check for track-specific patterns
    track_specific = {
        "A": ["vol", "realized_vol", "gk", "garman", "volatility", "vol_pct", "vol_rank"],
        "B": ["hour", "dayofweek", "session", "weekday", "day_of_week"],
    }
    patterns = track_specific.get(config.track_id, [])
    if not patterns:
        return ComplianceFlag(
            check_name="signal_fidelity",
            passed=True,
            severity="info",
            message="No signal fidelity check defined for this track.",
        )

    # Check both entry block and the broader compute_signals for signal usage
    found_in_entry = any(pat in entry_block.lower() for pat in patterns)

    if found_in_entry:
        return ComplianceFlag(
            check_name="signal_fidelity",
            passed=True,
            severity="info",
            message=f"Entry condition references track-specific signals ({config.track_id}).",
        )

    # Check if signal is computed anywhere (may feed into entries indirectly)
    found_anywhere = any(pat in source.lower() for pat in patterns)
    if found_anywhere:
        return ComplianceFlag(
            check_name="signal_fidelity",
            passed=False,
            severity="warning",
            message=(
                f"Track {config.track_id} signal class is computed but may "
                f"not be the primary entry driver — appears in function but "
                f"not directly in `entries = ...` block."
            ),
        )

    return ComplianceFlag(
        check_name="signal_fidelity",
        passed=False,
        severity="violation",
        message=(
            f"Track {config.track_id} signal class patterns "
            f"({', '.join(patterns)}) not found in strategy. "
            f"Strategy may have drifted to a generic archetype."
        ),
    )


def _check_param_count(source: str, config: TrackConfig) -> ComplianceFlag:
    """Check 2: Is PARAMS within budget?"""
    match = re.search(r"PARAMS\s*:\s*dict\s*=\s*\{([^}]+)\}", source, re.DOTALL)
    if not match:
        return ComplianceFlag(
            check_name="param_count",
            passed=True,
            severity="info",
            message="Could not parse PARAMS dict; skipping check.",
        )

    param_keys = re.findall(r'"(\w+)"', match.group(1))
    count = len(param_keys)
    max_p = config.max_params

    if count > max_p + 5:
        return ComplianceFlag(
            check_name="param_count",
            passed=False,
            severity="violation",
            message=f"{count} params (max {max_p}+5). Severe parameter proliferation.",
        )
    elif count > max_p:
        return ComplianceFlag(
            check_name="param_count",
            passed=False,
            severity="warning",
            message=f"{count} params (max {max_p}). Over budget.",
        )
    else:
        return ComplianceFlag(
            check_name="param_count",
            passed=True,
            severity="info",
            message=f"{count} params (max {max_p}).",
        )


def _check_archetype(source: str, config: TrackConfig) -> ComplianceFlag:
    """Check 3: Does the strategy match a forbidden entry archetype?"""
    entry_block = _extract_entry_block(source)
    if not entry_block:
        return ComplianceFlag(
            check_name="archetype_detection",
            passed=True,
            severity="info",
            message="Could not extract entry block for archetype check.",
        )

    detected_archetypes: list[str] = []

    # Z-score momentum: rolling mean + rolling std + division
    if ("rolling" in source and ".mean()" in source and ".std()" in source):
        zscore_vars = re.findall(r"(\w+)\s*=.*?rolling.*?mean\(\).*?rolling.*?std\(\)", source, re.DOTALL)
        if zscore_vars or ("zscore" in source.lower() or "z_score" in source.lower()):
            if any(v in entry_block for v in ["zscore", "z_score", "z_"]):
                detected_archetypes.append("z-score momentum")

    # EMA/SMA crossover
    if re.search(r"ema_indicator|sma_indicator|ema_fast.*ema_slow|sma.*cross", source, re.IGNORECASE):
        if any(v in entry_block.lower() for v in ["ema", "sma", "crossover"]):
            detected_archetypes.append("EMA/SMA crossover")

    # Generic momentum: close > close.shift(N) or close > rolling max
    if re.search(r"close\s*>\s*close\.shift", entry_block):
        detected_archetypes.append("generic momentum (close > close.shift)")

    if not detected_archetypes:
        return ComplianceFlag(
            check_name="archetype_detection",
            passed=True,
            severity="info",
            message="No forbidden archetypes detected in entry logic.",
        )

    # Check against forbidden patterns
    violations = []
    for archetype in detected_archetypes:
        for forbidden in config.forbidden_entry_patterns:
            if any(keyword in forbidden.lower() for keyword in archetype.lower().split()):
                violations.append(f"'{archetype}' matches forbidden pattern: '{forbidden}'")

    if violations:
        return ComplianceFlag(
            check_name="archetype_detection",
            passed=False,
            severity="violation",
            message=f"Forbidden archetype detected: {'; '.join(violations)}",
        )

    return ComplianceFlag(
        check_name="archetype_detection",
        passed=True,
        severity="info",
        message=f"Detected archetypes {detected_archetypes} — none match forbidden patterns.",
    )


def _check_stagnation(log_entries: list[dict]) -> ComplianceFlag:
    """Check 4: Has the agent been parameter-tuning too long without structural changes?"""
    if len(log_entries) < 5:
        return ComplianceFlag(
            check_name="structural_stagnation",
            passed=True,
            severity="info",
            message=f"Only {len(log_entries)} entries — too early for stagnation check.",
        )

    recent = log_entries[-5:]
    all_parametric = all(
        e.get("change_type") == "parametric" for e in recent
    )

    if all_parametric:
        return ComplianceFlag(
            check_name="structural_stagnation",
            passed=False,
            severity="warning",
            message=(
                "Agent has been parameter-tuning for 5+ iterations without "
                "structural exploration. Consider forcing a structural change."
            ),
        )

    return ComplianceFlag(
        check_name="structural_stagnation",
        passed=True,
        severity="info",
        message="Recent iterations include structural changes.",
    )


def _tokenize(source: str) -> Counter:
    """Simple whitespace + punctuation tokenizer for code similarity."""
    tokens = re.findall(r"[a-zA-Z_]\w+|[^\s\w]", source)
    return Counter(tokens)


def _cosine_similarity(a: Counter, b: Counter) -> float:
    """Cosine similarity between two token frequency counters."""
    common = set(a.keys()) & set(b.keys())
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    mag_a = sum(v * v for v in a.values()) ** 0.5
    mag_b = sum(v * v for v in b.values()) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _check_cross_track_convergence(
    source: str,
    config: TrackConfig,
    other_strategies: dict[str, str],
) -> ComplianceFlag:
    """Check 5: Is this strategy converging on another track's strategy?"""
    if not other_strategies:
        return ComplianceFlag(
            check_name="cross_track_convergence",
            passed=True,
            severity="info",
            message="No other track strategies available for comparison.",
        )

    my_tokens = _tokenize(source)
    high_sim: list[tuple[str, float]] = []
    threshold = 0.85

    for track_id, other_source in other_strategies.items():
        if track_id == config.track_id:
            continue
        other_tokens = _tokenize(other_source)
        sim = _cosine_similarity(my_tokens, other_tokens)
        if sim > threshold:
            high_sim.append((track_id, sim))

    if high_sim:
        details = ", ".join(f"Track {tid} ({sim:.2f})" for tid, sim in high_sim)
        return ComplianceFlag(
            check_name="cross_track_convergence",
            passed=False,
            severity="warning",
            message=f"Strategy is converging toward: {details} (threshold={threshold}).",
        )

    return ComplianceFlag(
        check_name="cross_track_convergence",
        passed=True,
        severity="info",
        message="Strategy is sufficiently distinct from other tracks.",
    )


# ── Layer 2: Main entry point ─────────────────────────────────────────────────

def run_compliance_checks(
    strategy_source: str,
    config: TrackConfig,
    log_entries: list[dict],
    other_track_strategies: dict[str, str] | None = None,
) -> list[ComplianceFlag]:
    """
    Run all Layer 2 automated compliance checks.

    Args:
        strategy_source:        Full text of the current strategy.py.
        config:                 Track configuration.
        log_entries:            Recent experiment log entries (last ~10).
        other_track_strategies: Dict of {track_id: strategy_source} for other
                                tracks (for cross-track convergence check).

    Returns:
        List of ComplianceFlag results.
    """
    others = other_track_strategies or {}

    flags = [
        _check_signal_fidelity(strategy_source, config),
        _check_param_count(strategy_source, config),
        _check_archetype(strategy_source, config),
        _check_stagnation(log_entries),
        _check_cross_track_convergence(strategy_source, config, others),
    ]

    for f in flags:
        if f.severity == "violation":
            log.warning("COMPLIANCE VIOLATION: %s — %s", f.check_name, f.message)
        elif f.severity == "warning":
            log.info("COMPLIANCE WARNING: %s — %s", f.check_name, f.message)

    return flags


# ── Layer 2: Helper to load other track strategies ────────────────────────────

def load_other_track_strategies(current_track_id: str) -> dict[str, str]:
    """
    Load strategy.py source from all track run directories except the current one.

    Returns:
        Dict of {track_id: strategy_source_text}.
    """
    strategies: dict[str, str] = {}
    if not _RUNS_DIR.exists():
        return strategies

    for run_dir in _RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        # Extract track_id from directory name (track_X_...)
        name = run_dir.name
        if not name.startswith("track_"):
            continue
        parts = name.split("_", 2)
        if len(parts) < 2:
            continue
        track_id = parts[1]
        if track_id == current_track_id:
            continue

        strategy_file = run_dir / "strategy.py"
        if strategy_file.exists():
            try:
                strategies[track_id] = strategy_file.read_text()
            except Exception:
                pass

    return strategies


# ── Layer 3: Direction review data structures ─────────────────────────────────

class DirectionReview(BaseModel):
    """Structured output from the PI direction reviewer (§5.12.5)."""
    signal_class_fidelity: Literal["high", "medium", "low"]
    core_thesis_tested: bool
    drift_diagnosis: str
    course_correction: str | None = None
    recommended_action: Literal[
        "continue", "force_structural", "reset_to_seed", "extend_budget", "kill_track"
    ]
    confidence: float = PydField(ge=0.0, le=1.0)

    def to_dict(self) -> dict:
        return self.model_dump()


_REVIEWER_SYSTEM_PROMPT = """\
You are a research PI reviewing the progress of an autonomous trading strategy \
discovery agent. Your role is NOT to propose strategy changes — it is to assess \
whether the agent is exploring its assigned signal class thesis or has drifted \
to generic patterns.

You will receive:
1. The track's signal class thesis and constraints
2. The current strategy source code
3. Recent experiment history with compliance flags

Produce a structured assessment using the provided tool.

ASSESSMENT GUIDELINES:

signal_class_fidelity:
  - "high": The primary entry signal is derived from the track's assigned signal class.
  - "medium": The signal class is present but used as a secondary filter.
  - "low": The strategy is generic momentum/mean-reversion with minimal connection \
to the assigned signal class.

recommended_action:
  - "continue": Agent is on track, no intervention needed.
  - "force_structural": Inject a course correction requiring a structural change. \
Used when the agent is stuck parameter-tuning a bad architecture.
  - "reset_to_seed": Reset strategy.py to the initial seed and inject course \
correction. Used when drift is severe and incremental correction won't work.
  - "extend_budget": Agent is making genuine progress but needs more iterations.
  - "kill_track": Signal class thesis appears unviable after genuine exploration.

Be specific in drift_diagnosis and course_correction. Name the exact variables, \
patterns, or logic that should change."""


_REVIEWER_TOOL: dict = {
    "name": "submit_direction_review",
    "description": (
        "Submit a structured assessment of the research agent's direction. "
        "Evaluate whether the agent is exploring its assigned signal class "
        "or has drifted to generic patterns."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "signal_class_fidelity": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "How faithfully the strategy implements the assigned signal class.",
            },
            "core_thesis_tested": {
                "type": "boolean",
                "description": "Has the core signal class thesis been tested in isolation?",
            },
            "drift_diagnosis": {
                "type": "string",
                "description": (
                    "What the agent is actually doing vs what it should be doing. "
                    "Be specific about which parts of the strategy are on-thesis "
                    "and which have drifted."
                ),
            },
            "course_correction": {
                "type": ["string", "null"],
                "description": (
                    "Specific instruction to inject into the agent's next prompt. "
                    "Null if no correction needed. Be precise: name variables, "
                    "patterns, or logic that must change."
                ),
            },
            "recommended_action": {
                "type": "string",
                "enum": ["continue", "force_structural", "reset_to_seed", "extend_budget", "kill_track"],
                "description": "What action the research loop should take.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in this assessment (0.0 to 1.0).",
            },
        },
        "required": [
            "signal_class_fidelity", "core_thesis_tested", "drift_diagnosis",
            "recommended_action", "confidence",
        ],
    },
}


# ── Layer 3: Direction review entry point ─────────────────────────────────────

def run_direction_review(
    strategy_source: str,
    config: TrackConfig,
    log_entries: list[dict],
    compliance_flags: list[ComplianceFlag],
    model: str = "claude-sonnet-4-20250514",
) -> DirectionReview:
    """
    Run a Layer 3 LLM direction review.

    Uses a separate LLM call with a reviewer-specific system prompt to
    assess whether the research agent is exploring the assigned signal class.

    Args:
        strategy_source:  Full text of the current strategy.py.
        config:           Track configuration.
        log_entries:      Recent experiment log entries (last review_interval entries).
        compliance_flags: Compliance flags from the most recent iteration.
        model:            Anthropic model for the reviewer (default: Sonnet for cost).

    Returns:
        DirectionReview with assessment and recommended action.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Build the flag summary
    flag_summary_lines = []
    violation_count = sum(1 for f in compliance_flags if f.severity == "violation")
    warning_count = sum(1 for f in compliance_flags if f.severity == "warning")
    flag_summary_lines.append(f"Violations: {violation_count}, Warnings: {warning_count}")
    for f in compliance_flags:
        if f.severity in ("violation", "warning"):
            flag_summary_lines.append(f"  [{f.severity.upper()}] {f.check_name}: {f.message}")
    flag_summary = "\n".join(flag_summary_lines)

    # Build experiment history summary
    history_lines = []
    for entry in log_entries:
        ok = "✓" if entry.get("accepted") else "✗"
        ctype = str(entry.get("change_type", "?"))[:12]
        rationale = str(entry.get("rationale", ""))[:80]
        before = entry.get("fitness_before", 0.0)
        after = entry.get("fitness_after", 0.0)
        n = entry.get("iteration", "?")
        history_lines.append(
            f"  {n:>3}  {ctype:<12}  {before:>+.4f} → {after:>+.4f}  {ok}  {rationale}"
        )
    history_str = "\n".join(history_lines) if history_lines else "(no history)"

    user_message = f"""\
## Track Configuration

Track {config.track_id}: {config.description}
Signal class: {config.signal_class}

Primary signal requirement:
{config.primary_signal_requirement or "(none specified)"}

Forbidden patterns:
{chr(10).join(f"  - {p}" for p in config.forbidden_entry_patterns) or "  (none)"}

Signal class brief:
{config.signal_class_brief[:500]}

## Current Strategy Source

```python
{strategy_source}
```

## Recent Experiment History

{history_str}

## Compliance Flags (latest iteration)

{flag_summary}

---

Assess whether the agent is exploring its assigned signal class thesis or has \
drifted to generic patterns. Use the submit_direction_review tool."""

    log.info(
        "Running PI direction review for Track %s [%s]…",
        config.track_id, model,
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_REVIEWER_SYSTEM_PROMPT,
        tools=[_REVIEWER_TOOL],
        tool_choice={"type": "tool", "name": "submit_direction_review"},
        messages=[{"role": "user", "content": user_message}],
    )

    tool_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_block is None:
        raise RuntimeError(
            "Reviewer returned no tool_use block — unexpected API response. "
            f"Stop reason: {response.stop_reason}"
        )

    review = DirectionReview(**tool_block.input)

    log.info(
        "PI Review: fidelity=%s  thesis_tested=%s  action=%s  confidence=%.2f",
        review.signal_class_fidelity,
        review.core_thesis_tested,
        review.recommended_action,
        review.confidence,
    )
    if review.course_correction:
        log.info("Course correction: %s", review.course_correction[:200])

    return review
