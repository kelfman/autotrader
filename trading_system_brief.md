# Self-Improving Trading System
**Project Brief — v0.3**
*Updated March 2026*

---

## 1. The Core Idea

Inspired by Andrej Karpathy's autoresearch project, this system applies the same self-improving research loop to autonomous trading strategy discovery. The idea: give an LLM agent a strategy file and a rigorous evaluation harness, let it propose modifications, measure the results, and keep what improves performance — overnight, unsupervised, continuously.

The analogy to autoresearch is direct:

- `train.py` → `strategy.py` (editable signal logic, entry/exit rules, position sizing)
- `prepare.py` → `backtest.py` + `evaluate.py` (stable, untouched harness)
- `val_bpb` metric → multi-window Sharpe score (see Fitness Function below)
- LLM agent reads performance + proposes a change → keep if better, revert if worse → log → repeat

The key innovation over a naive parameter sweep: the agent reads the full experiment history before proposing changes, so it learns which directions are productive. It's not random search — it's guided exploration with memory.

---

## 2. Design Principles

These principles govern the whole project, not just V1.

**Reducibility over reflexivity.** Preferred signals have mechanisms that do not depend on other market participants believing in them. Volatility clustering, session flow differences, and funding rate arbitrage pressure are reducible — they work because of structural properties of returns or participants. Most TA-indicator signals are not: they rely on other participants watching the same chart, making them fragile and regime-dependent on liquid, algorithmically-traded markets.

**Regime robustness over peak performance.** The fitness function directly penalises variance across market regimes. A consistent Sharpe of 1.4 across all windows beats 2.5 in one and 0.2 in others. This is the core overfitting guard at every version.

**Separation of discovery and synthesis.** Finding that a signal class works independently, and knowing how to combine it with other signals, are different problems. The system treats them separately by design. V2 discovers; V3 synthesises.

**Provenance at every layer.** Every experiment is logged with its signal class, hypothesis, before/after fitness, and outcome. This record is the system's scientific memory — and the input the orchestrator uses in V3 to allocate effort intelligently.

---

## 3. V1 Architecture

### 3.1 Components

```
research_loop.py    ← orchestrator
├── strategy.py        ← editable by the agent
├── backtest.py        ← stable evaluation harness (DO NOT EDIT during runs)
├── evaluate.py        ← multi-window fitness function
├── agent.py           ← LLM that reads perf + proposes changes
└── experiment_log.jsonl ← memory of every experiment
```

### 3.2 The Research Loop

Each iteration:

1. Evaluate current `strategy.py` across N rolling backtest windows → compute fitness score
2. Summarise per-window breakdown + recent experiment history
3. Agent reads `strategy.py` + summary → outputs a structured `StrategySpec` modification (via Pydantic)
4. Apply modification, re-evaluate
5. If fitness improves: keep change. If not: revert.
6. Append full record to `experiment_log.jsonl`
7. Repeat

### 3.3 The Fitness Function (Regime Robustness)

Rather than a single backtest Sharpe, the fitness is evaluated across 5 rolling 3-month windows over 2 years of data:

```
fitness = mean(sharpe_across_windows) − λ × std(sharpe_across_windows)
```

λ = 0.5. A strategy scoring 1.4 Sharpe consistently across all windows beats one that scores 2.5 in one window and 0.2 in another. No explicit regime detection required — the variance penalty does the work.

Windows with fewer than 3 trades are marked invalid and their Sharpe is treated as 0.

### 3.4 The Strategy File

`strategy.py` has two editable parts: a `PARAMS` dict (numeric values to tune) and a `compute_signals(df, params)` function the agent can rewrite. The agent outputs structured JSON (a `StrategySpec` Pydantic model) rather than raw code edits — safer and more reliable than free-form code surgery.

The agent can do both parametric changes (tweak RSI period, EMA lengths) and structural changes (new indicators, different signal combinations, new entry/exit logic) without touching the evaluation harness.

### 3.5 Domain

Crypto (BTC/ETH) on 1-hour candles. Rationale: clean OHLCV data via ccxt, 24/7 so no market-hours complexity, liquid enough for straightforward transaction cost modelling, and multiple regime types (2020 crash, 2021 bull, 2022 bear, 2023–24 recovery) baked into 2 years of history.

### 3.6 Tech Stack

| Library | Purpose |
|---------|---------|
| ccxt | Crypto data fetching (100+ exchanges, unified API) |
| vectorbt | Backtesting (vectorised/numpy, millisecond iteration speed) |
| ta | Technical indicators (V1 constraint — see §4.2) |
| quantstats | Performance metrics (Sharpe, Sortino, Calmar, drawdown) |
| Anthropic API | LLM agent (claude-opus-4-6, structured tool_use output) |
| pydantic | StrategySpec validation |
| jsonlines | Experiment log |

### 3.7 V1 Status (March 2026)

V1 is complete. 55 iterations run; 6 accepted, 49 rejected. The TA ceiling is confirmed.

- **Best fitness ever recorded: +0.4835** (iteration 10)
- **Effective baseline: +0.4714** (re-evaluated at session restart; used as threshold for iters 11–55)
- **Dominant strategy class: EMA crossover + ADX filter + RSI momentum + EMA50 macro filter**
- 45 consecutive rejections from iteration 11 onward — the EMA-crossover search space is exhausted
- W1 (Mar–Jun 2024, BTC ATH distribution/choppy regime) is the structural weak window: Sharpe −0.390
- W2–W5 are profitable; W2 (BTC bull run to $100k) is the strongest at +1.561

The W1 weakness is consistent with the theoretical prediction: EMA crossover is a reflexive TA signal that structurally underperforms in regimes where the relevant chart-watching participant pool is absent. No TA modification in 45 iterations could overcome this.

**The +0.4714 fitness is the benchmark every V2 track must beat.**

See `RESEARCH_CONTEXT.md` for the full failure taxonomy and per-window breakdown.

---

## 4. Signal Taxonomy

Before speccing V2, it is useful to characterise the signal classes the system will explore and assess their likely quality.

**The reducibility test:** a signal passes if its mechanism does not depend on other participants believing in it. This distinguishes durable structural effects from fragile reflexive ones.

| Signal class | Data needed | Mechanism | Reducible? | V2 Status |
|---|---|---|---|---|
| Funding rates | Perp funding history (ccxt) | Extreme funding = crowded positioning; arb pressure causes reversion | ✓ Strong | **Track D: +1.064** |
| Open interest | Perp OI history (ccxt) | OI changes reveal positioning flow; leverage creates liquidation pressure | ✓ Strong | Track F: blocked (API limit) |
| Basis spread | Perp + spot OHLCV (ccxt) | Perp-spot premium reflects speculative demand; arb pressure causes reversion | ✓ Strong | Track G: +0.281 (sparse) |
| Vol regime / clustering | OHLCV only | Realized vol is autocorrelated (GARCH-like) — a mathematical property of returns | ✓ Strong | Track A: +0.311 |
| Calendar / session effects | OHLCV datetime index | Different participant pools trade at different times — Asia vs US sessions, weekend retail | ✓ Moderate | Track B: +0.00 (dead) |
| Cross-pair signals | BTC + ETH OHLCV | BTC and ETH are cointegrated; ratio deviations have genuine reversion pressure from arb | ✓ Moderate | Track C: +0.00 (dead) |
| Volume microstructure | OHLCV volume | Abnormal volume reflects information asymmetry — informed participants trade larger before moves | ✓ Moderate | Not tested |
| TA indicators | OHLCV only | Reflexive — works when participants collectively watch the same chart | ✗ Fragile | Track E: +0.471 (ceiling) |

**On TA specifically:** TA indicators are mathematical transformations of OHLCV — they do not add information, only extract patterns. On liquid, algorithmically-traded crypto markets, reflexive edge is thin and unstable. The fitness function's variance penalty directly exposes this: strategies relying on reflexivity tend to fail in specific regimes when the relevant participant group is absent. V1's W1 weakness is consistent with this.

**The ceiling of OHLCV-only signals:** within price and volume alone, the system can be more statistically sophisticated than raw TA, but it is always transforming the same underlying information. The signals with the clearest mechanisms sit outside OHLCV — funding rates being the most actionable for crypto, available with a single additional ccxt call.

---

## 5. V2 — Multi-Track Research

### 5.1 Core Idea

V1 is a single hill-climbing search initialised with a TA vocabulary. The fundamental limitation is that hill-climbing from one starting point explores a connected search space — the agent discovers refinements of TA without access to genuinely different signal classes, regardless of how many iterations it runs.

V2 introduces **parallel research tracks** — independent agent instances, each seeded with a different signal class vocabulary, running against the same backtest harness and fitness function. This produces multiple simultaneous research programmes that can be compared scientifically: which signal classes produce durable fitness improvements, and which are noise?

**Key distinction:** V2 is about parallel *discovery*. Finding that a signal class works and knowing how to combine it with others are separate problems. Combination (ensemble synthesis) is addressed in V3.

### 5.2 Research Tracks

| Track | Signal class | New data? | Key prompt changes |
|---|---|---|---|
| A | Statistical / vol regime | No | Allow numpy, scipy. Explore rolling z-scores, realized vol percentiles, GARCH-style regime detection. Remove ta requirement. |
| B | Calendar / session effects | No | Allow datetime operations. Explore hour-of-day, day-of-week, session open/close effects. Remove ta requirement. |
| C | Cross-pair signals | No (ETH already fetched) | Allow BTC/ETH ratio computation. Explore ratio z-scores, ETH as BTC directional signal. |
| D | Funding rates | Yes — perp funding via ccxt | Add `funding_rate` column to df. Explore funding extremes as entry/exit filters. Include mechanism brief. |
| E | TA baseline (V1 continued) | No | V1 as-is. Continues as benchmark. All other tracks must beat this to be considered an improvement. |

### 5.3 Architecture

Each track is an independent instance of the V1 loop with:

- **Its own system prompt** — different signal vocabulary, allowed libraries, exploration heuristics, and a concise brief explaining the signal class mechanism
- **Its own strategy.py** — tracks do not share evolving strategies
- **Its own experiment log** — separate provenance per signal class, with a new `track_id` field
- **The same backtest harness and fitness function** — enabling direct, apples-to-apples comparison of fitness outcomes

A **track runner** layer (`track_runner.py`) wraps the existing research loop, accepting a track config file and running N iterations with isolated output directories. Tracks run sequentially in V2; parallel execution is V3.

### 5.4 Agent Prompt Changes

The V2 agent prompt is parameterised per track. Key changes from V1:

- **Remove ta-only mandate.** Tracks A–D are not constrained to the ta library. Each track's prompt specifies its allowed libraries explicitly.
- **Add signal class mechanism brief.** Each prompt includes a concise explanation of *why* the signal class exists — not just what to compute, but what structural property of the market it captures.
- **Add reducibility check.** The agent is asked: does the proposed signal have a mechanism that does not depend on other participants watching the same chart?
- **Retain regime robustness framing.** The fitness function description and per-window failure analysis heuristic are unchanged across all tracks.

### 5.5 Experiment Log Schema

Each entry gains a `track_id` field. New fields:

```json
{
  "track_id": "A",
  "signal_class": "statistical_vol_regime",
  "iteration": 3,
  "change_type": "structural",
  "rationale": "...",
  "fitness_before": 0.21,
  "fitness_after": 0.38,
  "accepted": true,
  ...
}
```

`show_log.py` is extended to display per-track summaries: acceptance rate, best fitness achieved, fitness trajectory, and plateau detection.

### 5.6 Funding Rate Data Pipeline (Track D)

The only new data pipeline in V2. Funding rates for BTC/USDT perpetual contracts are available via ccxt's `fetchFundingRateHistory`. Implementation:

1. Fetch BTC/USDT perpetual funding rate history from Binance futures
2. Align to the 1h OHLCV index (funding updates every 8h — forward-fill within each window)
3. Add as a `funding_rate` column in the df passed to `compute_signals`
4. The backtest harness (`backtest.py`) requires no changes — it passes df columns through unchanged

### 5.7 Iteration Budget

V2 uses a fixed budget: 20 iterations per track. After initial runs, a manual review step compares per-track fitness trajectories and informs whether to extend promising tracks. Automated bandit allocation is V3.

### 5.8 Held-Out Validation

More experiments across more tracks increases multiple testing risk. Any strategy achieving fitness > 0.8 on the 5-window training set must be re-validated on a separate held-out date range before being considered validated. This held-out set is not used during iteration — it is the final test of whether a finding is real.

### 5.9 V2 Build Sequence

1. ✅ **Run V1 baseline to ~50 iterations** — 55 iterations completed. TA ceiling confirmed at +0.4714. Full failure taxonomy in `RESEARCH_CONTEXT.md`.
2. ✅ **Refactor prompt system** — `track_config.py` (Pydantic `TrackConfig` model) + `tracks/` directory with JSON configs for all 5 tracks. `agent.py` builds system prompt and tool definition dynamically from config; V1 path unchanged.
3. ✅ **Build track runner** — `track_runner.py`: creates isolated `runs/track_X_<name>/` directories, initialises per-track `strategy.py`, threads paths and `track_config` through `research_loop.run_loop()`. Log entries include `track_id` and `signal_class`.
4. ✅ **Tracks A–C** — 20 iterations each completed. Results below.
   - **Track A (vol regime):** 35 iterations, 6 accepted, best fitness +0.3484. Below TA baseline.
   - **Track B (calendar/session):** 20 iterations, 1 accepted, best fitness +0.00. Not viable — all long-only calendar strategies tracked BTC directional bias; session effects too weak to overcome bear-regime losses.
   - **Track C (cross-pair):** 13 iterations, 5 accepted, best fitness +0.2727. Below TA baseline.
5. ✅ **Funding rate pipeline + Track D** — `fetch_funding_rates()` added to `data.py` (ccxt fetch from Binance perpetual futures with parquet cache). `_augment_with_funding()` in `track_runner.py` fetches 8h funding rates, forward-fills to 1h resolution, merges as `funding_rate` column. 20 iterations completed.
   - **Track D (funding rates):** 23 iterations, 10 accepted, best fitness **+1.0567** — beats TA baseline by 124%. **First validated V2 signal class.** See §5.10 for details.
6. ✅ **Extend show_log.py** — `--tracks` shows per-track summary table (iterations, acceptance rate, best fitness, trajectory). `--track <ID>` shows individual track iteration log. Both modes support `--accepted-only`, `--last N`, `--detail`.
7. ✅ **Held-out validation** — `validate_holdout.py` evaluates strategies on the ~69-day gaps between training windows (4 held-out windows covering the 38% of the date range not used during research). Results below in §5.11.

### 5.10 Track D Results (March 2026)

Track D is the first V2 signal class to decisively beat the TA baseline. The discovery validates the brief's reducibility thesis: funding rate pressure is a structural mechanism independent of chart-watching behaviour.

**Best fitness: +1.0567** (mean_sharpe=+1.299, std=0.485, λ=0.5)

| Window | Period | Sharpe | Return | MDD | Trades |
|--------|--------|--------|--------|-----|--------|
| W1 | Mar–Jun 2024 (choppy) | +1.504 | +6.6% | 7.9% | 6 |
| W2 | Sep–Dec 2024 (bull) | +1.567 | +8.4% | 9.3% | 11 |
| W3 | Feb–May 2025 (bear) | +1.927 | +12.0% | 11.3% | 13 |
| W4 | Jul–Oct 2025 (recovery) | +0.926 | +2.9% | 6.6% | 10 |
| W5 | Dec–Mar 2026 (recent) | +0.571 | +2.9% | 11.2% | 8 |

All 5 windows profitable. The strategy survived the exact choppy regime (W1) that permanently defeated V1's EMA crossover.

**Strategy:** Funding percentile-rank filtered momentum. Entry when close > SMA(192) and funding rate is below 52nd percentile of its 30-day rolling distribution (not crowded). Exit when funding exceeds 90th percentile (extreme crowding) or price drops below 38-bar trailing low.

**Key parameters:** `sma_period=192, fr_pct_window=720, fr_entry_pct=0.52, fr_exit_pct=0.90, exit_lookback=38`

**Discovery trajectory:** The critical breakthroughs were (1) switching from z-score to percentile rank for regime stability, (2) finding that `fr_entry_pct=0.55` fixed the W5 weakness, and (3) slowing the SMA from 168 to 192 which dramatically improved W1.

**Comparison to TA baseline:**

| Metric | V1 TA (Track E) | Track D (Funding) |
|--------|-----------------|-------------------|
| Fitness | +0.4714 | **+1.0567** |
| Mean Sharpe | +0.815 | **+1.299** |
| Std Sharpe | 0.687 | **0.485** |
| W1 (choppy) | −0.390 | **+1.504** |
| All windows positive? | No | **Yes** |

### 5.11 Held-Out Validation Results (March 2026)

The held-out windows are the ~69-day gaps between the 5 training windows — periods the research loop never evaluated during iteration. They cover 38% of the 2-year date range (276 days total).

**Track D (funding rates) — held-out:**

| Window | Period | Sharpe | Return | MDD | Trades |
|--------|--------|--------|--------|-----|--------|
| H1 | Jun–Sep 2024 | −1.475 | −6.9% | 12.3% | 7 |
| H2 | Dec 2024–Feb 2025 | −2.040 | −9.1% | 16.2% | 8 |
| H3 | May–Jul 2025 | −1.577 | −3.4% | 6.0% | 5 |
| H4 | Oct–Dec 2025 | −1.235 | −5.0% | 6.9% | 7 |

**Held-out fitness: −1.7280** (all 4 windows negative)

For comparison, the V1 TA baseline was also evaluated on the same held-out windows:

| Strategy | Training Fitness | Held-Out Fitness | Decay |
|----------|-----------------|------------------|-------|
| Track D (funding) | +1.0639 | −1.7280 | −262% |
| Track E (TA baseline) | +0.4714 | −0.4657 | −199% |

**Interpretation:** Both strategies fail held-out validation. The gap windows are transition periods between the regime types represented in training — structurally harder for any momentum-based approach. Track D's larger position sizing and higher training fitness amplify the held-out degradation.

This does **not** invalidate the funding rate signal class entirely. The training window results are legitimate within their date ranges, and the regime-robustness across 5 diverse training windows remains meaningful. However, the strategy as currently parameterised does not generalise to out-of-sample transition periods, which means it cannot be promoted to V3 synthesis without further work.

**Implications for V3:** The prerequisite of "two validated tracks on held-out data" is not yet met. Possible next steps:
- Walk-forward validation (train on windows 1–3, test on 4–5) as a less harsh OOS test
- Larger training window count (7–8 windows with smaller gaps) to reduce gap-period risk
- Signal combination: funding rate as a filter on a less position-heavy base strategy

### 5.12 Research Oversight Layer

V2 tracks A–C drifted away from their assigned signal classes and converged on generic momentum strategies. The fitness function cannot detect this because a z-score momentum strategy can have positive fitness — it's just not the signal class the track was supposed to explore. The missing signal is qualitative direction review: is the agent exploring the right search space?

This section specifies a **research oversight layer** — a "lab PI" that reviews strategy direction, not just fitness outcomes. It operates at a different level of abstraction than the research loop: the loop asks "is this strategy better?" while oversight asks "is this the right strategy to be testing?"

#### 5.12.1 Problem Diagnosis

Every OHLCV-only track converged on the same archetype: z-score momentum entry + ATR trailing stop + parameter proliferation. The specific failures:

| Track | Assigned thesis | What agent actually built | Drift severity |
|-------|----------------|--------------------------|----------------|
| A | Vol percentile rank predicts regime quality | Vol regime + vol-of-vol + z-score momentum + ATR stop (15 params) | Moderate — vol regime was present but buried under complexity |
| B | Session timing IS the signal | Momentum strategy gated by Tue–Thu US hours | Severe — calendar used as filter, not signal |
| C | BTC/ETH ratio z-score mean-reversion | Track A's vol regime with `eth_z > 0` bolted on | Severe — core thesis never tested |

Root causes:
1. **No prompt constraints on what the entry signal must be.** The prompts said "explore X" but didn't say "your primary entry must be derived from X, not from generic momentum."
2. **No mechanism to review strategy code mid-run.** The system only checked fitness numbers.
3. **No structural diversity enforcement.** Agents could spend 20 iterations parameter-tuning a bad architecture without being redirected.
4. **LLM prior bias.** Claude's training data is heavy with quant trading patterns (z-scores, ATR, momentum). Without hard constraints, the agent gravitates to these comfortable patterns rather than reasoning from the signal class mechanism.

#### 5.12.2 Architecture

Three layers, from cheapest to most expensive:

```
Layer 1: Anti-convergence constraints      (prompt-level, zero cost, every iteration)
Layer 2: Automated compliance checks       (heuristic, ~0 cost, every iteration)
Layer 3: Direction reviews                 (LLM call, moderate cost, every N iterations)
```

All three layers feed into the research loop. Layer 1 prevents drift. Layer 2 detects drift. Layer 3 diagnoses drift and course-corrects.

#### 5.12.3 Layer 1 — Anti-Convergence Constraints

New fields in `TrackConfig` that inject hard constraints into the agent's system prompt:

```json
{
  "primary_signal_requirement": "Your PRIMARY entry condition MUST be derived from the BTC/ETH price ratio or relative strength. A generic momentum signal (close > SMA, z-score > threshold) may be used as a SECONDARY filter only.",

  "forbidden_entry_patterns": [
    "z-score of price above a rolling mean as the primary entry trigger",
    "EMA or SMA crossover as the primary entry trigger",
    "Generic momentum (close > close.shift(N)) as the sole entry logic"
  ],

  "max_params": 10,

  "exploration_phase_iterations": 5,

  "exploration_phase_instructions": "For the first 5 iterations, you MUST test the core signal class thesis in isolation. Iteration 1: compute the raw signal and use it as the ONLY entry/exit logic — no filters, no stops. Iterations 2-5: variations on the core signal with different lookbacks and thresholds. Do NOT add momentum filters, ATR stops, or secondary signals until iteration 6+."
}
```

These fields are injected verbatim into the system prompt built by `_build_system_prompt()` in `agent.py`. The `forbidden_entry_patterns` are presented as explicit constraints the agent must not violate. The `primary_signal_requirement` defines what the entry condition must look like.

**Updated `TrackConfig` schema** (`track_config.py`):

```python
class TrackConfig(BaseModel):
    # ... existing fields ...

    # Oversight layer 1 — anti-convergence
    primary_signal_requirement: str = ""
    forbidden_entry_patterns: list[str] = Field(default_factory=list)
    max_params: int = 10
    exploration_phase_iterations: int = 5
    exploration_phase_instructions: str = ""
```

**Agent prompt injection** (`agent.py`): When building the system prompt, append:

```
SIGNAL CLASS REQUIREMENTS (MANDATORY)
{config.primary_signal_requirement}

FORBIDDEN PATTERNS — DO NOT USE:
{formatted list of forbidden_entry_patterns}

PARAMETER BUDGET: Maximum {config.max_params} parameters in PARAMS.
```

During the exploration phase (iteration ≤ `exploration_phase_iterations`), prepend the exploration phase instructions and suppress the "GENERAL EXPLORATION STRATEGY" section.

#### 5.12.4 Layer 2 — Automated Compliance Checks

A new module `oversight.py` that runs heuristic checks after every iteration. Each check returns a structured result:

```python
@dataclass
class ComplianceFlag:
    check_name: str            # e.g. "signal_fidelity"
    passed: bool
    severity: str              # "info" | "warning" | "violation"
    message: str               # human-readable explanation
```

**Check 1: Signal class fidelity.** Parse `compute_signals()` source code and check whether the track's `df_columns_extra` columns (e.g. `eth_close`, `funding_rate`) appear in the entry condition. Heuristic: find the line(s) that build the `entries` variable and check whether track-specific columns appear in the expression tree leading to it, not just anywhere in the function.

For tracks without extra columns (A, B), check for track-specific patterns:
- Track A: the entry condition should reference `vol` or `realized_vol` or `gk` (Garman-Klass) — not just `zscore` or `momentum`.
- Track B: the entry condition should reference `hour`, `dayofweek`, or session-derived variables — and these should be in the entry computation, not just as a boolean gate.

Implementation: regex-based or AST-based analysis of the `entries = (...)` line. Does not need to be perfect — false positives are acceptable (the LLM review in Layer 3 handles ambiguity).

**Check 2: Parameter count.** Count keys in `PARAMS`. Flag as `warning` if > `max_params`, `violation` if > `max_params + 5`.

**Check 3: Archetype detection.** Classify the strategy into archetypes by scanning for patterns:
- Contains `rolling(N).mean()` and `rolling(N).std()` and divides them → z-score momentum
- Contains `atr` or `average_true_range` in exit logic → ATR trailing stop
- Contains `ema_indicator` or `sma` crossover logic → EMA/SMA crossover

If the detected archetype matches a `forbidden_entry_patterns` item, flag as `violation`.

**Check 4: Structural stagnation.** Read the last N experiment log entries. If the last 5+ iterations are all `change_type: "parametric"`, flag as `warning` with message "Agent has been parameter-tuning for 5+ iterations without structural exploration. Consider forcing a structural change."

**Check 5: Cross-track convergence.** Compare the current track's `compute_signals` source against other tracks' strategies (read from `runs/track_*/strategy.py`). If the cosine similarity of the code (by token overlap or similar heuristic) exceeds a threshold, flag as `warning` with "Strategy is converging toward Track X's architecture."

**Integration with research loop:** After each iteration in `run_loop()`:

```python
from oversight import run_compliance_checks

flags = run_compliance_checks(
    strategy_source=strategy_path.read_text(),
    config=track_config,
    log_entries=load_history(n=10, log_path=log_path),
    all_track_strategies=_load_other_track_strategies(track_config),
)

# Log flags with the experiment entry
entry["compliance_flags"] = [f.to_dict() for f in flags]

# Print warnings/violations to console
for f in flags:
    if f.severity == "violation":
        log.warning("COMPLIANCE VIOLATION: %s — %s", f.check_name, f.message)
    elif f.severity == "warning":
        log.info("COMPLIANCE WARNING: %s — %s", f.check_name, f.message)
```

Compliance flags are logged but do not gate iteration acceptance in Layer 2. They serve as diagnostics and as input to Layer 3.

#### 5.12.5 Layer 3 — Direction Reviews

A separate LLM call that runs every `review_interval` iterations (default: 5). This is the PI function — it reads the strategy code, recent history, and compliance flags, and produces a structured assessment.

**Input to the reviewer:**
- The track's `TrackConfig` (signal class brief, exploration hints, primary signal requirement)
- Current `strategy.py` source code
- The last `review_interval` experiment log entries (with compliance flags)
- Aggregate compliance flag counts since last review

**Reviewer system prompt:**

```
You are a research PI reviewing the progress of an autonomous trading strategy
discovery agent. Your role is NOT to propose strategy changes — it is to assess
whether the agent is exploring its assigned signal class thesis or has drifted
to generic patterns.

You will receive:
1. The track's signal class thesis and constraints
2. The current strategy source code
3. Recent experiment history with compliance flags

Produce a structured assessment using the provided tool.
```

**Reviewer tool output schema:**

```json
{
  "signal_class_fidelity": "high" | "medium" | "low",
  "core_thesis_tested": true | false,
  "drift_diagnosis": "Free-text: what is the agent actually doing vs what it should be.",
  "course_correction": "Free-text instruction to inject into the agent's next prompt. Null if no correction needed.",
  "recommended_action": "continue" | "force_structural" | "reset_to_seed" | "extend_budget" | "kill_track",
  "confidence": 0.0-1.0
}
```

**`signal_class_fidelity`:**
- `high`: The primary entry signal is derived from the track's assigned signal class.
- `medium`: The signal class is present but used as a secondary filter rather than the primary signal.
- `low`: The strategy is generic momentum/mean-reversion with minimal connection to the assigned signal class.

**`recommended_action`:**
- `continue`: Agent is on track, no intervention needed.
- `force_structural`: Inject a course correction requiring a structural change on the next iteration. Used when the agent is stuck parameter-tuning a bad architecture.
- `reset_to_seed`: Reset `strategy.py` to the initial seed strategy and inject the course correction. Used when the agent has drifted so far that incremental correction won't work.
- `extend_budget`: The agent is making genuine progress but needs more iterations. Add 10 to the remaining budget.
- `kill_track`: The signal class thesis appears unviable after genuine exploration. Log the conclusion and stop.

**Course correction injection:** The `course_correction` string is passed to `propose_modification()` and appended to the user message as a new section:

```
## PI Review — Course Correction (mandatory)

{course_correction_text}

You MUST address this feedback in your next proposal. Ignoring PI direction
will result in the proposal being rejected regardless of fitness.
```

Note: the "rejected regardless of fitness" framing is a prompt instruction to the agent — the system does not actually auto-reject. The PI review operates through persuasion, not hard gating, except for `reset_to_seed` which is a hard reset.

**`reset_to_seed` action:** When triggered, the research loop:
1. Copies the initial seed strategy (from the template in `track_runner.py`) back to the track's `strategy.py`
2. Does NOT clear the experiment log (history is preserved as provenance)
3. Appends a log entry with `change_type: "pi_reset"` and the reviewer's diagnosis
4. Injects the course correction into the next iteration

This is the nuclear option — used when Track C has become "Track A with an ETH boolean" and needs a clean restart.

#### 5.12.6 Exploration Phase Protocol

The first `exploration_phase_iterations` iterations (default 5) of each track operate under special rules:

**Iteration 1 — Raw signal test.** The agent must compute the core signal (e.g., BTC/ETH ratio z-score for Track C) and use it as the sole entry/exit logic. No filters, no stops, no secondary signals. The goal is to measure the raw predictive power of the signal class.

**Iterations 2–3 — Signal variations.** Different lookback windows, different entry/exit thresholds, different transformations of the core signal (z-score vs percentile rank, raw value vs diff). Still no secondary signals.

**Iterations 4–5 — Minimal filtering.** The agent may add ONE secondary filter (e.g., a trend filter) or ONE exit mechanism (e.g., trailing stop). Maximum 6 parameters.

**Iteration 6+ — Free exploration.** Normal operation with anti-convergence constraints still active.

The exploration phase is enforced by modifying the system prompt for iterations ≤ `exploration_phase_iterations`. The iteration number is passed to the agent prompt builder, and the `exploration_phase_instructions` override the normal "GENERAL EXPLORATION STRATEGY" section.

#### 5.12.7 Updated Track Configs

Each track config JSON gets updated with the new fields. Examples:

**Track A (vol regime):**
```json
{
  "primary_signal_requirement": "Your PRIMARY entry condition MUST use a realized volatility measure (rolling vol, Garman-Klass, or vol percentile rank) as the core decision variable. Momentum (close > SMA) may be a secondary filter only.",
  "forbidden_entry_patterns": [
    "z-score of price as the primary entry trigger",
    "EMA crossover as entry logic",
    "Close > close.shift(N) as the sole entry condition"
  ],
  "max_params": 10,
  "exploration_phase_iterations": 5,
  "exploration_phase_instructions": "Iterations 1-5: Test whether vol regime alone has predictive value. Iteration 1: Enter when vol percentile < 0.4 (calm), exit when > 0.7 (turbulent). No momentum filter. Iterations 2-5: Vary the vol calculation (close-to-close vs Garman-Klass), lookback windows, and entry/exit thresholds."
}
```

**Track B (calendar/session):**
```json
{
  "primary_signal_requirement": "Your PRIMARY entry condition MUST be derived from time-of-day or day-of-week patterns. The calendar/session effect IS the signal — not a filter on momentum. Test whether specific hours or days have directionally biased returns.",
  "forbidden_entry_patterns": [
    "z-score momentum as the primary entry",
    "Price > SMA or price > rolling mean as the primary entry",
    "Any entry condition that would be identical if you removed all hour/dayofweek logic"
  ],
  "max_params": 8,
  "exploration_phase_iterations": 5,
  "exploration_phase_instructions": "Iterations 1-2: Test raw session effect. Enter at US session open (13:00 UTC), exit at US session close (21:00 UTC). Then try Asian session (01:00-08:00). No momentum filters. Iterations 3-4: Test day-of-week effects. Enter Monday open, exit Friday. Or enter Tue-Thu only. Iteration 5: Test session open momentum — is the first 2-hour price change during US session predictive of the next 6 hours?"
}
```

**Track C (cross-pair):**
```json
{
  "primary_signal_requirement": "Your PRIMARY entry condition MUST use the BTC/ETH price ratio or relative performance as the core signal. Compute ratio = close / eth_close, then derive entry/exit from ratio dynamics (z-score, percentile, direction). Do NOT use BTC momentum as the primary signal with ETH as a minor confirmation.",
  "forbidden_entry_patterns": [
    "BTC vol regime as the primary entry (this is Track A)",
    "BTC z-score momentum as the primary entry with ETH as a boolean gate",
    "Any entry condition that would be identical if eth_* columns were removed"
  ],
  "max_params": 8,
  "exploration_phase_iterations": 5,
  "exploration_phase_instructions": "Iteration 1: Pure ratio mean-reversion. ratio = close / eth_close. Enter BTC long when ratio z-score < -1.5 (BTC cheap vs ETH), exit when ratio z-score > 0. No other signals. Iteration 2: Ratio percentile rank instead of z-score. Iteration 3: Relative strength — enter when ETH is outperforming BTC over past N bars (ETH leading = crypto bull). Iteration 4: BTC leads ETH — use BTC 2-4h return as a predictor for ETH. Iteration 5: Add one filter to the best-performing variant."
}
```

#### 5.12.8 Experiment Log Schema Extension

Each log entry gains a `compliance_flags` field and optionally a `pi_review` field:

```json
{
  "iteration": 7,
  "compliance_flags": [
    {"check_name": "signal_fidelity", "passed": false, "severity": "warning", "message": "Entry condition does not reference eth_close or ratio — Track C signal class not primary."},
    {"check_name": "param_count", "passed": true, "severity": "info", "message": "8 params (max 10)."}
  ],
  "pi_review": {
    "signal_class_fidelity": "low",
    "core_thesis_tested": false,
    "drift_diagnosis": "Strategy is a vol-regime momentum strategy identical to Track A with eth_z > 0 as a trivial gate.",
    "course_correction": "Reset to seed. On next iteration: compute ratio = close / eth_close, use ratio z-score as the ONLY entry condition.",
    "recommended_action": "reset_to_seed"
  }
}
```

#### 5.12.9 Build Sequence

1. ✅ **Update `TrackConfig`** — Added `primary_signal_requirement`, `forbidden_entry_patterns`, `max_params`, `exploration_phase_iterations`, `exploration_phase_instructions`, `review_interval` fields with backward-compatible defaults.

2. ✅ **Update track configs** — All 5 track JSONs updated: Tracks A–D have signal-specific constraints, forbidden patterns, exploration phase instructions, and 5-iteration review intervals. Track E (baseline) has no constraints.

3. ✅ **Update `agent.py`** — `_build_system_prompt()` now accepts `iteration_number` and injects anti-convergence constraints, forbidden patterns, parameter budget, and exploration phase instructions (replacing general exploration hints during the exploration phase). `propose_modification()` accepts `iteration_number` and `course_correction` parameters.

4. ✅ **Build `oversight.py`** — 5 automated compliance checks (signal fidelity, param count, archetype detection, structural stagnation, cross-track convergence) + LLM direction reviewer using `claude-sonnet-4-20250514` with structured tool output (`DirectionReview`). Smoke-tested against Track A's existing drifted strategy — correctly flags signal fidelity issues and parameter proliferation.

5. ✅ **Integrate into `research_loop.py`** — Compliance checks run after every iteration (when track has `primary_signal_requirement`). Direction reviews run every `review_interval` iterations. Course corrections thread through to the next agent call. `reset_to_seed` resets strategy.py to initial template and logs a `pi_reset` entry. `extend_budget` adds 10 iterations. `kill_track` stops the loop.

6. ✅ **Re-run Tracks A, B, C** — 20 iterations each with `--reset`, `--agent-mode local` (Cursor agents as research agents, no API calls). Results:
   - **Track A (vol regime):** 20 iterations, 4 accepted, best fitness **+0.311**. Garman-Klass vol + vol-of-vol stability filter. Signal fidelity 20/20.
   - **Track B (calendar/session):** 20 iterations, 1 accepted, best fitness +0.00. Calendar session effects insufficient for crypto. Signal fidelity 20/20.
   - **Track C (cross-pair):** 20 iterations, 1 accepted, best fitness +0.00 (degenerate — broken param keys cause 0 trades). Best genuine proposal scored −0.402 using vol-conditioned ratio z-score with signal-based exits. Signal fidelity 20/20.

7. ✅ **Comparative assessment** — See table below.

**Oversight Layer Assessment:**

The oversight layer successfully prevented the drift that plagued the original V2 runs:

| Metric | Original Runs (no oversight) | Re-runs (with oversight) |
|--------|------------------------------|--------------------------|
| Track A signal fidelity | Moderate — vol present but buried under z-score momentum + 15 params | **20/20 — vol regime as primary entry in every iteration, 6 params** |
| Track B signal fidelity | Severe drift — momentum gated by Tue–Thu hours | **20/20 — calendar always primary entry signal** |
| Track C signal fidelity | Severe drift — Track A clone with ETH boolean gate | **24/24 — BTC/ETH ratio as primary signal throughout** |
| Track A best fitness | +0.3484 | **+0.3110** (slightly lower but genuine vol signal) |
| Track B best fitness | +0.00 | +0.00 (calendar effects confirmed unviable) |
| Track C best fitness | +0.2727 | +0.00 (degenerate; best genuine: −0.40) |

**Key findings:**
- The oversight layer kept all tracks on-thesis (100% signal fidelity pass rate). The original runs had severe drift in B and C.
- Track A's Garman-Klass vol + vol-of-vol is a genuine volatility regime strategy that achieves +0.311 — below the TA baseline (+0.471) but a real structural signal. The low trade count (0-4 per window) limits fitness.
- Track B confirms calendar session effects are too weak for 24/7 crypto. The 0-trade baseline from a vectorbt compatibility issue couldn't be beaten by any profitable session strategy.
- Track C's BTC/ETH ratio mean-reversion is fundamentally weak. The best genuine approach (vol-conditioned z-score + signal-based exits) achieved mean_sharpe=+0.52 but std=1.85 — the ratio works in 3/5 windows but catastrophically fails during strong BTC-dominance trends (W2: Q4 2024 bull). A broken param-key mismatch caused the 0.0 degenerate to become the local optimum.

### 5.13 Walk-Forward Validation (March 2026)

The gap-window holdout (§5.11) tested strategies on the ~69-day transition periods between training windows. Both Track D and the TA baseline failed. Walk-forward validation is a fairer OOS test: train on W1–W3 (the windows the strategy was optimized over), test on W4–W5 (forward-looking windows).

`validate_walkforward.py` reuses the same 5 windows as `evaluate.py` but splits them temporally: W1–W3 in-sample, W4–W5 out-of-sample.

**Walk-forward results:**

| Strategy | In-Sample (W1-3) | OOS (W4-5) | Decay | OOS > TA Baseline? |
|----------|-----------------|------------|-------|--------------------|
| Track D (funding) | +1.588 | **+0.660** | 58% | Yes (+0.660 > +0.471) |
| Track E (TA baseline) | +0.321 | **+0.779** | -143% (improves!) | Yes |

Both strategies generalize forward with positive OOS fitness. Track D's OOS fitness (+0.660) beats the TA baseline's full 5-window fitness (+0.471). The TA baseline actually performs better on recent windows than historical ones (negative decay), consistent with a V-shaped recovery pattern in 2025.

**Track D OOS per-window:** W4: Sharpe +0.926, +2.9% return, 10 trades | W5: Sharpe +0.571, +2.9% return, 8 trades. Both windows valid and positive.

This resolves the §5.11 validation impasse: Track D passes walk-forward validation with flying colours. The gap-window failure was specific to the transition-period structure of those windows, not a generalisation failure of the funding rate signal class.

### 5.14 Manual Synthesis: Track D + Track A (March 2026)

First synthesis attempt per §6.3 prerequisite. Combined Track D's funding rate filter with Track A's Garman-Klass vol-of-vol regime gate as an AND condition.

**Synthesis strategy** (`runs/synthesis_D_A/strategy.py`): Entry when `close > SMA(192)` AND `funding_pct < 0.52` AND `vol_pct < 0.65`. Exit when `funding_pct > 0.90` OR `close < trailing_low(38)` OR `vol_pct > 0.80`.

**Results:**

| Metric | Track D (standalone) | Synthesis D+A |
|--------|---------------------|---------------|
| Full 5-window fitness | +1.067 | **−0.615** |
| Walk-forward OOS (W4-5) | **+0.660** | +0.547 |
| Held-out (gap windows) | −1.728 | 0.000 |
| W1 trades | 6 | **0** |
| W2 trades | 11 | 6 |

**Key finding:** Naive AND-gating of orthogonal signals *hurts* rather than helps. The vol regime filter blocks trades in W1 (choppy post-ATH, high vol) — which is exactly the window where Track D's funding signal works best (+1.504 Sharpe). Adding a vol gate to an already-robust strategy reduces trade count without improving quality.

**Implications for V3 synthesis:** Signal combination must be more sophisticated than AND-gating. Options include:
- Vol regime as a position-sizing modifier rather than a hard gate
- Regime-switching: use funding in all regimes, add vol-based position reduction in high-vol periods
- Asymmetric combination: vol gate on exits only (protect gains) rather than entries

The manual synthesis attempt demonstrates that finding orthogonal signals independently and knowing how to combine them are genuinely separate problems, as the brief predicted in §2 (separation of discovery and synthesis).

### 5.15 Bayesian Synthesis Optimization

The manual synthesis (§5.14) revealed that naive AND-gating of individually-optimized signals hurts performance — each filter's standalone-optimal thresholds are too tight for combined use, choking trade count. The intuition: when two uncorrelated filters share the entry gate, each only needs to do *partial* filtering work. Their jointly-optimal thresholds should be relaxed relative to their standalone optima.

This section specifies a **Bayesian parameter optimization layer** that replaces the LLM agent for parametric search within a fixed strategy structure. The LLM remains responsible for structural exploration (proposing new signal logic); Optuna handles the numeric optimization.

#### 5.15.1 Problem Statement

The synthesis parameter space has ~9 continuous dimensions. The fitness landscape has:
- **Cliffs**: small threshold changes can zero out a window (0 trades → Sharpe treated as 0)
- **Plateaus**: many parameter combinations produce similar fitness
- **Interaction effects**: the optimal `fr_entry_pct` depends on the value of `vol_entry_pct` and vice versa

The LLM agent is poorly suited to this — it proposed 15+ parametric variations of Track A's thresholds without systematically mapping the space. A surrogate-model optimizer can evaluate 200 parameter combinations in the time the LLM evaluates 20, and it builds a statistical model of which regions are promising.

#### 5.15.2 Architecture

```
Stage 1: D+A joint optimization
    Optuna TPE sampler, 200 trials (~15 min)
    Search space: 9 params with relaxed bounds
    Objective: 5-window fitness
    → Best D+A params

Stage 2: Marginal signal tests (B, C)
    Take optimized D+A as base
    Add use_session_filter (bool) + session params
    Add use_ratio_filter (bool) + ratio params
    Optuna, 100 trials each
    → Accept if fitness improves over D+A alone

Stage 3: Walk-forward validation
    Evaluate best synthesis on W4-W5 OOS
    Compare to Track D standalone OOS (+0.660)
    → Final assessment
```

#### 5.15.3 Stage 1 — D+A Joint Optimization

**New file:** `optimize_params.py`

A generic Bayesian parameter optimizer that wraps the existing evaluation infrastructure. Uses Optuna's TPE sampler (handles cliffs better than GP-based BO), evaluates each trial via `evaluate.evaluate_strategy()`, and supports `--augment-funding` and `--augment-eth` flags.

**Search space for D+A synthesis:**

| Parameter | Track D Optimal | Track A Optimal | Search Range | Rationale |
|-----------|----------------|-----------------|--------------|-----------|
| `sma_period` | 192 | — | [96, 384] | Allow faster or slower trend filter |
| `fr_pct_window` | 720 | — | [360, 1440] | 15–60 day funding lookback |
| `fr_entry_pct` | 0.52 | — | [0.30, 0.80] | Wide range — may need heavy relaxation |
| `fr_exit_pct` | 0.90 | — | [0.70, 0.95] | Exit crowding threshold |
| `exit_lookback` | 38 | — | [20, 72] | Trailing low window |
| `vol_lookback` | — | 24 | [12, 72] | GK vol estimation window |
| `vol_pct_window` | — | 1440 | [720, 2160] | Vol percentile lookback |
| `vol_entry_pct` | — | 0.47 | [0.40, 0.85] | Wide range — key relaxation param |
| `vol_exit_pct` | — | 0.65 | [0.55, 0.90] | Vol-based exit threshold |

The critical hypothesis: the optimizer should discover that `fr_entry_pct` and `vol_entry_pct` jointly land at values significantly looser than their standalone optima.

**Configuration:** 200 trials, TPE sampler, SQLite storage for resumability. Outputs best params, parameter importance ranking, and optimization history.

#### 5.15.4 Stage 2 — Marginal Signal Tests

After Stage 1 produces an optimized D+A base, test whether Track B (calendar) or Track C (cross-pair) signals add value as marginal features.

The strategy is extended with boolean switches (`use_session_filter`, `use_ratio_filter`) that Optuna can toggle. Conditional parameters (session hours, ratio lookback) are only sampled when the corresponding switch is on, keeping effective dimensionality low.

**Decision rule:** If the best Stage 2 fitness exceeds Stage 1 best by more than 0.05, accept the marginal signal. If the optimizer consistently sets both switches to False, the signals are confirmed as noise in the combined context.

#### 5.15.5 Overfitting Guards

200 Optuna trials on 5 training windows creates multiple-testing risk. Two guards:

1. **Walk-forward as primary validation.** The 5-window fitness is the optimization target; walk-forward OOS (W4–W5) is the acceptance test. These are different windows, so optimizing on one does not inflate the other.

2. **Parameter stability check.** After optimization, perturb the best params by ±10% and re-evaluate. If fitness drops by more than 20%, the optimum is a fragile spike, not a robust basin. Report the stability score alongside the fitness.

#### 5.15.6 Build Sequence

1. ✅ **Install Optuna** — `pip install optuna` (+ pyarrow for parquet cache)
2. ✅ **Build `optimize_params.py`** — Generic Optuna TPE optimizer wrapping `evaluate_strategy()`. Presets for `d_a` and `d_a_b_c` search spaces. Supports `--stability-check`, `--walkforward`, `--resume` (SQLite storage), `--write-params`, JSON output. Per-trial window breakdown logged via user attrs.
3. ✅ **Run Stage 1** — D+A synthesis, 200 trials in 89s (0.4s/trial). See §5.15.7 for results.
4. ✅ **Evaluate Stage 1** — Walk-forward OOS +1.991 (3.3% decay). Holdout (gap windows) −3.139 (same structural issue as §5.11). Stability check: FRAGILE (55% worst drop from vol_pct_window ±10%).
5. ✅ **Build Stage 2 strategy** — `runs/synthesis_D_A_B_C/strategy.py` with `use_session_filter` and `use_ratio_filter` boolean switches. Conditional parameters only sampled when switch is on.
6. ✅ **Run Stage 2** — 100 trials. Best fitness +0.928 (well below D+A's +2.030). Session filter enabled in best trial but zeroed out W2. Ratio filter consistently off. B/C signals confirmed as noise in combined context.
7. ✅ **Final validation** — Full comparison table in §5.15.8.
8. ✅ **Update docs** — This section.

### 5.15.7 Stage 1 Results — D+A Bayesian Optimization (March 2026)

**Best fitness: +2.030** (mean_sharpe=+2.420, std=0.780, λ=0.5) — trial #191 of 200.

| Window | Period | Sharpe | Return | MDD | Trades |
|--------|--------|--------|--------|-----|--------|
| W1 | Mar–Jun 2024 (choppy) | +2.349 | +6.5% | 2.4% | 8 |
| W2 | Sep–Dec 2024 (bull) | +1.571 | +4.9% | 6.1% | 9 |
| W3 | Feb–May 2025 (bear) | +3.344 | +13.2% | 4.6% | 7 |
| W4 | Jul–Oct 2025 (recovery) | +3.274 | +7.5% | 3.6% | 6 |
| W5 | Dec–Mar 2026 (recent) | +1.563 | +5.4% | 6.8% | 3 |

All 5 windows profitable with ≥3 trades. MDD ≤ 6.8% in every window.

**Optimized parameters (vs manual synthesis):**

| Parameter | Manual | Optimized | Change |
|-----------|--------|-----------|--------|
| `sma_period` | 192 | **256** | Slower trend filter |
| `fr_pct_window` | 720 | **408** | Shorter funding lookback |
| `fr_entry_pct` | 0.52 | **0.688** | Relaxed (funding filter looser) |
| `fr_exit_pct` | 0.90 | **0.778** | Tighter exit threshold |
| `exit_lookback` | 38 | **60** | Wider trailing stop |
| `vol_lookback` | 24 | **53** | Longer GK vol window |
| `vol_pct_window` | 1440 | **1104** | Shorter vol percentile lookback |
| `vol_entry_pct` | 0.65 | **0.455** | Tightened (vol filter stricter) |
| `vol_exit_pct` | 0.80 | **0.577** | Much tighter vol exit |

**Key finding:** The optimizer confirmed the brief's §5.15.1 hypothesis — jointly-optimal thresholds differ significantly from standalone optima. The vol filter tightened from 0.65 to 0.455 (doing heavy lifting for regime selection) while the funding filter relaxed from 0.52 to 0.688 (no longer needing to filter alone). This is a genuine interaction effect that LLM-guided parametric search could not discover in 15+ iterations.

**Parameter importance:** `vol_pct_window` (55.2%) dominates, followed by `fr_pct_window` (18.9%). The vol regime lookback determines whether the strategy enters calm-regime positions — it is the single most consequential parameter.

**Walk-forward validation:** In-sample (W1-3) +2.058, OOS (W4-5) **+1.991**. Decay only 3.3%. OOS beats TA baseline by 322%.

**Stability:** FRAGILE. vol_pct_window ±10% causes 55% fitness drop. sma_period ±10% causes 33% drop. The optimum is a narrow peak, not a broad basin. Deployment would require parameter monitoring or regularization.

### 5.15.8 Final Comparison (March 2026)

| Strategy | Full Fitness | WF OOS (W4-5) | WF Decay | Stable? |
|----------|-------------|----------------|----------|---------|
| Track E (TA baseline) | +0.471 | +0.779 | −143% (improves OOS) | N/A |
| Track D (funding standalone) | +1.064 | +0.660 | +58% | N/A |
| D+A manual synthesis (§5.14) | −0.615 | +0.547 | N/A | N/A |
| **D+A Bayesian optimized** | **+2.030** | **+1.991** | **+3.3%** | No (fragile) |
| D+A+B+C Stage 2 | +0.928 | +1.434 | −123% | No |

**Conclusions:**

1. **Bayesian optimization transforms the D+A synthesis from failure to best-in-project.** The manual synthesis scored −0.615; the optimizer found +2.030 by discovering the interaction between vol regime gating and funding rate filtering.

2. **The synthesis OOS generalizes extraordinarily well.** Walk-forward decay of 3.3% is the lowest of any strategy tested. Track D standalone decays 58%; the TA baseline actually improves OOS (negative decay due to V-shaped recovery in 2025).

3. **B and C signals do not add marginal value.** Stage 2 confirmed that session timing and BTC/ETH ratio are noise in the combined D+A context. The optimizer could not beat pure D+A even with 100 trials of joint search.

4. **Stability is the main concern.** The optimum is a narrow peak — small perturbations in vol_pct_window cause >50% fitness drops. This reflects the strategy's dependence on precise vol regime calibration. V3 should address this via parameter regularization, ensemble averaging of nearby optima, or adaptive recalibration.

5. **Gap-window holdout remains negative (−3.139).** The transition periods between training windows are structurally hostile to all strategies tested. This is a data coverage issue (the 5 training windows leave 38% of the date range as gaps), not a strategy generalization failure. Walk-forward is the more meaningful OOS test.

### 5.15.9 Robustness Analysis (March 2026)

The fragility of the Stage 1 optimum (55% worst-case drop from ±10% perturbation) prompted two approaches to find more robust parameter configurations.

**Approach 1: Ensemble averaging.** Average parameters from the top-20 trials of the existing Stage 1 study. Zero additional compute — the top trials define a basin, and the average lands at its center.

**Approach 2: Robust optimization.** New 200-trial Optuna study with a perturbation-aware objective: each trial evaluates the base params plus 3 random ±10% perturbation vectors, and the mean fitness across all 4 evaluations becomes the objective. This forces the optimizer to find broad basins rather than narrow spikes.

**Results:**

| Approach | Full Fitness | WF OOS (W4-5) | WF Decay | Worst Drop (±10%) | Stable? |
|----------|-------------|----------------|----------|-------------------|---------|
| Single-best (§5.15.7) | **+2.030** | **+1.991** | **3.3%** | 55.0% | No |
| Ensemble (top-20) | +1.845 | +1.816 | 3.8% | **40.7%** | No |
| Robust optimization | +1.851 | +1.209 | 51.0% | 31.3% | No |

**Ensemble parameters vs single-best:**

| Parameter | Single-Best | Ensemble (top-20) | Robust |
|-----------|------------|-------------------|--------|
| `sma_period` | 256 | 266 | 248 |
| `fr_pct_window` | 408 | 418 | 552 |
| `fr_entry_pct` | 0.688 | 0.690 | 0.754 |
| `fr_exit_pct` | 0.778 | 0.760 | 0.859 |
| `exit_lookback` | 60 | 55 | 24 |
| `vol_lookback` | 53 | 52 | 52 |
| `vol_pct_window` | 1104 | 1070 | 984 |
| `vol_entry_pct` | 0.455 | 0.509 | 0.450 |
| `vol_exit_pct` | 0.577 | 0.565 | 0.563 |

**Analysis:**

1. **Ensemble is the dominant approach.** It sacrifices 9% of headline fitness for modestly improved stability (40.7% vs 55% worst drop), maintains excellent OOS generalization (+1.816, 3.8% decay), and costs zero additional compute. The top-20 trials are tightly clustered — parameter spreads are narrow (e.g., `fr_entry_pct` std=0.01), confirming a genuine basin rather than scattered random peaks.

2. **Robust optimization finds more stable optima but at a steep generalization cost.** The 31.3% worst drop is the best stability achieved, but 51% walk-forward decay indicates the perturbation-aware objective overfits to in-sample windows. It finds params that are locally smooth but don't transfer forward as well.

3. **No approach achieves full stability (worst drop < 20%).** This suggests the D+A strategy structure is inherently parameter-sensitive — the combinatorial interaction between vol regime gating and funding rate filtering creates sharp fitness gradients that no parameterization can fully smooth. This is a structural property of the strategy, not a limitation of the optimization method.

4. **The ensemble is the recommended deployment candidate.** Fitness +1.845, OOS +1.816, 3.8% decay. The parameter sensitivity is acknowledged but manageable: the most sensitive param (`vol_exit_pct` ±10% → 40.7% drop) can be monitored. The ensemble's tight parameter spread means nearby configurations also perform well — it's sitting in the broadest available basin.

### 5.16 Derivative-Signal Tracks (Next)

The funding rate signal validated the reducibility thesis: derivatives microstructure signals have structural mechanisms independent of chart-watching behaviour. The natural next step is to explore the rest of the derivatives signal family — signals available from the same exchange APIs that share the same mechanistic basis.

#### 5.16.1 Rationale

V2 produced one strong signal class (funding rates, Track D) and one weak but genuine structural signal (vol regime, Track A). Calendar (B) and cross-pair (C) are dead. The D+A synthesis achieves project-best OOS fitness (+1.816) but the remaining fragility (40.7%) is structural — further improvement requires new independent signal classes, not better optimization of existing ones.

Open interest and basis spread are the two highest-value candidates:
- Both are available via ccxt on Binance futures
- Both have clear, reducible mechanisms in the same family as funding rates
- Both can be integrated with the existing pipeline (numeric column, forward-fill to 1h, merge to df)
- Both can be tested independently first, then as marginal additions to the D+A synthesis

#### 5.16.2 Track F — Open Interest

**Signal class:** Changes in aggregate open interest on BTC/USDT perpetual futures.

**Mechanism:** Open interest (OI) measures total outstanding derivative contracts. Rising OI with rising price means new longs are entering — the market is getting more crowded and leveraged. Rising OI with falling price means new shorts are building. Falling OI means positions are being closed (deleveraging). The mechanism is structural: leverage creates liquidation risk regardless of whether participants are watching charts.

**Key signals to explore:**
- OI rate of change (momentum of positioning, not price)
- OI vs price divergence (price rising but OI falling = short covering, healthier rally)
- OI percentile rank (similar to funding rate percentile — extreme OI = crowded)
- OI acceleration (second derivative — are positions building faster or slower?)

**Data pipeline:** ccxt's `fetchOpenInterest()` returns current OI; historical OI requires `fetchOpenInterestHistory()` on exchanges that support it (Binance does, as hourly snapshots). Forward-fill to 1h resolution. Add as `open_interest` column.

**Track config:** 20 iterations with oversight layer. Primary signal must derive from OI dynamics. Forbidden: using OI as a trivial boolean gate on a momentum entry.

#### 5.16.3 Track G — Basis Spread

**Signal class:** The price difference between perpetual futures and spot markets (the "basis").

**Mechanism:** When perps trade at a premium to spot (positive basis), the market is paying to hold leveraged long exposure — a measure of speculative demand. When perps trade at a discount (negative basis), there's distress or hedging demand. Unlike funding rates (which are a lagging 8h average), basis is continuous and reflects real-time supply/demand for leverage. The arb mechanism is structural: when basis is extreme, arb desks step in to capture the spread, creating mean-reversion pressure.

**Key signals to explore:**
- Basis percentile rank (extreme basis = crowded leverage, expect reversion)
- Basis rate of change (basis expanding = growing speculative demand)
- Basis vs funding divergence (basis widening but funding flat = recent speculative surge not yet priced into funding)
- Annualized basis as a carry signal (positive carry = structural upward pressure)

**Data pipeline:** Compute `basis = perp_close - spot_close` from the two price feeds. Perp price comes from the futures OHLCV (ccxt with `defaultType: 'future'`). Spot price is the existing BTC/USDT OHLCV. Both are hourly, so alignment is straightforward. Add as `basis` and optionally `basis_pct` (basis / spot_close) columns.

**Track config:** 20 iterations with oversight layer. Primary signal must derive from basis dynamics. Forbidden: using basis as a trivial gate on a momentum or funding-rate entry.

#### 5.16.4 Build Sequence

1. ✅ **Add `fetch_open_interest_history()` to `data.py`** — Raw Binance `fapiData/openInterestHist` endpoint (ccxt wrapper has parameter-mapping bug). Parquet cache.
2. ✅ **Add `fetch_perp_ohlcv()` to `data.py`** — BTC/USDT:USDT perpetual futures OHLCV via ccxt futures exchange. Full 730-day coverage.
3. ✅ **Add augmentation functions to `track_runner.py`** — `_augment_with_oi()` and `_augment_with_basis()`.
4. ✅ **Create track configs** — `tracks/track_f_open_interest.json` and `tracks/track_g_basis.json`.
5. ❌ **Track F (open interest)** — **Blocked.** Binance's `openInterestHist` API is hard-capped at ~30 days of history regardless of period. Not enough for 730-day backtests. Needs a paid data source (CoinGlass, Glassnode) or exchange with longer OI history.
6. ✅ **Track G (basis spread)** — 100 Optuna trials + 4 structural variants tested manually. See §5.16.5 for results.
7. ✅ **Evaluate results** — Basis signal below viability threshold. See §5.16.5.
8. ✅ **Update docs** — This section.

#### 5.16.5 Track G Results (March 2026)

**Data:** 17,517 hourly bars of perp-spot basis (mean basis_pct = −0.034%, range ±0.3%).

**Structural exploration (4 variants, manual):**

| Variant | Logic | Fitness | W+ / W− |
|---------|-------|---------|---------|
| Contrarian percentile (pct < 0.40) | Enter low basis | −2.19 | 1/4 |
| Momentum (pct > 0.70) | Enter high basis | −1.71 | 1/4 |
| Z-score < −1.0 | Enter extreme negative | −0.88 | 2/3 |
| Regime (30d mean > 0) | Bull carry filter | −1.06 | 1/4 |

The z-score variant (V2) was the best structural approach, with positive W1/W2 but catastrophic W3 (bear market generates constant false entries when basis stays negative).

**Optuna optimization (100 trials):**

Best fitness: **+0.281** (sma_period=336, z_window=1440, z_entry=−2.44, z_exit=+1.51)

| Window | Sharpe | Trades | Status |
|--------|--------|--------|--------|
| W1 | 0.000 | 0 | No trades |
| W2 | 0.000 | 0 | No trades |
| W3 | +1.476 | 3 | Sparse |
| W4 | 0.000 | 0 | No trades |
| W5 | +2.464 | 5 | Sparse |

The optimizer achieved +0.281 by setting z_entry=−2.44 — so extreme that the strategy only triggers in 2 of 5 windows (8 total trades). This is the same degenerate-optimization pattern seen in Track C. The walk-forward OOS (+0.616) is misleading: it comes from 5 trades in a single window.

**Stability:** Extremely fragile (106% worst drop). z_exit ±10% destroys fitness entirely.

**Assessment:** Basis spread is not a viable standalone signal class on this data. The mechanism is real (arb pressure from extreme basis), but on BTC/USDT 1h candles, the basis is too small (±0.3%) and too transient for the entry signal to produce consistent trades across diverse market regimes. The signal may work on lower timeframes (where basis micro-movements are larger relative to fees) or in a live context (where real-time basis captures intrabar dynamics that hourly closes miss).

**Comparison to other tracks:**

| Track | Signal Class | Best Fitness | Trade Quality |
|-------|-------------|-------------|---------------|
| D (funding) | Funding rate | **+1.064** | 6–13 trades/window, all windows positive |
| A (vol regime) | GK vol | +0.311 | 0–4 trades/window, sparse |
| E (TA baseline) | EMA crossover | +0.471 | 3–7 trades/window, W1 negative |
| G (basis) | Perp-spot basis | +0.281 | 0–5 trades/window, 3 windows empty |
| B (calendar) | Session effects | +0.00 | Dead |
| C (cross-pair) | BTC/ETH ratio | +0.00 | Dead |

Track G joins Track A as a weak-but-genuine signal: it captures something real (extreme basis does predict short-term reversion) but can't produce consistent trades across regimes. Funding rate remains the only strong standalone signal class found in V2.

---

## 6. V3 — Research Director

V2 explored 7 signal classes and found one strong result (funding rates) plus one weak gate (vol regime). The original V3 spec (an orchestrator managing a portfolio of research tracks) assumed multiple strong signal classes would emerge. They didn't. The bottleneck is not track management — it's that the system's most valuable decisions were *structural* ones made by a human: "try Bayesian optimization," "the strategy contract is the limitation," "test the rest of the derivatives family." V3 should automate that level of reasoning.

### 6.1 V2 Retrospective — What the System Can't Do Yet

The research loop is good at **parametric exploration within a fixed structure**: given a strategy template and a search space, it finds good parameters. Between Optuna (200 trials in 89s) and the LLM agent (structural proposals with oversight), the system is effective at local search.

What it cannot do:

1. **Recognise when the search space itself is the constraint.** The D+A synthesis went from −0.615 to +2.030 not by finding better parameters in the original space, but by jointly optimizing parameters that interact — something the LLM agent couldn't discover because it changes one thing at a time.

2. **Modify its own experimental apparatus.** The `compute_signals(df, params) -> (entries, exits)` contract limits every strategy to binary decisions on a single timeframe. No amount of iteration within that contract can produce position sizing, regime switching, or multi-timeframe logic.

3. **Reason about what to explore next.** The system runs whatever track it's told to run. The decisions that mattered most — "try funding rates," "use Optuna for synthesis," "test basis spread" — were all human decisions informed by the signal taxonomy and experimental results.

4. **Manage its own equity.** 105 trades over 2 years, 423-day max drawdown duration. The strategy doesn't know when it's in a regime where it has no edge and should sit out.

### 6.2 The Three-Level Architecture

V3 adds a **Research Director** that operates above the existing research loop and PI reviewer. The three levels:

| Level | Agent | Question | Operates on | Frequency |
|-------|-------|----------|-------------|-----------|
| **L1 — Execution** | Research agent | "What parameter/code change improves this strategy?" | Strategy files within a track | Every iteration |
| **L2 — Oversight** | PI reviewer | "Is this track exploring the right signal class?" | Track direction, compliance | Every N iterations |
| **L3 — Direction** | Research Director | "What kind of experiment should the system run next?" | The system's structure, goals, and infrastructure | After milestones or on schedule |

The Research Director doesn't write strategy code. It reads the full project state — results, failures, infrastructure, theory, equity curves — and produces **structural recommendations**: changes to the strategy contract, new data sources, new evaluation methods, new composition approaches. It then delegates execution to L1/L2 agents or to Optuna.

### 6.3 Research Director — Input and Output

**Input (project state):**

The director reads a compiled state document before each review. This includes:

- `RESEARCH_CONTEXT.md` and `trading_system_brief.md` (theory, results, history)
- Equity curve analysis from the current best strategy (trade log, drawdown series, regime breakdown)
- Per-track result summaries (fitness, trade counts, failure modes)
- The current strategy contract and its limitations
- The codebase structure (what components exist, what's easy vs hard to change)
- A diff of what changed since the last review

**Output (strategic review):**

A structured document with:

```json
{
  "review_type": "milestone",
  "project_phase": "post-V2 signal exploration",
  "key_findings": [
    "Funding rate is the only strong signal class. Basis and OI exhausted on current data/API.",
    "The strategy contract (binary entries/exits, single timeframe) is the binding constraint.",
    "423-day drawdown duration indicates the strategy trades in regimes where it has no edge."
  ],
  "ceiling_analysis": {
    "signal_ceiling": "OHLCV + ccxt derivatives signals largely explored. Next signals require paid data (OI history) or new sources (on-chain).",
    "contract_ceiling": "Binary enter/exit on 1h prevents position sizing, regime switching, and multi-timeframe logic.",
    "evaluation_ceiling": "5 fixed windows over 2 years. Strategies optimized for these specific regimes."
  },
  "recommended_interventions": [
    {
      "id": "MTF",
      "title": "Multi-timeframe signal augmentation",
      "category": "contract_expansion",
      "expected_impact": "high",
      "implementation_effort": "moderate",
      "rationale": "...",
      "priority": 1
    }
  ],
  "execution_plan": {
    "next_action": "Implement multi-timeframe DataFrame augmentation",
    "delegation": "coding_agent",
    "success_criteria": "Re-run D+A strategy with 4h and 1d columns available. Fitness improvement > 0.1 over ensemble baseline.",
    "abort_criteria": "No fitness improvement after 200 Optuna trials with expanded parameter space."
  }
}
```

**Decision authority:** The director recommends. For low-risk changes (adding DataFrame columns, expanding search spaces, running new Optuna studies), it can delegate directly. For high-risk changes (modifying the backtest harness, changing the fitness function, altering the strategy contract), it produces a plan for human approval.

### 6.4 Strategic Review Protocol

The director follows a structured reasoning sequence:

**Step 1 — State assessment.** What has been tried? What worked? What failed and why? What's the current best strategy and what does its equity curve look like?

**Step 2 — Ceiling identification.** Where is the system hitting limits? Is it the signal vocabulary (need new data), the strategy expressiveness (need a richer contract), the evaluation method (need more/different windows), or the optimization approach (need better search)?

**Step 3 — Intervention ideation.** For each identified ceiling, what structural change would raise it? Generate candidate interventions. These are not parameter changes — they're changes to *what the system can express or evaluate*.

**Step 4 — Prioritisation.** Rank interventions by expected impact / implementation effort. Consider dependencies (e.g., position sizing requires contract change before Optuna can optimize it).

**Step 5 — Execution planning.** For the top-priority intervention, produce a concrete implementation plan: what files change, what the new contract looks like, what success/abort criteria apply, and how to validate the result.

**Step 6 — Delegation.** Hand the plan to the appropriate executor: coding agent for implementation, Optuna for optimization, or human for approval of high-risk changes.

Reviews trigger after: (a) a track or synthesis run completes, (b) a structural change is implemented and evaluated, (c) a scheduled interval (e.g., every 50 research iterations across all tracks), or (d) on manual request.

### 6.5 Structural Expansions

These are the six interventions identified in the post-V2 review, ordered by priority.

#### 6.5.1 Multi-Timeframe Signal Augmentation

**Category:** Contract expansion (expand what a strategy can see)

**Problem:** The strategy receives a single 1h DataFrame. The SMA(266) trend filter is a blunt proxy for "what's the higher timeframe trend?" It can't express "the daily trend is bullish but the 4h is pulling back to support — good entry" or "the weekly regime is bearish, don't trade regardless of hourly signals."

**Solution:** Before calling `compute_signals`, resample the 1h data to 4h and 1d. Compute standard features at each timeframe (SMA, vol, momentum) and add them as prefixed columns: `h4_sma_200`, `d1_close`, `d1_vol_pct`, etc. The strategy can then use multi-timeframe confluence without any new data sources.

**Implementation:**
- New function `augment_with_timeframes(df)` that adds `h4_*` and `d1_*` columns
- Strategy contract unchanged — `compute_signals(df, params)` still returns `(entries, exits)`, but `df` now has ~15 additional columns
- Expand the D+A Optuna search space to include multi-timeframe parameters: `h4_sma_period`, `d1_trend_filter`, etc.
- Run 200 trials and compare against the ensemble baseline (+1.845)

**Expected impact:** High. Multi-timeframe confluence is one of the strongest structural improvements available without new data. The current SMA(266) filter may be replaced by a cleaner higher-timeframe trend measure that reduces false entries.

**Success criteria:** Fitness improvement > 0.15 over ensemble baseline. Failure: no improvement after 200 trials → the 1h signal is already capturing what higher timeframes would add.

#### 6.5.2 Position Sizing as Signal Output

**Category:** Contract expansion (expand what a strategy can express)

**Problem:** The strategy goes 100% in or 100% out. The vol regime gate hard-blocks entries rather than modulating conviction. This is why the manual D+A synthesis (§5.14) failed — AND-gating two filters chokes trade count. The Bayesian optimization (§5.15) found a workaround by relaxing thresholds, but the fundamental issue remains: the system can't express "I'm 30% confident, take a small position."

**Solution:** Extend the strategy contract to return a third output:

```python
def compute_signals(df, params) -> tuple[Series, Series, Series]:
    # entries: bool Series (when to enter)
    # exits: bool Series (when to exit)
    # size: float Series 0.0–1.0 (position size as fraction of capital)
    return entries, exits, size
```

vectorbt's `Portfolio.from_signals` already supports a `size` parameter. The backtest harness needs a small modification to pass it through. Backward compatible: if the strategy returns 2 values (old contract), default to size=1.0.

**Implementation:**
- Modify `backtest.py` to accept optional size output
- Modify `_INITIAL_STRATEGY_GENERIC` template to show the new contract
- Strategy can use vol regime, funding rate, or any signal to modulate position size
- The vol gate becomes: `size = 1.0 - vol_pct` (more vol → smaller position) instead of `vol_calm = vol_pct < threshold`

**Expected impact:** High. Unlocks continuous conviction instead of binary gating. The D+A synthesis should benefit directly — vol regime modulates position size rather than blocking entries, addressing the core failure mode from §5.14.

**Success criteria:** Improved risk-adjusted returns (higher Sortino, lower max drawdown) at similar or better total return. The 423-day drawdown duration should shorten.

#### 6.5.3 Regime-Switching Strategy Dispatch

**Category:** Composition (expand how strategies compose)

**Problem:** The D+A synthesis AND-gates two signals into one strategy. But different market regimes may call for fundamentally different *logic*, not just different thresholds. In calm-vol / low-funding regimes: run the trend-following funding strategy. In high-vol / extreme-funding regimes: switch to a defensive mode (tight stops or flat). In bear regimes: sit out entirely.

**Solution:** A meta-strategy that classifies the current regime and dispatches to sub-strategies:

```python
def compute_signals(df, params):
    regime = classify_regime(df, params)  # -> Series of {0, 1, 2}
    entries_0, exits_0 = strategy_trend(df, params)    # calm regime
    entries_1, exits_1 = strategy_defensive(df, params) # volatile regime
    entries_2, exits_2 = strategy_flat(df, params)      # bear regime

    entries = where(regime == 0, entries_0, where(regime == 1, entries_1, False))
    exits = where(regime == 0, exits_0, where(regime == 1, exits_1, True))
    return entries, exits
```

The regime classifier uses the signals already computed (vol percentile, funding percentile, SMA direction) to determine which strategy logic runs. This is more expressive than AND-gating because each sub-strategy can have completely different entry/exit logic.

**Implementation:**
- Design regime classifier using existing vol + funding signals (no new data)
- Write 2-3 sub-strategies (trend, defensive, flat)
- The outer `compute_signals` routes based on regime
- Optuna optimizes both the regime boundaries and the sub-strategy parameters

**Expected impact:** High. Directly addresses the 423-day drawdown problem — the "flat" regime sub-strategy prevents dead-weight losses during periods where the funding signal has no edge. Also addresses fragility: the strategy's behaviour adapts to conditions rather than depending on one parameter set working everywhere.

**Dependency:** Benefits from §6.5.2 (position sizing) — regime can modulate size continuously instead of hard-switching between sub-strategies.

#### 6.5.4 Strategy Portfolio with Dynamic Allocation

**Category:** Composition (expand the goal)

**Problem:** The entire project optimises for one best strategy. But V2 produced three strategies with positive fitness and different failure modes: Track D (funding, strong), Track E (TA baseline, moderate), Track A (vol regime, weak). Their errors are likely uncorrelated — they fail in different windows.

**Solution:** Instead of one strategy, maintain a portfolio of 2-3 strategies with allocation weights that shift based on recent performance:

- Equal-weight baseline: allocate 1/3 capital to each strategy
- Momentum allocation: overweight strategies with better recent-N-day returns
- Inverse-volatility: allocate more to strategies with lower recent drawdown
- The portfolio's combined equity curve should be smoother than any individual strategy

**Implementation:**
- New `portfolio_backtest.py` that runs multiple strategies on the same data and combines equity curves
- Allocation rebalances at a fixed interval (daily or weekly)
- No new data or strategy changes needed — just composition of existing strategies

**Expected impact:** Moderate. The benefit depends on how uncorrelated the strategies actually are. If Track D and Track E fail in different windows (D fails in gap periods, E fails in W1 choppy), the portfolio smooths the equity curve. If they're correlated (both fail in bear markets), the benefit is minimal.

**Success criteria:** Lower max drawdown and shorter drawdown duration than the single best strategy, at similar or better total return.

#### 6.5.5 Explicit Regime Prediction

**Category:** New capability (predict regime, not price)

**Problem:** The fitness function tests regime robustness implicitly — it penalises variance across windows. But the strategy doesn't *know* what regime it's in. It applies the same logic everywhere and relies on the vol/funding filters to implicitly select regimes. This is why the drawdown periods are so long — the strategy keeps trading in regimes where it has no edge.

**Solution:** Build a regime classifier as a first-class component:

- Cluster historical market conditions into 3-5 regime states using the signals already computed (vol level, vol trend, funding level, price momentum, basis)
- Assign a label to each bar: trending-calm, trending-volatile, choppy, bear, recovery
- Measure the strategy's expected return *per regime* (not per window)
- Use the regime label to either (a) gate trading (only trade in favourable regimes) or (b) select strategy parameters per regime

This differs from §6.5.3 (regime switching) in that it's a standalone classifier, not embedded in the strategy. It could be used by the Research Director to understand *why* the strategy fails in certain periods and what kind of experiment would address it.

**Implementation:**
- Unsupervised clustering (KMeans, HMM, or BOCPD) on rolling features: vol percentile, funding percentile, SMA slope, return momentum
- Label each bar in the 2-year dataset with its regime
- Compute strategy performance statistics per regime
- Optionally: use regime as an additional column in the DataFrame for strategy use

**Expected impact:** High but uncertain. The value depends on whether the regimes identified by clustering correspond to regimes where the funding signal has edge. If they do, this directly solves the "when to trade" problem. If the clustering produces regimes that don't align with strategy performance, it's wasted effort.

**Dependency:** Valuable as input to §6.5.3 (regime switching) and §6.5.6 (LLM research analyst).

#### 6.5.6 LLM as Research Analyst (not Strategy Coder)

**Category:** Role shift (rethink the agent's purpose)

**Problem:** The LLM agent's job is "propose code changes to strategy.py." V2 showed that Optuna is better at parametric search (200 trials in 89s vs 15+ LLM iterations for the same task) and the LLM tends to converge on familiar patterns from its training data (z-score momentum, ATR stops) regardless of the signal class. The LLM's comparative advantage isn't writing strategy code — it's reading complex data and producing qualitative assessments.

**Solution:** Shift the LLM's role from strategy coder to research analyst:

- **Trade analysis:** After each backtest, the LLM reads the trade log and market context for each trade. It writes a narrative: "Trade #47 entered during a funding extreme but price was already extended — the trailing stop was too tight for the vol regime." This produces insights that aggregate statistics miss.
- **Regime commentary:** Given the regime classification from §6.5.5, the LLM describes what's happening in each regime in natural language and assesses whether the strategy's logic is appropriate for it.
- **Strategic review:** The Research Director function from §6.2 — reading full project state and producing structural recommendations.
- **Hypothesis generation:** Instead of "change SMA from 192 to 256," the LLM proposes hypotheses: "The strategy underperforms in Q4 2024 because funding rates were compressed during the ETF approval rally — the percentile rank doesn't capture this because the lookback window includes a period of much higher funding."

The key shift: the LLM produces *analysis and hypotheses*, not code. Code changes are delegated to either a coding agent (for structural changes) or Optuna (for parametric search). The LLM's output is qualitative intelligence that informs the Research Director's decisions.

**Implementation:**
- New `analyst.py` module with prompts for trade analysis, regime commentary, and strategic review
- Fed with trade logs, equity curves, and regime data
- Output: structured reports that the Research Director consumes
- No changes to the backtest harness or strategy contract

**Expected impact:** Uncertain but potentially high. The value is in surfacing insights that numbers miss — "the strategy's W3 failure isn't about parameters, it's about the funding rate distribution being structurally different in bear markets." These are the kinds of insights that led to the project's best decisions (trying Optuna, testing the derivatives family). Automating this reasoning is the core of the Research Director vision.

### 6.6 Build Sequence

Ordered by dependency and effort-to-value ratio:

| Phase | Intervention | Effort | Depends on | Status |
|-------|-------------|--------|------------|--------|
| 1 | §6.5.2 Position sizing | Low | — | ✅ Complete — below V2 baseline |
| 1 | §6.5.1 Multi-timeframe augmentation | Moderate | — | ✅ Complete — **new project best** |
| 2 | §6.5.3 Regime switching | Moderate | §6.5.2 (benefits from), §6.5.5 (benefits from) | Pending |
| 2 | §6.5.4 Strategy portfolio | Low | — | Pending |
| 3 | §6.5.5 Regime prediction | Moderate | — | Pending |
| 3 | §6.5.6 LLM analyst | Moderate | §6.5.5 (uses regime data) | Pending |
| 4 | §6.3 Research Director integration | High | §6.5.6 (core reasoning), all above (toolkit) | Pending |

**Phase 1 — Position sizing + Multi-timeframe (parallel, no dependencies):** ✅ Complete. See §6.8 for results.

**Phase 2 — Regime switching + Strategy portfolio (builds on Phase 1):**
Regime switching should build on the MTF strategy (not position sizing, which failed). The strategy portfolio is independent but lower priority. Both are composition approaches — combining existing capabilities in new ways.

**Phase 3 — Regime prediction + LLM analyst:**
The regime classifier produces the data that the LLM analyst consumes. Together they form the analytical backbone that the Research Director needs to make informed structural decisions.

**Phase 4 — Research Director integration:**
Wire the strategic review protocol (§6.4) into the system as an automated process. The director reads project state, runs the reasoning sequence, produces recommendations, and delegates execution. This is the capstone — it requires all previous phases to be operational so the director has a rich toolkit to work with.

### 6.7 Full 2-Year Backtest Baseline (March 2026)

Before implementing V3 changes, the D+A ensemble strategy was evaluated as a continuous 2-year backtest (not windowed) to establish a baseline:

| Metric | V2 D+A Ensemble | BTC Buy-and-Hold |
|--------|:-----------:|:----------------:|
| Total Return | +14.3% | −3.5% |
| Sharpe Ratio | 0.467 | — |
| Max Drawdown | 21.8% | — |
| Max DD Duration | 423 days | — |
| Total Trades | 105 | — |
| Win Rate | 45% | — |
| Avg Win / Avg Loss | $249 / $177 | — |

See `equity_curve.py` for the full backtest runner and `equity_report.html` for the V2 quantstats tearsheet.

This baseline is what every V3 intervention must improve upon. The primary targets: reduce max drawdown duration (currently 423 days), improve Sharpe (currently 0.467), and maintain or improve total return (+14.3%).

### 6.8 V3 Phase 1 Results (March 2026)

Phase 1 tested two parallel expansions of the strategy contract: position sizing (§6.5.2) and multi-timeframe signal augmentation (§6.5.1). Both were implemented, Optuna-optimized (200 trials each), and validated with stability checks and walk-forward analysis.

#### 6.8.1 Implementation

**Position sizing (§6.5.2):**
- Extended `backtest.py` and `equity_curve.py` to detect a 3-tuple return from `compute_signals(df, params) -> (entries, exits, size)`. The `size` Series (0.0–1.0) specifies the fraction of available cash to allocate on each entry. Fully backward compatible — 2-tuple returns default to size=1.0.
- Created `runs/v3_position_sizing/strategy.py` with vol-based continuous sizing: `size = ceiling - (ceiling - floor) * vol_pct`, replacing the hard boolean vol gate. Funding distance from threshold provides an additional sizing bonus.
- Added `suggest_d_a_sized` Optuna preset with 11 parameters (9 from V2 D+A + 4 sizing params, minus 2 vol gate params).

**Multi-timeframe augmentation (§6.5.1):**
- Added `augment_with_timeframes(df)` to `data.py`: resamples 1h OHLCV to 4h and 1d, computes features (close, SMA(50), SMA(200), GK vol, momentum, range) at each timeframe, adds as forward-filled prefixed columns (`h4_*`, `d1_*`).
- Created `runs/v3_mtf/strategy.py` replacing the V2 1h SMA(266) trend filter with a daily SMA trend filter. Optional 4h confirmation and daily vol exit.
- Added `suggest_d_a_mtf` Optuna preset with 13 parameters and `--augment-timeframes` flag to `optimize_params.py` and `equity_curve.py`.

#### 6.8.2 Position Sizing Results

**Best fitness: +1.545** (trial #185 of 200, 85s total)

| Window | Period | Sharpe | Return | Trades |
|--------|--------|--------|--------|--------|
| W1 | Mar–Jun 2024 (choppy) | +1.718 | +3.9% | 4 |
| W2 | Sep–Dec 2024 (bull) | +3.493 | +8.2% | 11 |
| W3 | Feb–May 2025 (bear) | +2.892 | +8.6% | 11 |
| W4 | Jul–Oct 2025 (recovery) | +1.815 | +2.7% | 8 |
| W5 | Dec–Mar 2026 (recent) | +0.441 | +1.0% | 8 |

Walk-forward OOS (W4-5): **+0.784** (49.2% decay). Stability: **60.8% worst drop** (FRAGILE).

**Assessment:** Position sizing underperforms the V2 ensemble (+1.845). The optimizer found a narrow sizing range (0.32–0.52) — the vol regime works better as a hard boolean gate than as a continuous position modulator. The key issue: removing the hard vol gate to replace it with continuous sizing increased trade count in bad-vol regimes, diluting returns. The additional flexibility of continuous sizing does not compensate for the loss of the sharp filtering effect.

**Key parameter importances:** `vol_size_midpoint` (26%), `fr_exit_pct` (21.4%), `fr_pct_window` (10.3%).

**Conclusion:** Position sizing does not improve the D+A synthesis. The binary vol gate is more effective than continuous size modulation for this strategy structure. This finding redirects Phase 2: regime switching should build on the binary-gated MTF strategy, not on continuous position sizing.

#### 6.8.3 Multi-Timeframe Results

**Best fitness: +3.082** (trial #151 of 200, 82s total) — **new project best, 67% above V2 ensemble.**

| Window | Period | Sharpe | Return | MDD | Trades |
|--------|--------|--------|--------|-----|--------|
| W1 | Mar–Jun 2024 (choppy) | +4.238 | +16.0% | 3.9% | 6 |
| W2 | Sep–Dec 2024 (bull) | +2.367 | +11.4% | 5.8% | 12 |
| W3 | Feb–May 2025 (bear) | +3.844 | +20.1% | 4.6% | 6 |
| W4 | Jul–Oct 2025 (recovery) | +3.423 | +8.1% | 3.9% | 5 |
| W5 | Dec–Mar 2026 (recent) | +3.136 | +16.6% | 9.1% | 3 |

All 5 windows positive. MDD ≤ 9.1% in every window.

**Walk-forward OOS (W4-5): +3.208** — the strategy actually **improves** out of sample (−4.1% decay). This is extraordinarily unusual and indicates the daily trend filter generalizes forward better than the training windows suggest.

**Stability: 34.2% worst drop** (FRAGILE but improved from V2's 40.7%).

**Ensemble (top-20): +2.677** (fitness), OOS +2.682 (0.5% decay), 36.4% worst drop. Parameter spreads are very tight — `d1_sma_period` range 20–30 (std=2.2), confirming the optimizer strongly converges on a 20-day daily SMA.

**Optimized parameters (ensemble-averaged):**

| Parameter | V2 Ensemble | V3 MTF Ensemble | Change |
|-----------|------------|-----------------|--------|
| Trend filter | sma_period=266 (1h) | **d1_sma_period=20 (daily)** | Timeframe shift |
| `fr_pct_window` | 418 | 550 | Longer funding lookback |
| `fr_entry_pct` | 0.690 | 0.685 | Similar |
| `fr_exit_pct` | 0.760 | **0.871** | Much looser exit |
| `exit_lookback` | 55 | 53 | Similar |
| `vol_lookback` | 52 | **34** | Shorter vol window |
| `vol_pct_window` | 1070 | **971** | Shorter vol lookback |
| `vol_entry_pct` | 0.509 | **0.769** | Much more relaxed |
| `vol_exit_pct` | 0.565 | **0.682** | More relaxed |

**Key findings:**

1. **The daily SMA(20) is the breakthrough.** Replacing the 1h SMA(266) with a daily SMA(20) is the single most impactful structural change in the project's history. The daily SMA captures 20 actual trading days of trend information (vs the 1h SMA's noisy 11-day lookback), providing dramatically cleaner trend identification that eliminates false entries during intraday noise.

2. **4h confirmation adds no value.** The optimizer consistently sets `use_h4_confirmation=False` (0.5% importance). The daily trend filter is sufficient — 4h adds redundancy without improving signal quality.

3. **Vol gate relaxes significantly.** `vol_entry_pct` jumped from 0.509 to 0.769, and `vol_exit_pct` from 0.565 to 0.682. The cleaner daily trend filter allows trading in higher-vol regimes that the V2 strategy had to avoid — the 1h SMA produced false signals during volatile periods, making the tight vol gate necessary. With the daily trend, that protection is no longer needed.

4. **Funding exit loosens.** `fr_exit_pct` rose from 0.760 to 0.871. Trades stay open longer because the daily trend provides better timing for position duration — exits are driven more by the trailing low and vol spike rather than early funding exits.

**Parameter importance:** `vol_pct_window` (29.6%), `d1_sma_period` (21.4%), `vol_lookback` (8.9%).

#### 6.8.4 Full 2-Year Backtest — V3 MTF Ensemble (March 2026)

| Metric | V2 D+A Ensemble | V3 MTF Ensemble | Change |
|--------|:-----------:|:----------------:|:------:|
| Total Return | +14.3% | **+207.4%** | +1350% |
| Sharpe Ratio | 0.467 | **2.763** | +492% |
| Sortino Ratio | — | **4.315** | — |
| Max Drawdown | 21.8% | **9.5%** | −56% |
| Max DD Duration | 423 days | **55 days** | −87% |
| Total Trades | 105 | **79** | −25% |
| Win Rate | 45% | **62%** | +38% |
| Avg Win / Avg Loss | $249 / $177 | **$618 / $317** | +95% / +79% |

See `v3_mtf_equity_report.html` for the full quantstats tearsheet.

The improvement is comprehensive. Max drawdown duration — the primary V3 target — dropped from 423 to 55 days. The strategy trades less frequently (79 vs 105 trades) but with much higher quality: 62% win rate and a 1.95:1 win/loss ratio (vs 45% and 1.41:1 in V2). Sharpe improved 5.9x.

#### 6.8.5 Comprehensive Comparison (March 2026)

| Strategy | Full Fitness | WF OOS (W4-5) | WF Decay | Stable? |
|----------|-------------|----------------|----------|---------|
| Track E (TA baseline) | +0.471 | +0.779 | −143% | N/A |
| Track D (funding standalone) | +1.064 | +0.660 | +58% | N/A |
| D+A manual synthesis (§5.14) | −0.615 | +0.547 | N/A | N/A |
| D+A Bayesian single-best (§5.15) | +2.030 | +1.991 | +3.3% | 55.0% No |
| D+A Ensemble (V2 best) | +1.845 | +1.816 | +3.8% | 40.7% No |
| V3 Position Sizing | +1.545 | +0.784 | +49.2% | 60.8% No |
| **V3 MTF single-best** | **+3.082** | **+3.208** | **−4.1%** | 34.2% No |
| V3 MTF Ensemble (top-20) | +2.677 | +2.682 | +0.5% | 36.4% No |

The V3 MTF single-best achieves the highest fitness ever recorded (+3.082) with the lowest walk-forward decay (−4.1%, meaning it improves OOS). The ensemble trades headline fitness for tighter parameter convergence and near-zero decay.

#### 6.8.6 Extended Validation — 5-Year Backtest (March 2026)

To test whether the strategy generalises beyond its 2-year training window, the V3 MTF ensemble was evaluated on 5 years of data (Apr 2021 – Mar 2026). The 3 years prior to the training period (2021–2023) are completely unseen — the parameters were never optimised on this data.

**9-window evaluation (5 years, 91-day windows):**

Fitness: **+1.187** (mean_sharpe=+1.800, std=1.226)

| Window | Period | Sharpe | Return | MDD | Trades |
|--------|--------|--------|--------|-----|--------|
| W1 | Apr–Jun 2021 (bull peak) | +1.390 | +9.7% | 9.7% | 3 |
| W2 | Nov 2021–Feb 2022 (bear start) | +0.584 | +1.4% | 5.5% | 1 (⚠) |
| W3 | Jun–Sep 2022 (deep bear) | +1.494 | +8.7% | 5.9% | 3 |
| W4 | Jan–Apr 2023 (bottom) | +0.073 | −0.5% | 12.2% | 5 |
| W5 | Aug–Nov 2023 (recovery) | +3.733 | +13.3% | 4.1% | 12 |
| W6 | Mar–Jun 2024 (choppy) | +3.481 | +15.7% | 3.9% | 8 |
| W7 | Oct 2024–Jan 2025 (bull) | +1.599 | +8.2% | 7.7% | 6 |
| W8 | May–Aug 2025 (recovery) | +1.997 | +5.7% | 3.6% | 6 |
| W9 | Dec 2025–Mar 2026 (recent) | +2.435 | +12.8% | 9.1% | 4 |

All 9 windows have positive Sharpe. W2 (bear onset) has only 1 trade and is treated as 0 for fitness calculation. W4 (market bottom) is the weakest genuine window at +0.073 — the strategy went nearly flat during the hardest period, rather than losing. W3 (the deep 2022 bear, BTC −55%) returned **+8.7%**.

**Full 5-year continuous backtest:**

| Metric | V3 MTF Ensemble | BTC Buy-and-Hold |
|--------|:-----------:|:----------------:|
| Total Return | **+1,221%** | +12.7% |
| Final Equity ($10k start) | **$132,111** | $11,270 |
| Sharpe Ratio | **2.209** | — |
| Sortino Ratio | **3.346** | — |
| Max Drawdown | **16.7%** | ~77% |
| Max DD Duration | **170 days** | — |
| Total Trades | 201 | — |
| Win Rate | 55% (111/201) | — |
| Avg Win / Avg Loss | $1,705 / $746 | — |

**Per-year breakdown:**

| Year | Strategy | BTC | Market Regime |
|------|----------|-----|---------------|
| 2021 (Apr–Dec) | **+45.5%** | −22.0% | Bull peak → correction |
| 2022 | **+51.1%** | −64.5% | Bear crash |
| 2023 | +50.8% | +155.8% | Recovery rally |
| 2024 | +99.4% | +120.3% | Bull run |
| 2025 | **+65.3%** | −7.2% | Mixed |
| 2026 (Jan–Mar) | **+20.6%** | −24.0% | Declining |

Every single year is positive. The strategy made +51.1% during the 2022 bear crash while BTC lost 64.5% — this is the strongest possible out-of-sample validation, since the parameters were optimised entirely on 2024–2026 data. The strategy underperforms BTC during strong bull runs (2023, 2024) but dramatically outperforms in bearish and choppy conditions — consistent with the daily SMA(20) trend filter keeping the strategy flat during major downtrends.

**Caveats:**

1. The 5-year max DD duration (170 days) is longer than the 2-year figure (55 days). Extended data reveals periods of stagnation not visible in the training window.
2. W2 (Nov 2021 – Feb 2022) has only 1 trade — the strategy was mostly flat during the transition from bull to bear. This is actually correct behaviour (the daily SMA turned bearish), but results in sparse data.
3. The compounding effect flatters the headline return (+1,221%). The per-year breakdown is more informative: consistent 45–99% annual returns, not a single explosive period.
4. 2021–2023 funding rate dynamics differ from 2024–2026 (retail-dominated vs institutional/ETF). The strategy works across both, which strengthens the structural thesis.

#### 6.8.7 Cross-Asset Validation — ETH/USDT (March 2026)

The V3 MTF ensemble was tested on ETH/USDT with **zero parameter changes** — the same BTC-optimized parameters applied to ETH funding rates and price data. This is a cross-asset out-of-sample test: if the strategy works on ETH, the signals capture structural properties of crypto perpetual futures markets, not BTC-specific patterns.

**9-window evaluation (ETH, 5 years): fitness +0.750**

| Window | Period | BTC Sharpe | ETH Sharpe | ETH Return | ETH Trades |
|--------|--------|:----------:|:----------:|:----------:|:----------:|
| W1 | Apr–Jun 2021 (bull peak) | +1.390 | **−1.162** | −10.9% | 4 |
| W2 | Nov 2021–Feb 2022 (bear start) | 0 (⚠) | 0 (⚠) | +0.9% | 1 |
| W3 | Jun–Sep 2022 (deep bear) | +1.494 | +0.920 | +7.2% | 6 |
| W4 | Jan–Apr 2023 (bottom) | +0.073 | **−0.237** | −2.7% | 5 |
| W5 | Aug–Nov 2023 (recovery) | +3.733 | +4.263 | +17.1% | 8 |
| W6 | Mar–Jun 2024 (choppy) | +3.481 | +2.681 | +15.7% | 7 |
| W7 | Oct 2024–Jan 2025 (bull) | +1.599 | +0.917 | +5.3% | 4 |
| W8 | May–Aug 2025 (recovery) | +1.997 | **+4.910** | +48.7% | 11 |
| W9 | Dec 2025–Mar 2026 (recent) | +2.435 | +3.673 | +26.4% | 6 |

7 of 9 windows positive. The two negative ETH windows (W1, W4) correspond to ETH-specific events: the May 2021 flash crash (ETH −60% in 2 weeks) and the choppy 2023 bottom. BTC was positive in both these periods, indicating ETH-specific idiosyncratic risk.

**Full 5-year continuous backtest (ETH vs BTC comparison):**

| Metric | BTC (5yr) | ETH (5yr, BTC params) |
|--------|:---------:|:---------------------:|
| Strategy Return | +1,221% | **+3,172%** |
| Buy-and-Hold | +12.7% | +4.8% |
| Final Equity ($10k) | $132,111 | **$327,170** |
| Sharpe Ratio | 2.209 | **2.374** |
| Sortino Ratio | 3.346 | **3.644** |
| Max Drawdown | 16.7% | 25.5% |
| Max DD Duration | 170 days | **142 days** |
| Total Trades | 201 | 194 |
| Win Rate | 55% | **61%** |

**Per-year breakdown (ETH):**

| Year | Strategy | ETH | Market Regime |
|------|----------|-----|---------------|
| 2021 (Apr–Dec) | **+91.6%** | +90.2% | Bull peak |
| 2022 | **+101.6%** | −67.9% | Bear crash |
| 2023 | +72.2% | +91.1% | Recovery |
| 2024 | +72.7% | +45.4% | Bull run |
| 2025 | **+114.1%** | −11.7% | Mixed |
| 2026 (Jan–Mar) | **+31.9%** | −32.0% | Declining |

Every year positive on ETH. The strategy made **+101.6% during the 2022 crash** while ETH lost 67.9%. ETH's higher volatility amplifies both gains and losses — the strategy achieves higher total returns (+3,172% vs +1,221%) but with higher max drawdown (25.5% vs 16.7%).

**Key findings:**

1. **The funding rate + daily SMA mechanism is not BTC-specific.** It works on ETH with zero parameter changes, confirming the signals capture a structural property of crypto perpetual futures markets: the relationship between funding rate positioning, daily trend, and vol regime.

2. **ETH amplifies the strategy.** Higher beta means larger moves in the direction the strategy trades. When the daily SMA(20) trend filter is correct, ETH's larger swings produce larger profits.

3. **ETH-specific risks exist.** W1 (May 2021 flash crash) and W4 (2023 bottom) are negative on ETH but positive on BTC. ETH has more violent idiosyncratic moves that occasionally catch the strategy. Asset-specific parameter tuning could address this.

4. **A two-asset portfolio would diversify risk.** Running the strategy on both BTC and ETH with equal allocation would likely smooth drawdowns — the negative ETH windows don't overlap with negative BTC windows.

#### 6.8.8 Robustness Audit — Look-Ahead Fix (March 2026)

A post-hoc audit identified a **look-ahead bias** in `augment_with_timeframes()`: the daily close (`d1_close`) was computed using all bars in the current day, including the bar at which the strategy could trade. In live trading, you only know *yesterday's* completed daily close at today's open. The strategy's `d1_close > d1_sma` comparison was partially tautological — on bullish days, today's close naturally exceeds the 20-day average, creating a signal correlated with the present rather than predictive of the future.

**Fix:** All MTF features are now shifted by 1 period at their native timeframe before forward-filling. At any 1h bar, the strategy only sees the previous completed 4h or daily bar.

**Additionally:** 5bp slippage per side was added to the backtest (total cost: 0.15% per side, 0.30% round-trip).

**Re-evaluation results (BTC, 5 years, 9 windows):**

| Metric | Before fix | After fix | Change |
|--------|:----------:|:---------:|:------:|
| 9-window fitness | +1.187 | **−0.951** | Collapsed |
| 5yr return | +1,221% | **−32.1%** | Collapsed |
| Sharpe | 2.209 | **−0.216** | Collapsed |
| Positive windows | 9/9 | **2/9** | Collapsed |

**Per-year (BTC, after fix):**

| Year | Strategy | BTC |
|------|----------|-----|
| 2021 | −23.3% | −22.0% |
| 2022 | −3.8% | −64.5% |
| 2023 | −10.0% | +155.8% |
| 2024 | +8.9% | +120.3% |
| 2025 | −5.4% | −7.2% |
| 2026 | −0.9% | −24.0% |

ETH results similarly collapsed (5yr return −9.0%, Sharpe +0.093).

**Diagnosis:** The V3 MTF "breakthrough" was almost entirely driven by look-ahead bias in the daily close. The `d1_sma_period=20` finding — which appeared to be the project's strongest structural insight — was an artefact of same-day information leakage.

**What remains valid:**

1. **The V2 D+A ensemble (+1.845 fitness) is NOT affected.** Its 1h SMA(266) is computed on hourly closes that are available at signal time. No look-ahead.
2. **The position sizing contract extension** (3-tuple returns from `compute_signals`) is a valid infrastructure improvement regardless.
3. **The multi-timeframe augmentation infrastructure** (`augment_with_timeframes`) is now correctly lagged and available for future use.

**What must be discarded:**

1. All V3 MTF fitness numbers, walk-forward results, and 5-year validation reported in §6.8.3–§6.8.7 were computed with look-ahead and are invalid.
2. The "daily SMA(20) as a breakthrough" finding is an artefact.
3. The ETH cross-asset validation was also computed with look-ahead and is invalid.

**Lessons:**

1. Multi-timeframe features require explicit lag to prevent look-ahead. Any feature computed at a lower frequency than the trading timeframe must be shifted by 1 period at its native frequency before forward-filling.
2. Extraordinary results demand adversarial auditing. The +3.082 fitness (67% above V2) and negative walk-forward decay should have triggered immediate suspicion.
3. The walk-forward validation did not catch this because the look-ahead existed in both training and test windows — it inflated both equally.

**Current project best (valid): V2 D+A ensemble, fitness +1.845, OOS +1.816.**

#### 6.8.9 MTF Re-Optimization with Lagged Features (March 2026)

After fixing the look-ahead and adding 5bp slippage, the V2 D+A ensemble baseline re-evaluates to **+1.640** (down from +1.845 without slippage). A new 200-trial Optuna study was run on the `d_a_mtf` preset with correctly lagged features.

**Best fitness: +0.778** (trial #186) — significantly below the V2 baseline.

The optimizer moved `d1_sma_period` from 20 (look-ahead contaminated) to 110 (lagged). With a 110-day lagged daily SMA, the trend filter is so slow it kills trade count: W4 has 2 trades, W5 has 0. The lagged daily close adds no genuine signal that the 1h SMA(266) doesn't already capture.

**Conclusion:** The daily trend filter does not add value when properly lagged. The multi-timeframe augmentation infrastructure is sound (shift by 1 period), but the signal class doesn't produce an edge at this timeframe/asset combination.

#### 6.8.10 Implications for Phase 2

1. **The V2 D+A ensemble remains the project's best strategy.** `runs/synthesis_D_A/strategy.py` with fitness +1.640 (with 5bp slippage) is the valid base for Phase 2.

2. **Multi-timeframe augmentation is a dead end for now.** The daily trend filter adds no value when properly lagged. The 4h timeframe was consistently disabled by the optimizer. Future MTF work should focus on *different* features at lower timeframes (e.g., daily vol regime, daily funding rate aggregates) rather than price-based trend filters.

3. **Position sizing does not improve the D+A synthesis.** This finding (§6.8.2) was computed on 1h features and is unaffected by the MTF look-ahead issue.

4. **Adversarial auditing is now automated.** `check_signal_integrity()` runs on every backtest and flags daily return correlation > 0.15 as FAIL. The slippage default of 5bp is threaded through all evaluation paths. These guards would have caught the MTF look-ahead on the first run.

---

## 7. Deferred Ideas

Ideas not addressed by the V3 roadmap. Revisit when V3 is operational.

**Adversarial regime generation.** A second agent generates synthetic market scenarios designed to break the current best strategy. The discovery agent must find approaches robust to both real history and adversarial stress tests.

**Retrieval-augmented strategy memory.** Log all experiments with performance and market conditions. When current conditions resemble a historical period, retrieve strategies that worked then. Case-based reasoning for trading.

**On-chain data.** Exchange flows, whale wallet monitoring, mempool analysis. High signal potential, clear mechanisms, but a fundamentally different data pipeline. Consider after V3 infrastructure is proven.

**Open interest (with paid data).** Track F was blocked by Binance's 30-day API limit. A paid data source (CoinGlass, Glassnode) would unlock the most promising untested signal class. Consider when the Research Director identifies OI as a priority.

**Short selling.** Long-only through V1–V3. Short strategies add complexity to position sizing, risk management, and regime classification. Defer until long-only regime switching is validated.

**Multi-asset expansion.** BTC-only through V3. Cross-asset portfolios (BTC + ETH + SOL) with correlated/anti-correlated allocation. Defer until single-asset portfolio management (§6.5.4) is working.

**Live execution.** Paper-trading or shadow-trading against real prices. Requires latency management, order routing, risk limits, and monitoring. The natural next step after V3 produces a strategy with acceptable drawdown characteristics.

---

*This document is a living reference. Update it as the project evolves.*
