# The Trading Agent
**Design Brief — v0.1**
*March 2026*

---

## 1. What V1–V4 Taught Us

Four versions of a statistical trading system, hundreds of Optuna trials, and months of research produced one clear conclusion: **the edge in systematic crypto trading is thin and friction-bound.** The key findings, condensed:

1. **Transaction cost is the binding constraint.** The best system (V2 D+A ensemble, fitness +1.640) trades ~100 times per year with ~30bp of edge per trade. After 5bp slippage per side, almost nothing remains. Every V3 structural improvement (MTF, regime switching, position sizing, portfolio composition) failed because none increased per-trade profitability.

2. **Structural signals are real but competed away.** Funding rates are a genuine signal — they measure positioning pressure from leveraged traders. But precisely because they're systematic and computable, quant desks have already arbitraged them thin. The mechanism is real; the edge is paper-thin.

3. **Behavioral signals identify high-quality entries but are too rare.** The V4 Fear & Greed experiment found 370bp of edge per trade — 12x the V2 number. But only ~3 trades per year. The signal works; there just aren't enough fear washouts per year to build a strategy around.

4. **More complexity makes things worse.** V3 added four layers of sophistication on top of V2. Every single one degraded performance. The best strategy does one thing simply: buy when funding says positions are light and vol says the market is calm.

5. **Honest testing kills most ideas.** The integrity guard (look-ahead detection + 5bp slippage) caught two major episodes of self-deception: the MTF "breakthrough" (inflated from −0.22 to +2.21 by look-ahead) and the FNG result (inflated from +0.57 to +1.43). Any system that produces exciting results should be assumed guilty until the integrity check passes.

6. **The `compute_signals(df, params)` paradigm has a ceiling.** Reducing trading to a mathematical function that maps price data to buy/sell signals can't express the kind of contextual judgment that discretionary traders use. Market structure — FVGs, order blocks, liquidity zones, accumulation/distribution — is spatial and contextual, not a threshold on an indicator.

**Bottom line:** Statistical signals work but are too thin to survive friction. Behavioral signals work but are too rare. The approaches can be combined (they weren't — Optuna disabled FNG when overlaid on D+A). But the deeper issue is representational: the signal-function paradigm can't express the patterns that actually carry thick, tradable edge.

---

## 2. The Thesis

Good discretionary crypto traders do something different from what V1–V4 tried. They:

- **Read market structure** — support/resistance, fair value gaps, order blocks, liquidity zones, areas of accumulation and distribution
- **Know the regime** — whether the market is trending, ranging, distributing, accumulating
- **Track the meta** — what's currently driving price (ETF flows, macro fear, altcoin rotation, memecoin cycle)
- **Watch the crowd** — BTC dominance, Fear & Greed, funding rates, liquidation levels, social sentiment
- **Exercise patience** — wait for high-conviction setups instead of forcing trades
- **Learn from experience** — "last time I saw this pattern in this context, here's what happened"

None of this maps to `close.rolling(N).rank(pct=True) < threshold`. It's pattern recognition, contextual reasoning, and accumulated judgment. These are exactly what LLMs are good at — but the previous project used LLMs as code writers, not as traders.

**The new thesis:** Build a trading agent that reasons about markets the way a good trader does — reading structure, understanding context, exercising judgment — but with the consistency, patience, and emotional discipline that humans lack. The agent develops skill over time through structured memory and periodic self-review.

This is not a parameter optimization problem. It's an apprenticeship.

---

## 3. Architecture

### 3.1 The top-down principle

Good traders always zoom out before zooming in. Before looking at a 1h chart for an entry, they check the weekly and daily to understand where price sits in the bigger picture — major pivot zones, long-term trend direction, whether the market is in distribution or accumulation at the macro scale. They also check what BTC is doing (it leads the market), BTC dominance (risk-on vs. risk-off), and the general sentiment landscape.

This top-down flow is baked into the architecture. The system maintains a **macro context** — a cached, periodically updated view of the broader landscape — that sits upstream of any trade-level analysis. Think of it as the answer to "what's the big picture right now?" that the agent reads before ever looking at a specific setup.

### 3.2 Macro context (updated daily or on significant moves)

The macro context is the stable foundation that all analysis sits on top of. It's cheap to maintain because it changes slowly.

**What it contains:**

| Layer | Charts | Key questions |
|-------|--------|---------------|
| BTC weekly | Weekly candle chart (1–2 years visible) | Where are the major pivot zones? What's the long-term trend? Are we in a macro range, breakout, or breakdown? Where are the yearly/monthly levels that will act as magnets? |
| BTC daily | Daily candle chart (3–6 months visible) | What's the medium-term structure? Is this a trending or ranging environment? Where are the swing highs/lows that define the current structure? |
| BTC dominance | Daily BTC.D chart | Rising = risk-off, money flowing to BTC from alts. Falling = risk-on, alt season. Flat = neutral. Is BTC.D at a key level itself? |
| Sentiment | FNG, funding rates, narrative | What's the crowd state? Where are we in the fear/greed cycle? What's the current market narrative? |

**How it's used:** The macro context is generated by the LLM reading the weekly/daily charts + BTC.D + sentiment, and writing a concise briefing (~200 words). This briefing is cached and prepended to every analysis call. Updated once per day in normal conditions, or immediately after a large move (>5% daily candle, significant news event).

**Cost:** One LLM call per day with ~4 chart images (BTC weekly, BTC daily, BTC.D daily, trading asset daily if not BTC) + text context = ~12k tokens/day. This is the most expensive single call, but it only runs once and the output is reused for every analysis that day.

### 3.3 The five layers

```
┌─ MACRO CONTEXT (updated daily or on significant moves) ─────┐
│ Weekly + daily charts for BTC and BTC.D.                     │
│ Sentiment landscape (FNG, funding, narrative).               │
│ Output: cached briefing — "the big picture right now."       │
│ Cost: ~12k tokens, once per day.                             │
└──────────────┬──────────────────────────────────────────────┘
               │ always available as context
               ▼
┌─ SCREEN (cheap, runs every 1–4h) ──────────────────────────┐
│ Numeric checks against key levels, sentiment, positioning.  │
│ Is anything interesting happening? Usually: no → sleep.     │
│ Cost: near zero (code only). No LLM.                        │
└──────────────┬──────────────────────────────────────────────┘
               │ triggers ~2–6x per day
               ▼
┌─ ANALYZE (LLM reasoning, runs on trigger) ──────────────────┐
│ Inputs: macro context briefing, annotated setup charts (1h/  │
│         4h), playbook, similar past journal entries           │
│ Reasoning: does this setup fit the macro picture? Setup      │
│            identification, conviction grading, R:R assessment │
│ Output: decision (trade / wait / exit) + written rationale   │
│ Cost: ~8k–10k tokens per analysis                            │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─ EXECUTE (deterministic, no LLM) ───────────────────────────┐
│ Place orders, manage positions, enforce risk limits.         │
│ Hard guardrails: max position size, max drawdown, stop-loss. │
│ No LLM in the execution path — rules only.                   │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─ RECORD (after every decision) ─────────────────────────────┐
│ Write structured journal entry: what was seen, what was      │
│ decided, why, what the market context was.                   │
│ After trade closes: record outcome + lessons.                │
└──────────────┬──────────────────────────────────────────────┘
               │ periodic (weekly / after N trades)
               ▼
┌─ REVIEW (LLM reflection, runs periodically) ────────────────┐
│ Read recent journal entries. What's working? What isn't?     │
│ Update the playbook. Recalibrate confidence.                 │
│ Monthly: strategic review — is the meta shifting?            │
│ Cost: ~5k–20k tokens per review                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Why five layers

**Macro context** establishes the big picture. A setup on the 1h chart means nothing if the weekly shows price grinding into major resistance. A beautiful FVG fill doesn't matter if BTC.D is spiking (meaning the asset you're trading is probably about to bleed against BTC regardless of its own chart). The macro context prevents the agent from tunnel-visioning on a setup while ignoring the environment it sits in. It also keeps the agent from overtrading in unfavorable macro conditions — if the daily briefing says "distribution at resistance, expect chop," the agent's conviction threshold for entries should be higher.

**Screen** exists for cost control. Markets are boring most of the time. A simple numeric filter (price near key level? FNG extreme? volume spike? funding anomaly?) costs nothing and eliminates 80% of analysis calls.

**Analyze** is where judgment lives. Critically, it starts by reading the macro context before looking at the specific setup. The LLM should be reasoning: "The weekly shows we're in a range between $74k and $88k. The daily is trending down within that range. BTC.D is rising. In this context, this 1h FVG fill at support is a counter-trend bounce attempt — conviction should be moderate, not high, and targets should be conservative." The macro framing shapes the trade thesis.

**Execute** is deliberately non-intelligent. Hard rules, no LLM. This is where risk management lives, and risk management should never be negotiable or "creative." Max position size, max drawdown trigger, stop-loss enforcement — these are guardrails, not suggestions.

**Record + Review** is where learning happens. The system can't update its own weights, but it can build a structured memory of what it's seen and what worked. The periodic review synthesizes that memory into the playbook, which feeds back into future analysis.

---

## 4. Core Concepts

### 4.1 The Playbook

A living document that encodes everything the agent has learned. It starts seeded with knowledge from V1–V4 and grows with each review cycle. Structured sections:

- **Market structure patterns** — which setups the agent looks for, how to identify them, and historical hit rates from its own journal. Examples: "FVG fill at major support after washout," "breakout from compression with volume confirmation," "liquidity grab below range low followed by reclaim."

- **Context rules** — how broader conditions modify setup quality. "In declining BTC.D environments, BTC setups are less reliable — alt rotation is underway." "When FNG < 20 and funding is negative, washout recovery setups have our highest historical conviction."

- **Risk rules** — position sizing, stop placement, when to sit out. "Never enter a new position when existing drawdown exceeds 5%." "Size inversely to ATR — wider stops, smaller positions."

- **Mistakes log** — recurring errors and their corrections. "We've been stopped out by wicks below support 6 times in the last 20 trades — consider placing stops below the wick zone, not at the level itself."

- **Meta awareness** — what's currently driving the market. Updated weekly. "Current regime: macro fear, declining prices, extreme negative sentiment. BTC.D rising (flight to BTC from alts). ETF flows negative. Liquidation cascades creating volatile intraday moves."

The playbook is NOT a fixed strategy. It evolves. After enough experience, certain patterns get promoted to "high confidence" and others get demoted or removed. This is the mechanism for learning.

### 4.2 The Journal

A structured, append-only log of every significant decision. Each entry includes:

```
{
  "timestamp": "...",
  "type": "analysis" | "entry" | "exit" | "skip" | "review",

  // What was observed
  "market_snapshot": {
    "price": ..., "btc_dominance": ..., "fng": ...,
    "funding_rate": ..., "regime": "...",
    "key_levels": [...],
    "structure_notes": "Price testing support at $X after FVG fill..."
  },

  // What was decided and why
  "decision": "enter_long" | "exit" | "wait" | "skip",
  "conviction": "high" | "moderate" | "low",
  "rationale": "Seeing accumulation at range low with declining volume...",
  "setup_type": "fvg_fill_at_support",
  "playbook_rules_applied": ["washout_recovery", "vol_compression"],

  // What happened (filled in after trade closes)
  "outcome": {
    "entry_price": ..., "exit_price": ..., "return_pct": ...,
    "hold_duration": "...",
    "what_actually_happened": "Price bounced from support but...",
    "lesson": "..."
  }
}
```

The journal serves three purposes:
1. **Retrieval** — When analyzing a new situation, retrieve similar past entries to inform the decision ("What happened last time we saw this?")
2. **Review input** — The periodic review reads the journal to identify patterns
3. **Audit trail** — Every decision is traceable and reviewable

### 4.3 Market Structure Vocabulary

The agent needs to understand these concepts as first-class objects, not as indicator thresholds:

| Concept | What it is | Why it matters |
|---------|-----------|----------------|
| **Support / Resistance** | Price levels where buying/selling pressure has historically concentrated | S/R levels are where the crowd has placed orders. Price reacts to them because participants act on them. |
| **Fair Value Gap (FVG)** | A gap in price action where one side (buying or selling) overwhelmed the other, leaving an imbalance | FVGs tend to get "filled" — price returns to the gap zone because unfilled orders and imbalances attract mean-reversion flow. |
| **Order Block** | The last opposing candle before a strong impulsive move | Institutional entries often occur at order blocks. When price returns to one, similar buying/selling pressure tends to reappear. |
| **Liquidity Zone** | Areas where stop-losses and liquidations are clustered (below swing lows, above swing highs, round numbers) | Smart money hunts liquidity. Price often spikes through a liquidity zone, triggers stops, then reverses. |
| **Accumulation / Distribution** | Extended ranging periods where large players are building or unwinding positions | Accumulation precedes markup (price rise). Distribution precedes markdown (price fall). Identifying which is happening tells you the likely direction. |
| **BTC Dominance** | BTC's share of total crypto market cap | Rising BTC.D = risk-off (money rotating from alts to BTC). Falling BTC.D = risk-on (alt season). Important context for any trade. |
| **Market regime** | The broad behavioral state of the market: trending, ranging, distributing, capitulating, recovering | Different setups work in different regimes. A breakout strategy in a ranging market will get chopped up. |

These are not indicators to compute — they're structural features to identify. The agent reads them from price data the way a trader reads them from a chart.

---

## 5. The Learning Loop

### 5.1 How the agent improves

The agent can't update its weights. It learns by building structured memory and refining its playbook:

```
Experience (journal entries)
    ↓
Periodic review (LLM reads journal, identifies patterns)
    ↓
Playbook updates (new rules, revised confidence, removed patterns)
    ↓
Better future decisions (playbook fed into every analysis)
    ↓
New experience...
```

This mirrors how humans learn to trade: trade → journal → review → adjust approach → trade again.

### 5.2 Review cadence

- **After every trade closes:** Quick debrief. What happened? Did the rationale hold? What was the lesson? (~500 tokens)
- **Weekly:** Read the last week's journal entries. Which setups worked? Which contexts were favorable? Any recurring mistakes? Update the playbook. (~5k tokens)
- **Monthly:** Strategic review. Has the meta shifted? Are there setup types that used to work but have stopped? Are there new patterns appearing? Broader playbook revision. (~10k tokens)

### 5.3 Confidence calibration

Track prediction accuracy by setup type and conviction level:

```
fvg_fill_at_support:
  high conviction: 7/10 profitable (70%)
  moderate conviction: 4/8 profitable (50%)
  low conviction: 1/5 profitable (20%)

breakout_from_compression:
  high conviction: 3/6 profitable (50%)
  ...
```

If "high conviction" setups only hit 50%, the agent learns to recalibrate. If "moderate conviction" setups hit 70%, it learns to trust them more. This data goes into the playbook and directly affects future decision-making.

---

## 6. Oversight and Experiments

### 6.1 The problem: silent degradation

A trading agent can fail in ways that don't look like failure. It can stop trying new things and settle into a rut. It can slowly drift from its principles without any single decision being obviously wrong. It can repeat the same mistake 10 times without noticing the pattern. It can develop superstitious rules that happened to correlate with a few wins. None of these show up as a crash or an error — they show up as gradually worsening results that the agent rationalizes away.

The system needs a built-in oversight function — a "PI" — that operates across every level and asks the uncomfortable questions.

### 6.2 The oversight function

The oversight layer runs as part of the weekly and monthly reviews, but with a distinct mandate: it's not asking "what happened?" (that's the regular review). It's asking "is the system behaving well?"

**At the macro context level:**
- Is the daily briefing actually capturing what matters? Did any trade fail because the macro context missed something obvious (regime shift, BTC.D reversal, major event)?
- Is the briefing getting stale or formulaic? ("Ranging market, be cautious" every day for a month = not useful.)

**At the screening level:**
- Is the screen triggering at the right rate? Too often = wasting analysis tokens on noise. Too rarely = missing setups.
- Were there setups the agent should have caught but didn't? (Identify by looking at significant price moves and checking whether the screen triggered before them.)

**At the analysis level:**
- Is the agent stuck on one setup type? (If 80% of analyses are the same pattern, it's not exploring enough.)
- Is it repeating mistakes? (Same loss pattern appearing 3+ times without a playbook update = learning failure.)
- Is conviction calibrated? (Are "high conviction" calls actually winning more than "moderate" ones?)
- Is reasoning sound or is the agent confabulating? (Constructing plausible-sounding rationales for decisions that are really just pattern-matching to recent wins.)

**At the playbook level:**
- Is the playbook getting better or just bigger? Complexity should plateau or decrease over time as the agent converges on what works. A playbook that only grows is accumulating noise.
- Are there rules that have never been tested? (Dead weight — either test them or remove them.)
- Is the agent actually following its own playbook? (Compare stated rules against recent decisions.)

**At the execution level:**
- Are stops being placed at structurally meaningful levels, or at arbitrary distances?
- Are there patterns in how trades are exited — too early, too late, right timing?

### 6.3 Experiments

When the agent (or the oversight layer) identifies something worth testing, it shouldn't just change its behavior — it should run a formal experiment. This is how the system tries new things without contaminating its core approach.

An experiment is a structured, time-bounded test of a specific hypothesis:

```yaml
experiment:
  id: "exp-007"
  title: "Wider stops on FVG fills"
  status: "in_progress"  # proposed | in_progress | review | adopted | rejected

  hypothesis: "We're getting stopped out by wicks too often on FVG fill setups"
  evidence: "8 of 12 recent FVG stop-outs were within 0.5 ATR of our stop level"
  proposed_by: "weekly_review_2026-04-14"

  change: "Use 2.0 ATR stop instead of 1.5 ATR on FVG fill setups"
  scope: "FVG fill setups only — all other setups unchanged"
  duration: "Next 10 applicable trades"
  progress: "4/10 trades completed"

  success_criteria:
    - "Premature stop-outs drop from 67% to below 40%"
    - "Average loss per stopped trade does not increase by more than 30%"

  results: null  # filled in at review
  decision: null  # "adopt" | "reject" | "extend" | "modify"
  lessons: null
```

**Experiment lifecycle:**

1. **Proposed** — The weekly review or oversight check identifies a pattern worth testing. It writes up the hypothesis, supporting evidence, and success criteria. The experiment goes into a queue.

2. **In progress** — The agent applies the experimental change only within the defined scope. It tags affected journal entries with the experiment ID so outcomes are trackable. Everything outside the experiment's scope stays unchanged.

3. **Review** — After the duration is reached (N trades or N weeks), the agent evaluates results against the success criteria. Did it work? Why or why not?

4. **Decision** — Adopt (merge into playbook), reject (revert, log the lesson), extend (need more data), or modify (the idea has merit but needs adjustment — spin off a new experiment).

**Why this matters:**

- **Prevents stagnation.** The system must always have at least one active experiment. If the experiment queue is empty, the oversight layer's job is to propose one based on recent journal patterns.
- **Prevents reckless change.** The agent can't just overhaul its approach after one bad week. Changes are scoped, measured, and reviewed.
- **Creates a research record.** Over time, the experiment log shows what the system has tried, what worked, and what didn't — preventing it from re-running failed experiments.
- **Separates exploration from exploitation.** The playbook is "what we know works" (exploitation). Experiments are "what we think might work better" (exploration). Keeping them separate means a failed experiment doesn't corrupt the core approach.

### 6.4 Drift detection

The oversight layer specifically watches for these drift patterns:

| Drift type | Signal | Response |
|-----------|--------|----------|
| **Rut** | >80% of analyses use the same setup type for 2+ weeks | Force an experiment exploring a different setup type |
| **Complexity creep** | Playbook has grown by >30% in a month without corresponding hit-rate improvement | Prune: remove lowest-confidence rules, simplify |
| **Overconfidence** | Average stated conviction is "high" but win rate is <50% | Recalibrate conviction thresholds; restrict "high" to setups with >65% historical rate |
| **Undertrading** | Agent has skipped >20 consecutive screen triggers without a single analysis | Review skip rationale — is the macro context too bearish, or is the agent being gun-shy? |
| **Overtrading** | >5 trades per week sustained for 3+ weeks | Review whether trade quality is being maintained; check if recent win rate supports the frequency |
| **Playbook drift** | Current trading behavior contradicts stated playbook rules | Either update the playbook to match behavior (if behavior is working) or enforce the playbook (if not) |
| **Stale experiments** | An experiment has been "in_progress" for 4+ weeks without reaching its target | Force a review — either the scope is wrong, the setup is too rare, or the agent is avoiding it |

---

## 7. Token Efficiency

### 7.1 The cost problem

Running an LLM 24 times a day at ~5k tokens per call = ~120k tokens/day = significant cost at frontier model prices. The system must be designed for token efficiency from the start.

### 7.2 Solutions

**Tiered screening.** The screening layer uses zero LLM tokens — it's pure numeric checks. Most hours, nothing triggers, and the cost is zero.

**Chart rendering with TradingView Lightweight Charts.** For full analysis, render proper candlestick charts from raw OHLCV data using TradingView's [lightweight-charts](https://github.com/nicolefelice/lightweight-charts) library, captured as PNGs via headless browser (Puppeteer/Playwright).

Why TradingView specifically: TradingView charts are the dominant visual language of crypto trading. The LLM's training data contains millions of TradingView chart screenshots paired with analysis — "here's the FVG," "this is the order block," "accumulation happening here." Using the same visual format means the model's chart-reading ability is in-distribution. Render with matplotlib and it's looking at something it's seen far less.

The rendering pipeline:
1. Fetch OHLCV data (existing `data.py`)
2. Inject into an HTML template with lightweight-charts
3. Headless browser renders → screenshot to PNG
4. PNG sent to multimodal LLM

Render 2–3 timeframes per analysis (1h, 4h, daily). Each image is ~2000 tokens for a multimodal model, so 3 charts = ~6000 tokens of visual input.

**Pre-annotated charts.** The real efficiency gain: annotate charts *before* sending them to the LLM. Compute structural features in code — support/resistance levels, FVG zones, order block areas, key liquidity clusters — and draw them on the chart as overlays (shaded zones, horizontal lines, labels). Then ask the LLM to *reason about* the annotated chart rather than both *identify structure* AND reason about it. This splits the work: cheap code does the mechanical pattern detection, expensive LLM does the contextual judgment.

**Text summaries for routine context.** Not everything needs an image. Broader market state, sentiment, funding, BTC dominance — these are better as a compact text block:

```
BTC $78,420 | FNG: 11 (extreme fear) | Funding: -0.008% (shorts paying)
BTC.D: 62.3%, rising 2 weeks | Vol: 40% below 20d avg, declining 3 days
Regime: capitulation/washout | Key event: none
```

This is ~50 tokens and covers the context that doesn't need spatial representation. Reserve images for price structure, where spatial relationships (price relative to levels, FVGs, liquidity zones) are genuinely hard to convey in text.

**Hybrid approach: text for screening context, images for setup analysis.** When the screening layer triggers and the agent does a full analysis, it gets:
- Text summary of market context (~50 tokens)
- 2–3 annotated chart images at different timeframes (~6000 tokens)
- Relevant playbook section (~500 tokens)
- 3–5 similar past journal entries (~1000 tokens)
- Total per analysis: ~8000–10000 tokens

When no screen triggers: zero tokens.

**Cached context.** The broader regime assessment doesn't change hourly. Cache it and only update daily or on significant events. The playbook is relatively static — read it once per session, not per analysis.

**Selective retrieval.** Don't dump the entire journal into context. Retrieve 3–5 most similar past situations using embedding similarity or structured queries. This keeps the analysis prompt focused.

### 7.3 Target budget

| Component | Frequency | Tokens/call | Daily total |
|-----------|-----------|-------------|-------------|
| Macro context | 1x/day | ~12k | ~12k |
| Screen | 6x/day | 0 (code only) | 0 |
| Analyze (with charts) | 2–3x/day | 8k–10k | ~25k |
| Record | 2–3x/day | 500 | ~1.5k |
| Review (weekly) | 0.14x/day | 15k | ~2k amortized |
| **Total** | | | **~40k tokens/day** |

At frontier model pricing, roughly $3–8/day depending on the model. The macro context is the single biggest cost (~12k/day), but it's the most important — it prevents the agent from making well-reasoned trades in the wrong direction. The chart images in both macro and analysis calls are the main token driver; the text components are comparatively cheap.

---

## 8. Backtesting and Validation

### 8.1 Why V1–V4 backtesting doesn't work here

The old backtester runs a signal function over the entire DataFrame at once: every bar sees the same parameters, the same logic, the same everything. There's no state, no memory, no learning. That's fine for `close.rolling(N).rank() < threshold`, but it's fundamentally incompatible with an agent that builds a journal, evolves its playbook, and makes contextual decisions that depend on its own history.

The new system requires an **event-driven simulation** — a time cursor that advances through historical data, letting the agent see only the past and forcing it to accumulate memory as it goes. At any point in the simulation, the agent's state must be identical to what it would be if it had been running live up to that moment.

### 8.2 The simulation loop

```
Initialize:
  time_cursor = start_date
  journal = []
  playbook = initial_playbook (seeded from V1–V4)
  portfolio = { cash: $10,000, positions: [] }

While time_cursor < end_date:

  1. MACRO CONTEXT (daily)
     If new day:
       Render weekly + daily charts UP TO time_cursor (no future candles)
       Generate macro briefing from charts + sentiment at time_cursor
       Cache briefing

  2. SCREEN (every 1–4h)
     Advance cursor to next screen interval
     Run numeric checks on data up to time_cursor
     If nothing interesting → advance cursor, continue

  3. ANALYZE (on trigger)
     Render 1h/4h charts UP TO time_cursor
     Retrieve similar past journal entries (from entries BEFORE time_cursor)
     Feed: macro briefing + charts + playbook + retrieved entries
     LLM produces decision + rationale

  4. EXECUTE
     If entry: simulate fill at next available price (with slippage)
     If exit: simulate close at next available price
     Update portfolio

  5. RECORD
     Write journal entry timestamped at time_cursor
     If trade closed: compute outcome from actual subsequent prices

  6. REVIEW (at simulated weekly/monthly intervals)
     If simulated week boundary:
       Run review on journal entries accumulated so far
       Update playbook (the playbook the agent uses CHANGES mid-backtest)
       Run oversight checks
       Propose/review experiments

  Advance time_cursor
```

**The critical constraint:** at step 3, the chart images, text summaries, market structure annotations, journal retrieval, and playbook content must all be computed from data strictly before `time_cursor`. The simulation must be indistinguishable from live operation as far as the agent is concerned.

### 8.3 Playbook evolution during backtests

This is the key difference from V1–V4. The playbook at month 1 of the backtest is different from the playbook at month 6, because the agent has learned from its intervening trades. This means:

- Early trades are made with the initial (weaker) playbook — expect worse performance
- Later trades benefit from accumulated journal entries and refined rules
- The system's learning rate is itself a testable property — does the playbook actually improve trade quality over time?
- Playbook snapshots are saved at each review point, creating a versioned history that can be analyzed post-hoc

This also means **you can't run the backtest in parallel across time windows.** Each window depends on the state accumulated from previous windows. The simulation is inherently sequential.

### 8.4 Cost management for backtests

A 2-year backtest with 3 analyses/day = ~2,200 LLM calls. At ~10k tokens each + daily macro context, that's roughly 25M tokens — significant cost at frontier model prices.

Strategies to manage this:

**Cheaper model for backtesting.** Use a smaller, faster model (e.g. Claude Haiku, GPT-4o-mini) for backtests, frontier model for live trading. The backtest validates the *system* (architecture, memory, learning loop), not the model's maximum reasoning capability. If the system works with a cheaper model, it'll work better with a frontier model.

**Skip-to-trigger simulation.** Don't simulate every hour. Pre-compute the screening layer's triggers across the whole dataset (this is vectorized and cheap). Then only advance the cursor to timestamps where a screen would have fired. This might cut 80% of the simulation steps.

**Chunked backtests.** Run 3-month chunks instead of a full 2-year pass during development. Use the full simulation for final validation only.

**Replay mode.** Record the agent's live paper-trading decisions. Later, replay the exact same market data through a modified agent to compare approaches. The first run is expensive; replays only re-run the changed components.

### 8.5 Validation checks

**The honesty problem.** V1–V4's most important lesson: results that look good are usually wrong. Look-ahead bias, overfitting, and self-deception are the default. The simulation must be validated before trusting its results.

**Checks:**

- **Future data audit.** For every LLM call in the simulation, verify that no input data has a timestamp after `time_cursor`. Log all input timestamps and assert `max(input_timestamps) <= time_cursor` for every call.
- **Paper trading agreement.** Run the simulation over a period where the agent was also paper-trading live. Compare the simulation's decisions to the live agent's decisions. They should be similar (not identical — model stochasticity means some divergence is expected, but the general pattern should match).
- **Comparison to random.** For every N trades the agent makes, generate N random entries during the same period. The agent must demonstrably beat random to justify its existence.
- **Regime diversity.** Check that the agent's edge isn't concentrated in one market condition. If it only makes money in bull markets, it's not robust.
- **Trade-level review.** Not just aggregate metrics — examine individual trades. Did the rationale hold? Was the entry at a genuinely structural level, or did the agent confabulate a justification?
- **Playbook versioning.** Save and review playbook snapshots. If it drifts toward increasing complexity without better results, something is wrong. Good learning converges on simplicity.
- **Learning curve.** Plot trade quality (win rate, avg R:R) over time during the backtest. If the learning loop works, later trades should be measurably better than early ones.

### 8.6 What backtesting can and cannot tell you

There's a fundamental tension between the simulation and the live system that must be understood clearly.

**What the simulation answers:**
- Does the learning architecture work? (Does trade quality improve over the course of the simulation?)
- Is the system better than random? (Compare to random entries over the same period.)
- Does the system work across regimes? (Is edge clustered or distributed?)
- Is the pipeline correct? (Future data audit, no look-ahead.)

**What the simulation CANNOT answer:**
- Is the agent's *current* knowledge any good? The current playbook was shaped by the market data it has already seen. Running that playbook over the same data is testing on training data. For validating "how good is the agent right now," only unseen data works.

**How to validate current knowledge:**

**Paper trading (gold standard).** Point the agent at the live market with its current playbook and learning enabled. Wait. There is no shortcut. This is why paper trading in Phase 2 runs for 4–8 weeks — that's the minimum to accumulate enough out-of-sample trades to draw conclusions.

**Frozen-playbook forward test.** Take the current playbook, freeze it (disable learning), and run it alongside the learning-enabled version on live data. This separates two questions: "is the current playbook good?" (frozen) vs. "is the agent still improving?" (learning-enabled). If the learning version pulls ahead over a few weeks, the learning loop is adding real value.

**Periodic holdout windows.** Build holdout discipline into the design from day one. Every quarter, reserve one month of data that the agent never sees — it's excluded from the journal, never used in reviews, never rendered in charts. At the end of the quarter, freeze the current playbook and run the simulation over the held-out month. This provides a regular out-of-sample check without waiting months for paper trading results.

In practice, maintain a rolling holdout: at any time, the most recent 2–4 weeks of data are embargoed. The agent paper-trades through them live (generating real out-of-sample results), and after the embargo period ends, that data becomes available for journal retrieval and playbook review. This way there's always a recent unseen window being tested, and the test results are available frequently rather than quarterly.

**A/B testing.** Run two agents in parallel on live data: one with the current playbook + learning, one with a baseline (initial V1–V4 playbook, no learning). Compare over the same period. This directly measures how much value the accumulated knowledge provides.

**The practical summary:** The simulation engine is the development and architecture validation tool. Paper trading is the real-world validation tool. They answer different questions and both are necessary. The periodic holdout discipline ensures you always have a recent chunk of unseen data to test against — otherwise you're waiting 4–8 weeks of pure paper trading every time you want to know if a change helped.

---

## 9. Risk Management

Risk management is the one area where the agent gets NO discretion. These are hard rules enforced in the execution layer:

| Rule | Value | Rationale |
|------|-------|-----------|
| Max position size | X% of portfolio | No single trade should be catastrophic |
| Stop-loss | Required on every trade | The agent can choose WHERE, but not WHETHER |
| Max concurrent positions | 2–3 | Focus > diversification at this scale |
| Max daily loss | Y% of portfolio | Circuit breaker — stop trading for the day |
| Max drawdown | Z% of portfolio | Circuit breaker — stop trading, trigger strategic review |
| No leverage initially | 1x only | Earn the right to use leverage with a track record |

The agent's ANALYZE layer proposes entries, exits, stop levels, and sizes. The EXECUTE layer validates these against the rules and rejects anything that violates them. The LLM never directly places orders — it proposes, and deterministic code validates and executes.

---

## 10. What to Port from V1–V4

### Keep
- **Data pipeline** — `data.py` (OHLCV, funding rates, FNG fetching + parquet cache)
- **Slippage model** — 5bp per side is a reasonable default
- **Look-ahead detection** — `check_signal_integrity()` may still be useful for validating that the agent's journal-based decisions aren't benefiting from future data
- **The knowledge** — V1–V4's findings should seed the initial playbook:
  - Funding rates are a genuine signal (Track D confirmed)
  - Vol regime gates work (Track A confirmed)
  - FNG identifies fear washouts (V4 confirmed, 370bp/trade)
  - Less is more (V3 proved adding complexity hurts)
  - The edge per trade must exceed friction to matter

### Discard
- `compute_signals()` / strategy contract — wrong paradigm
- `evaluate.py` / fitness function — the agent doesn't have a single scalar score
- `optimize_params.py` / Optuna — no parameters to optimize
- Track system / `track_runner.py` — no parallel signal class exploration
- The research loop (`research_loop.py`, `agent.py`) — replaced by the new architecture

---

## 11. Interfaces and Parallel Build

The layers are designed to be decoupled enough to build and test independently. Define the interfaces first, then build each component against mocks of its dependencies.

### 11.1 Interface contracts

```
MarketSnapshot
  OHLCV data up to a point in time (multiple assets, multiple timeframes)
  Funding rates, FNG, BTC dominance at that point
  Produced by: Data Pipeline
  Consumed by: everything

StructuralFeatures
  Key S/R levels, FVG zones, order blocks, liquidity clusters
  Produced by: Market Structure Detection
  Consumed by: Chart Renderer (as annotations), Screen, text summaries

ChartImage[]
  Annotated PNG renders at multiple timeframes
  Produced by: Chart Renderer (snapshot + annotations → images)
  Consumed by: Macro Context, Analyze

MacroBriefing
  ~200-word cached text: big picture, regime, BTC.D, sentiment
  Produced by: Macro Context layer (weekly/daily charts + sentiment → LLM)
  Consumed by: Analyze (prepended to every analysis)

Decision
  { action: entry/exit/wait, conviction, rationale, stop, target, size }
  Produced by: Analyze layer
  Consumed by: Execute, Record

Order
  Validated decision that passed risk checks, ready for exchange
  Produced by: Execute layer
  Consumed by: Exchange API (live) or portfolio sim (backtest)

JournalEntry
  Structured record of observation + decision + outcome + lesson
  Produced by: Record layer
  Consumed by: Review, journal retrieval, oversight

PlaybookUpdate
  Proposed changes to the playbook: new rules, revised confidence, pruned rules
  Produced by: Review layer
  Consumed by: Playbook (applied after human approval in early phases)

ExperimentProposal
  Hypothesis, change, scope, duration, success criteria
  Produced by: Review / Oversight
  Consumed by: Experiment tracker
```

### 11.2 What can be built in parallel

Each of these workstreams depends only on the interface contracts above, not on each other's implementation. They can all be built and tested against mocks/stubs simultaneously.

| Workstream | Builds | Depends on (interface only) | Can mock |
|-----------|--------|---------------------------|----------|
| **A: Data + Charts** | Data pipeline, chart renderer, BTC.D source | — | Nothing to mock — this is the foundation |
| **B: Market Structure** | S/R detection, FVG detection, order blocks, liquidity zones | MarketSnapshot | Feed it sample OHLCV data |
| **C: Screen** | Numeric screening logic, trigger conditions | MarketSnapshot, StructuralFeatures | Feed it snapshots, check trigger/no-trigger |
| **D: Macro Context** | LLM prompt design, briefing generation, caching | ChartImage[] | Use pre-rendered sample charts |
| **E: Analyze** | LLM prompt design, decision output parsing, conviction grading | MacroBriefing, ChartImage[], Playbook, JournalEntry[] | Mock all inputs with sample data |
| **F: Execute** | Risk validation, order construction, portfolio simulation | Decision | Feed it sample decisions, verify rule enforcement |
| **G: Record + Journal** | Journal schema, storage, retrieval (embedding or structured) | Decision | Write/read sample entries |
| **H: Review + Oversight** | Review prompt design, playbook update logic, drift detection, experiment lifecycle | JournalEntry[], Playbook | Feed it a sample journal, check outputs |
| **I: Simulation Engine** | Time cursor, orchestration, future-data enforcement | All interfaces (but calls them abstractly) | Stub every layer, test the orchestration logic |
| **J: Playbook** | Format, versioning, read/write, initial V1–V4 seed content | — | Standalone |

**Workstreams A, B, F, G, J** have no LLM dependency — they're pure code and can be built and fully tested without any API calls.

**Workstreams D, E, H** are LLM-dependent but can be iterated on sample inputs independently of each other. The prompt design for Macro Context doesn't depend on the Analyze prompt being done, and vice versa.

**Workstream I (simulation engine)** is the integration layer. It calls every other component through the interfaces. Build it with stubs that return hardcoded results, then swap in real implementations as they're completed. The orchestration logic (time cursor, scheduling, data windowing) is testable without any real components.

### 11.3 Integration sequence

Once components are built independently, integrate in this order:

```
Step 1: A (data) + B (structure) + J (playbook)
        → Can render annotated charts and produce text summaries from real data

Step 2: + C (screen) + G (journal)
        → Can screen market data and store decisions

Step 3: + D (macro) + E (analyze) + F (execute)
        → Full trading loop works end-to-end (on sample data, manually triggered)

Step 4: + I (simulation engine)
        → Can run the full loop on historical data with time cursor

Step 5: + H (review + oversight)
        → Learning loop closes — agent improves during simulation runs
```

Each step is independently testable before proceeding to the next.

---

## 12. Build Phases

With the parallel workstreams defined, the build phases become about integration milestones rather than sequential construction.

### Phase 0 — Interfaces + Parallel Build
- Define all interface contracts (§11.1) as types/schemas in code
- Set up new repo with directory structure matching the workstreams
- Kick off all 10 workstreams in parallel
- **Exit criteria:** each workstream has a working implementation that passes tests against mocked inputs/outputs

### Phase 1 — Integration + Simulation Validation
- Integrate workstreams per §11.3 (steps 1–4)
- Run the simulation engine over a 3-month historical chunk with a cheaper model
- Validate: time cursor correct? Charts render at each point? Future data excluded? Journal accumulates?
- Run future data audit (§8.5)
- Manual review of simulation decisions for sanity
- **Exit criteria:** end-to-end simulation runs without errors, future data audit passes

### Phase 2 — Paper Trading + Learning Loop
- Integrate step 5 (review + oversight)
- Point the system at live data — same architecture, now watching the real market
- Run for 4–8 weeks of paper trading with learning + oversight active
- Track confidence calibration metrics and drift indicators (§6.4)
- Cross-validate: run simulation over the same period and compare to live decisions
- **Exit criteria:** agent has accumulated 30+ paper trades, learning curve shows improvement

### Phase 3 — Full Backtest + Validation
- Run full simulation over 2+ years of historical data with learning active
- Compare to random baseline (§8.5)
- Check regime diversity, playbook evolution, individual trade quality
- **Exit criteria:** demonstrably beats random, edge not concentrated in one regime, playbook converged on simplicity

### Phase 4 — Live Trading (if Phase 3 passes)
- Implement exchange integration in Execute layer
- Hard risk management guardrails (no LLM in execution path)
- Start with minimal position sizes
- Gradually increase as track record builds

---

## 13. Open Questions

1. **Which LLM?** Frontier models (Claude, GPT-4) reason better but cost more. Smaller models are cheaper but may miss nuance. The screening layer needs nothing; the analysis layer needs the best reasoning available; the review layer is infrequent enough to use frontier.

2. **Chart rendering infra.** TradingView Lightweight Charts rendered via headless browser (§6.2). Phase 0 deliverable: an HTML template + Puppeteer script that takes OHLCV JSON + annotations and produces a PNG. Needs to support overlays for S/R levels, FVG zones, order blocks, and liquidity clusters.

3. **How to prevent confabulation?** LLMs can construct convincing rationales for any decision. If the agent says "I see accumulation at this level" — is that real, or is the model pattern-matching to its training data? The journal + review loop should surface this over time, but it's a real risk.

4. **How long until the playbook is useful?** The initial playbook (seeded from V1–V4) covers sentiment and structural signals. But the market-structure setups (FVGs, order blocks) need to be learned from experience. This could take weeks or months of paper trading. Patience is required.

5. **Single asset or multi-asset?** Start with BTC. It has the most liquidity, the most data, and the most established patterns. Expand to ETH and alts only after demonstrating BTC proficiency.

6. **How to handle the meta?** The "current meta" (what's driving the market right now) is the hardest thing to systematize. It changes weekly or even daily. The monthly strategic review is meant to address this, but it might not be fast enough. Consider feeding the agent relevant news/narrative summaries as context.

---

*This document will evolve as the system is built and tested. The key commitment: intellectual honesty above all. If the agent doesn't beat random after 3 months of paper trading, we stop and reassess rather than tweaking until it "works."*
