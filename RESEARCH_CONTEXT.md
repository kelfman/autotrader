# Autotrader Research Loop — Agent Briefing

This document gives a self-contained briefing for any Claude session running a
research iteration on the BTC/USDT trading strategy. Read this first, then
read `strategy.py` and `experiment_log.jsonl` before proposing anything.

---

## Project Goal

Discover a BTC/USDT 1h trading strategy that performs consistently across
multiple market regimes (bull runs, choppy markets, bear phases). The fitness
function rewards consistent Sharpe across 5 non-overlapping 3-month windows
and penalises variance.

**Fitness formula:**
```
fitness = mean(sharpe_across_5_windows) − 0.5 × std(sharpe_across_5_windows)
```

Windows with fewer than 3 trades are marked invalid (⚠) and their Sharpe is
treated as 0. Avoid changes that zero out windows.

---

## Workspace Location

The Autotrader directory is mounted from the user's machine. Find it dynamically:

```python
import glob
autotrader = glob.glob('/sessions/*/mnt/Autotrader')[0]
```

Or in bash:
```bash
AUTOTRADER=$(find /sessions -maxdepth 3 -name "Autotrader" -path "*/mnt/*" -type d 2>/dev/null | head -1)
```

**Python venv:** `$AUTOTRADER/.venv`  — activate with `source $AUTOTRADER/.venv/bin/activate`

---

## Files

| File | Purpose |
|------|---------|
| `strategy.py` | **The only file you edit** (V1). Contains PARAMS dict + compute_signals() |
| `evaluate_only.py` | Runs backtest and prints JSON to stdout. Run with `2>/dev/null` to suppress logs |
| `experiment_log.jsonl` | Append-only log of every V1 iteration (accepted and rejected) |
| `data_cache/BTC_USDT_1h.parquet` | 2 years of cached 1h BTC OHLCV data |
| `data_cache/BTC_USDT_USDT_funding.parquet` | 2 years of cached 8h funding rate data (Track D) |
| `RESEARCH_CONTEXT.md` | This file |
| `track_runner.py` | V2 multi-track runner — creates isolated run dirs per track |
| `track_config.py` | Pydantic TrackConfig schema for per-track agent prompts |
| `tracks/` | JSON config files for each research track (A–E) |
| `runs/track_X_<name>/` | Per-track isolated strategy.py, backup, and experiment log |
| `data.py` | OHLCV + funding rate fetcher with parquet cache |
| `validate_holdout.py` | Held-out validation — tests strategies on gaps between training windows |
| `oversight.py` | Research oversight layer — 5 compliance checks + LLM direction reviewer (§5.12) |
| `validate_walkforward.py` | Walk-forward validation — train on W1-3, test on W4-5 (§5.13) |
| `runs/synthesis_D_A/` | Bayesian-optimized D+A synthesis (§5.15) — best project strategy |
| `optimize_params.py` | Bayesian parameter optimization via Optuna TPE (§5.15) |
| `runs/synthesis_D_A_B_C/` | Stage 2 synthesis with B/C boolean switches (§5.15.6) |
| `runs/synthesis_D_A/optuna_*.db` | Optuna study storage (SQLite, resumable) |

---

## Strategy Structure Contract

`strategy.py` must always contain:

1. A `PARAMS: dict` with named numeric parameters
2. A `compute_signals(df, params) -> (entries, exits)` function

**Do not change the function signature or remove the `PARAMS` variable.**

The `ta` library is used (NOT pandas-ta). Key APIs:
```python
import ta
ta.trend.ema_indicator(close, window=N)      # EMA
ta.momentum.rsi(close, window=N)             # RSI
ta.trend.adx(high, low, close, window=N)     # ADX
ta.trend.adx_pos(high, low, close, window=N) # DI+
ta.trend.adx_neg(high, low, close, window=N) # DI-
ta.volatility.bollinger_hband(close, window=N, window_dev=2)  # BB upper
ta.volatility.average_true_range(high, low, close, window=N) # ATR
ta.volume.on_balance_volume(close, volume)   # OBV
ta.trend.macd(close, window_slow, window_fast, window_sign)   # MACD line
ta.trend.macd_diff(close, window_slow, window_fast, window_sign) # MACD histogram
ta.trend.macd_signal(close, window_slow, window_fast, window_sign) # Signal line
```

---

## How to Run an Iteration

### Step 1 — Activate venv and get baseline
```bash
AUTOTRADER=$(find /sessions -maxdepth 3 -name "Autotrader" -path "*/mnt/*" -type d 2>/dev/null | head -1)
cd $AUTOTRADER
source .venv/bin/activate
python evaluate_only.py 2>/dev/null
```
Parse the JSON output. Key fields: `fitness`, `mean_sharpe`, `std_sharpe`, `windows` (list of per-window results with `sharpe`, `n_trades`, `win_rate`, `is_valid`).

### Step 2 — Backup current strategy
```bash
cp $AUTOTRADER/strategy.py $AUTOTRADER/strategy.py.bak
```

### Step 3 — Propose and apply a change
Edit `strategy.py` directly (use the Edit tool). Either:
- **Parametric:** change a value in PARAMS
- **Structural:** rewrite or augment compute_signals()

### Step 4 — Re-evaluate
```bash
python evaluate_only.py 2>/dev/null
```

### Step 5 — Accept or revert
- If `new_fitness > baseline_fitness`: keep it ✓
- If `new_fitness <= baseline_fitness`: revert — `cp $AUTOTRADER/strategy.py.bak $AUTOTRADER/strategy.py`

### Step 6 — Log the result
Append one JSON line to `experiment_log.jsonl`:
```json
{
  "iteration": <next integer>,
  "timestamp": "<ISO UTC>",
  "accepted": true/false,
  "fitness_before": <float>,
  "fitness_after": <float>,
  "delta": <float>,
  "change_type": "parametric" or "structural",
  "rationale": "<your reasoning>",
  "params": {<full PARAMS dict after change>},
  "has_new_code": true/false,
  "fitness_result": {"fitness": <f>, "mean_sharpe": <f>, "std_sharpe": <f>}
}
```

---

## Experiment History Summary

The log in `experiment_log.jsonl` has the full record. 55 iterations completed; 6 accepted, 49 rejected.

| Iter | Accepted | Δ fitness | Change |
|------|----------|-----------|--------|
| 1 | ✓ | +0.352 | ADX > 20 entry filter (biggest single jump) |
| 4 | ✓ | +0.049 | EMA50 macro filter (close > EMA50) |
| 7 | ✓ | +0.193 | RSI must be rising at entry (`rsi > rsi.shift(1)`) |
| 8 | ✓ | +0.251 | ADX-fading exit: exit when ADX < 20 for 1 bar |
| 9 | ✓ | +0.112 | 2-bar hysteresis on ADX exit (prevents premature exits) |
| 10 | ✓ | +0.047 | rsi_entry_min: 50 → 55 |
| 11–55 | ✗ | — | 45 consecutive rejections; TA ceiling reached |

**Best fitness ever recorded: +0.4835** (iteration 10, initial session)

**Effective current baseline: +0.4714** — when the loop was resumed in a new session,
re-evaluation of the same strategy.py produced +0.4714 instead of +0.4835. This is a
~0.012 drift from the rolling data window updating as time passed. All iterations 11–55
used +0.4714 as the acceptance threshold.

Current per-window results at the +0.4714 baseline:
- W1 (Mar–Jun 2024, BTC ATH distribution): Sharpe **-0.390**, 7 trades, 29% WR ← persistent weak link
- W2 (Sep–Dec 2024, BTC bull run to $100k): Sharpe **+1.561**, 3 trades, 67% WR
- W3 (Feb–May 2025): Sharpe **+1.026**, 3 trades, 67% WR
- W4 (Jul–Oct 2025): Sharpe **+1.175**, 3 trades, 67% WR
- W5 (Dec 2025–Mar 2026): Sharpe **+0.702**, 7 trades, 57% WR

**W1 is the structural problem.** Mar–Jun 2024 is a post-ATH distribution / choppy sideways
regime. The strategy generates 7 trades with 29% win rate in that window — more than double
any other window — because EMA crossovers produce false signals in non-trending markets.
Every attempt over 45 iterations to fix W1 either did nothing or broke W2–W5.
This is the expected behaviour: EMA crossover is a reflexive TA signal that depends on
other participants watching the same chart. It structurally underperforms in regimes where
that crowd is absent.

---

## What Has Been Tried and Failed

Read `experiment_log.jsonl` for the full list. The following have all been tested and rejected.

**Parametric tightening (multiple attempts each):**
- **ADX threshold 22, 23, 25**: kills W2/W3 trades (too few remain). Tried in iters 11, 12, 15, 21, 27–29, 31–32, 36–37, 43–44, 46–47, 49.
- **rsi_entry_min 58, 60**: over-filters. Tried paired with higher ADX many times; never improved.
- **rsi_entry_min 50, 52 (relaxed)**: more trades, but worse quality. All rejected.
- **ema_trend raised to 60, 100**: kills W2/W4 (too few qualifying bars). Iters 29, 37.
- **ADX threshold 25**: kills W2/W3 (0 trades). Tried as early as iter 2, re-confirmed iters 11–49.
- **rsi_exit_overbought raised to 75**: holds trades too long, gives back gains.

**Exit-side structural rewrites (the dominant failure mode, iters 16–55):**
- **ATR-based trailing stop (all variants)**: The agent proposed replacing/supplementing
  the `trend_fading` exit with an ATR trailing stop in at least 25 distinct iterations
  (16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 30, 31, 33, 34, 35, 36, 38, 39, 40,
  41, 42, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55). None improved on baseline.
  The agent lost awareness of this repetition due to the 15-entry history window.
- **Removing trend_fading exit entirely**: Tried as a standalone change many times; always
  degraded performance. The 2-bar ADX hysteresis exit (trend_fading) is load-bearing.
- **Minimum holding period / cooldown after exits**: Mentioned in multiple iters; never helped.
- **ATR-based entry volatility filter**: Avoid entries when ATR too high/low. Tried iters 34, 36, 39. Rejected.

**Entry-side structural rewrites:**
- **Bollinger Band width filter (expanding vol at entry)**: Iters 13–14. Killed W2/W3.
- **BB upper band breakout at entry**: Too rare at crossover time; empties W2/W3.
- **Mean-reversion (BB lower touch entry)**: Iters 25–26. Catastrophic failure (−3.03, −2.52).
- **Slower EMA pairs (15/30, 14/30)**: Iters 12, 39. W1 worse, W3 drops to 2 trades.
- **RSI rising for 2 bars (vs 1)**: Iter 40. No improvement.
- **EMA200 macro filter**: Iter 34. Kills W2/W4.
- **Price above recent swing low at entry**: Iter 30. Degraded.
- **Require price above EMA slow by a margin**: Multiple iters. Degraded.

**Previously logged failures (iters 1–10 search phase):**
- **MACD rising at entry**: redundant with EMA crossover.
- **RSI exit at 65**: takes profits too early from W4/W5.
- **EMA100/EMA75 macro filter**: kills W2/W4.
- **ADX exit threshold 15**: too loose, W1 returns to −0.797.
- **3-bar ADX hysteresis**: holds losing trades too long.
- **ADX period 10**: too many signals, all windows crash.
- **1-bar EMA crossover confirmation delay**: helps W1/W2 but breaks W4 badly.
- **DI+/DI− directional filter**: reduces std nicely but crushes mean (W2 drops to +1.03).
- **OBV accumulation filter**: over-restricts, empties windows.
- **Volume 1.5× multiplier**: W3 goes to 0 trades.
- **EMA fast declining 2 bars as exit**: cuts W5 winners early.
- **RSI < 40 oversold exit**: panic sells hurt W1 further.
- **N-bar high breakout at entry**: loses W5.
- **RSI period 10 (faster)**: flips W1 positive but destroys W2/W4.

---

## Status and Next Steps

**The TA ceiling is established.** 55 iterations with 45 consecutive rejections confirms that
the EMA-crossover framework is exhausted. The +0.4714 baseline is the TA floor for V2
comparison — every V2 track must beat this to be considered an improvement.

### V2 Multi-Track Results (March 2026)

V2 introduced parallel research tracks exploring different signal classes. Results:

| Track | Signal Class | Iterations | Accepted | Best Fitness | vs TA Baseline |
|-------|-------------|-----------|----------|-------------|----------------|
| A | Statistical / vol regime | 35 | 6 | +0.3484 | −26% |
| B | Calendar / session | 20 | 1 | +0.0000 | Not viable |
| C | Cross-pair (BTC/ETH) | 13 | 5 | +0.2727 | −42% |
| **D** | **Funding rates** | **23** | **10** | **+1.0567** | **+124%** |
| E | TA baseline (V1) | 55 | 6 | +0.4714 | — |

**Track D is the first validated V2 signal class.** It uses perpetual futures funding rate
percentile rank as an entry filter on a slow SMA momentum signal. The mechanism is structural
(funding payments create genuine positioning pressure) and regime-robust (all 5 windows
profitable, including the W1 choppy regime that permanently defeated TA).

**Track D final strategy:** `sma_period=192, fr_pct_window=720, fr_entry_pct=0.52,
fr_exit_pct=0.90, exit_lookback=38`. See `runs/track_D_funding_rates/strategy.py`.

**Held-out validation results (March 2026):**

Track D was evaluated on the 4 gap windows (~69 days each) between training windows:
- H1 (Jun–Sep 2024): Sharpe −1.475, H2 (Dec–Feb 2025): −2.040, H3 (May–Jul 2025): −1.577, H4 (Oct–Dec 2025): −1.235
- **Held-out fitness: −1.7280** (all 4 windows negative)
- The V1 TA baseline also fails held-out: −0.4657 (all windows negative)
- Both strategies degrade on the transition periods between regime windows

The training-window results remain valid (Track D genuinely outperforms TA across the 5 training regimes), but the strategy does not generalise to out-of-sample gap periods. This blocks V3 promotion.

### V2 Re-run Results with Oversight Layer (March 2026)

Tracks A–C were re-run with the §5.12 research oversight layer (anti-convergence
constraints, exploration phase protocol, compliance checks). Signal fidelity was
100% across all tracks — the drift that plagued the original runs was eliminated.

| Track | Signal Class | Iterations | Accepted | Best Fitness | Signal Fidelity |
|-------|-------------|-----------|----------|-------------|-----------------|
| A (re-run) | Vol regime (GK + vol-of-vol) | 20 | 4 | +0.3110 | 20/20 |
| B (re-run) | Calendar/session | 20 | 1 | +0.00 | 20/20 |
| C (re-run) | Cross-pair (BTC/ETH) | 20 | 1 | +0.00 | 20/20 |

**Track A best strategy:** Garman-Klass realized vol + vol-of-vol stability filter.
Entry when vol percentile < 0.47 AND vol-of-vol percentile < 0.40 (calm + stable
regime). Exit when vol percentile > 0.65. `vol_lookback=24, percentile_window=1440,
vov_lookback=168`. Fitness +0.311 — below TA baseline but a genuine vol-regime signal.

**Track C best non-degenerate proposals (all rejected against 0.0 degenerate):**
The champion 0.0 fitness came from a broken strategy (param key mismatch causing 0 trades).
The most promising genuine approach was vol-conditioned ratio mean-reversion with
signal-based exits: `z_lookback=72, z_entry=-1.8, z_exit=-0.3, max_hold=16,
vol_max_pctile=0.20`. This scored -0.402 (mean_sharpe=+0.52, std=1.85) with 3/5
windows positive (W1: +3.13, W4: +0.88, W5: +1.54) but W2 (-2.23, 8 trades) and
W3 (-0.71, 8 trades) dragged fitness negative. The ratio mean-reversion signal is
fundamentally weak — it only works in stable-correlation regimes.

### Walk-Forward Validation (March 2026)

Walk-forward validation (train on W1-W3, test on W4-W5) resolves the held-out
impasse. Both Track D and the TA baseline pass with positive OOS fitness:

| Strategy | In-Sample (W1-3) | OOS (W4-5) | Decay |
|----------|-----------------|------------|-------|
| Track D (funding) | +1.588 | **+0.660** | 58% |
| Track E (TA baseline) | +0.321 | **+0.779** | -143% |

Track D's OOS fitness (+0.660) beats the TA baseline's full 5-window fitness
(+0.471). The funding rate signal generalizes forward.

### Manual Synthesis: Track D + A (March 2026)

First synthesis attempt: AND-gating funding rate filter with vol-of-vol regime gate.
Result: full fitness -0.615, OOS +0.547. The vol gate blocks trades in W1 where
funding signals work best. **Naive AND-gating hurts rather than helps.**

### Bayesian Synthesis Optimization (March 2026)

Replaced LLM-driven parametric search with Optuna TPE for the D+A synthesis.
200 trials in 89s (0.4s/trial) — the equivalent of 200 LLM iterations.

**Stage 1 — D+A joint optimization:**
- **Best fitness: +2.030** (mean_sharpe=+2.420, std=0.780)
- All 5 windows profitable: W1 +2.349, W2 +1.571, W3 +3.344, W4 +3.274, W5 +1.563
- Walk-forward OOS (W4-5): **+1.991** with only 3.3% decay
- Held-out (gap windows): −3.139 (same structural issue as §5.11)
- Stability: FRAGILE (vol_pct_window ±10% → 55% fitness drop)

Key discovery: jointly-optimal thresholds differ from standalone optima. Vol filter
tightened (0.455 vs 0.65 manual) — doing heavy lifting for regime selection. Funding
filter relaxed (0.688 vs 0.52 standalone) — no longer filtering alone. This interaction
effect was invisible to 15+ iterations of LLM-guided parametric search.

**Optimized params:** `sma_period=256, fr_pct_window=408, fr_entry_pct=0.6879,
fr_exit_pct=0.7784, exit_lookback=60, vol_lookback=53, vol_pct_window=1104,
vol_entry_pct=0.4553, vol_exit_pct=0.5766`. See `runs/synthesis_D_A/strategy.py`.

**Parameter importance:** vol_pct_window (55.2%), fr_pct_window (18.9%), exit_lookback
(6.1%), sma_period (5.0%). Vol regime lookback is the single most consequential param.

**Stage 2 — B/C marginal tests (100 trials):**
Best fitness +0.928 — well below D+A's +2.030. Session filter (Track B) enabled in
best trial but zeroed out W2 (0 trades). Ratio filter (Track C) consistently off.
**B and C signals confirmed as noise in the combined D+A context.**

See `runs/synthesis_D_A_B_C/strategy.py` and `optimize_params.py`.

**Final comparison:**

| Strategy | Full Fitness | WF OOS (W4-5) | WF Decay |
|----------|-------------|----------------|----------|
| Track E (TA baseline) | +0.471 | +0.779 | −143% |
| Track D (funding standalone) | +1.064 | +0.660 | +58% |
| D+A manual synthesis | −0.615 | +0.547 | N/A |
| **D+A Bayesian optimized** | **+2.030** | **+1.991** | **+3.3%** |

### Robustness Analysis (March 2026)

The single-best optimum (fitness +2.030) was fragile: 55% worst-case drop from ±10%
perturbation. Two approaches tested to address this:

| Approach | Full Fitness | WF OOS | WF Decay | Worst Drop |
|----------|-------------|--------|----------|------------|
| Single-best | +2.030 | +1.991 | 3.3% | 55.0% |
| Ensemble (top-20 avg) | +1.845 | +1.816 | 3.8% | 40.7% |
| Robust optimization | +1.851 | +1.209 | 51.0% | 31.3% |

**Ensemble is the recommended approach.** It sacrifices 9% fitness for improved stability
and maintains excellent OOS generalization (3.8% decay). The top-20 trials are tightly
clustered (e.g., fr_entry_pct std=0.01), confirming a genuine basin.

**Robust optimization** found more stable optima (31.3% worst drop) but at a steep
generalization cost (51% decay). The perturbation-aware objective overfits to in-sample.

**No approach achieves full stability** (worst drop < 20%). The D+A strategy structure
is inherently parameter-sensitive — this is a structural property, not an optimization
limitation.

Ensemble params: `sma_period=266, fr_pct_window=418, fr_entry_pct=0.6898,
fr_exit_pct=0.7604, exit_lookback=55, vol_lookback=52, vol_pct_window=1070,
vol_entry_pct=0.5088, vol_exit_pct=0.5645`. See `runs/synthesis_D_A/optuna_synthesis_d_a_ensemble20.json`.

### Derivative Signal Tracks (March 2026)

**Track F (open interest):** Blocked. Binance's `openInterestHist` API is hard-capped at
~30 days regardless of period. Needs a paid data source for 730-day backtests.

**Track G (basis spread):** 100 Optuna trials + 4 structural variants. Best fitness +0.281
(z-score < -2.44 entry, 8 total trades in 2/5 windows). Same degenerate-optimization
pattern as Track C — the optimizer makes the entry so extreme that only a few trades fire
in windows where they happen to work. Walk-forward OOS +0.616 is from 5 trades in a
single window. Extremely fragile (106% worst drop). Not viable as standalone signal.

The basis mechanism is real but too weak on hourly BTC candles (basis ±0.3% is small
relative to hourly volatility). May work on lower timeframes or in real-time.

**Status / Next steps:**
1. ✅ All V3 prerequisites met — D+A ensemble achieves OOS +1.816 with 3.8% decay
2. ✅ Robustness addressed — ensemble averaging is the recommended deployment candidate; fragility reduced from 55% to 40.7% worst drop
3. ✅ Derivative tracks explored — Track F blocked by API limits; Track G (basis) weak (+0.281, sparse/degenerate)
4. Calendar (B), cross-pair (C), basis (G) confirmed unviable. Vol regime (A) weak but genuine gate.
5. `optimize_params.py` provides `--preset basis`, ensemble (`--ensemble K`), and robust (`--robust`) modes
6. ✅ **V3 Phase 1 complete.** Multi-timeframe augmentation is the new project best:
   - **V3 MTF ensemble: fitness +2.677, OOS +2.682, 0.5% decay** (see `runs/v3_mtf/strategy.py`)
   - V3 MTF single-best: fitness +3.082, OOS +3.208, negative decay
   - Position sizing (§6.5.2) did not improve over V2 — binary vol gate outperforms continuous sizing
   - Full 2-year backtest: +207.4% return, Sharpe 2.763, max DD 9.5%, DD duration 55 days (vs V2: +14.3%, 0.467, 21.8%, 423 days)
   - Key insight: daily SMA(20) trend filter replaces 1h SMA(266), providing dramatically cleaner trend identification
7. ❌ **V3 MTF results INVALIDATED** — Post-hoc audit found look-ahead bias in `augment_with_timeframes()`: daily close included same-day bars. After fixing (shift d1/h4 features by 1 period) + adding 5bp slippage, BTC 5yr collapses to −32.1% (Sharpe −0.22). The "daily SMA(20) breakthrough" was an artefact of same-day information leakage. V2 D+A ensemble (+1.845) remains the valid project best.
8. ❌ **MTF re-optimization with lagged features** — 200 Optuna trials with fixed augmentation (shift by 1 period) + 5bp slippage. Best fitness +0.778, well below V2 baseline (+1.640). Daily trend filter adds no genuine value when properly lagged. MTF dead end for now.
9. ✅ **Look-ahead guard added** — `check_signal_integrity()` in backtest.py detects daily return correlation > 0.15 as FAIL. Default slippage 5bp. Would have caught the MTF bug on first run.
10. **Next: V3 Phase 2** — Regime switching (§6.5.3) + Strategy portfolio (§6.5.4) building on V2 D+A base (+1.640 with slippage).

**Unexplored TA ideas (low priority — V1 is at ceiling):**

1. **Stochastic RSI**: replace raw RSI with StochRSI for more sensitive momentum.
   `ta.momentum.stochrsi(close, window=14, smooth1=3, smooth2=3)`

2. **Dual-timeframe RSI**: resample close to 4h, compute RSI, align back to 1h index;
   require 4h RSI > 50 as a macro momentum filter.

3. **min_volume_ma parametric sweep**: currently 20, not yet tested at 10, 15, 25, 30.

---

## Agent Guidelines

- **Propose one change per iteration.** Do not bundle multiple changes.
- **Never zero out windows.** If your proposed change would drop a window below 3
  trades, reject it proactively without running the full evaluation.
- **Read experiment_log.jsonl** before proposing to avoid repeating failures.
  In particular: ATR trailing stop replacements for trend_fading have been tried
  ~25 times and never worked. Do not propose this again.
- **Prefer parametric changes** (quick, safe) when unclear what structural change to make.
- **Update this file** at the end of your session — update the "current best fitness"
  and per-window results if a new best was found. Also add any new failures to the
  "What Has Been Tried" section.
- **V1 is at ceiling.** If you are a V2 track agent, this file is provided for
  context only — your task is to explore a different signal class entirely, not to
  continue iterating on TA. Your track config file specifies your signal vocabulary.
