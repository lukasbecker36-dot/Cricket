# Project handover — IPL/T20 win-probability trading

Last updated: 2026-05-26

This is a working handover for picking up the project mid-stream. Read CLAUDE.md
first for the architectural principles — they're non-negotiable. This document
covers **where we are**, **what we tried**, and **what to do next**.

## TL;DR

- **Strategy A (T20 phase Lines)**: trading 6/10/15/20-over Innings Runs Line markets. **Strict OOS validation (2025+) changed our view — see "OOS reckoning" below.** Only **phase_15 has robust OOS edge**; phase_6 needs a trend-aware prior fix; phase_10 is weak.
- **Strategy B (test cricket lay-the-draw)**: validated on 137 historical Betfair test Match Odds markets (2022-2026). **Market overestimates draws by ~8pp** in the post-Bazball era. Naive returns +50% ROI on stake / +5.5% on capital; filtered to implied ≥15% returns **+11.7% ROI on capital**. Not live yet; waiting for ENG-NZ series. **This is the most robust strategy we have.**
- **Weather features**: added via Open-Meteo (`src/live/weather.py`). Help phase_15 for **hot-climate leagues (IPL/PSL)** where data is dense. For England/NTB the signal is **unvalidated** — only ~50 hot English matches exist in all data, so weather predictions there are untrustworthy (treat as human context, not a model input). Now uses **anomaly features** (deviation from venue climatology) to remove the temperature-as-league-proxy confound. Powerplay (phase_6) is unaffected by weather.
- **Live workflow**: screenshot-driven Telegram bot (no Betfair API). User screenshots a market, Claude vision extracts it, model scores it, bot replies with the signal.
- **Lesson from live trading**: extreme-confidence signals (`|p − 0.5| > 0.30`) are unreliable. But ALSO — see OOS reckoning: even the mid-band edge for phase_6/10 didn't survive strict OOS until we added trend awareness.
- **Open question**: whether to bypass the Hetzner→Betfair IP block (if it is one) to automate price ingestion instead of screenshots. Diagnostic script ready (`scripts/diagnose_betfair_access.py`).

## OOS reckoning (2026-05-26) — READ THIS

We ran a strict walk-forward check: train phase models on seasons ≤2024, test on 2025+ Line markets (truly unseen). This **overturned earlier conclusions** that were based on full-sample backtests inflated by in-sample fit.

```
                  full ROI   mid-band ROI   (2025+ OOS, trained ≤2024)
─────────────────────────────────────────────────
phase_6  baseline  +9.5%     -7.3%   ⚠ mid-band LOSES
phase_10 baseline  +12.5%    -5.6%   ⚠ mid-band LOSES
phase_15 baseline  +32.7%    +14.2%  ✓ robust
phase_15 weather   +37.2%    +21.9%  ✓ weather helps
```

**Why phase_6/10 decayed** (investigated, see `scripts/investigate_phase_decay.py`):
- Powerplay scoring is inflating fast: PP avg went 45.5 (2020) → 53.1 (2025) → 56.8 (2026)
- The market sets PP lines too LOW — in 2025 the actual goes OVER 62% of the time. Naively backing OVER every PP market in 2025 returns +20% ROI.
- The edge IS real, but our model points the wrong way: its priors are backward-looking averages that lag the rising trend, so it bets UNDER and loses.
- **Fix found** (`scripts/trend_aware_phase6.py`): adding a `league_trend` feature (league's PP avg in the prior season) recovers phase_6 from +9.5%/-7.3% to **+14.9%/-0.2%** OOS. The trend feature is what matters; recency-weighting alone doesn't help.

**Consolidated OOS backtest at £5 stake (train ≤2024, eval 2025+, full signal set), `scripts/oos_backtest_production.py`:**

```
phase           n    P&L    ROI     win   trust
─────────────────────────────────────────────────
full_innings    73  +£162  +44.2%   74%   🟢 strong (trend-fixed)
phase_15       113  +£196  +34.6%   69%   🟢 robust (anomaly-weather)
phase_10       153  +£112  +14.7%   59%   🟡 decent
phase_6        235  +£132  +11.2%   57%   🟡 full-set only (mid-band NEGATIVE)
phase_6_inn2    57  +£183  +64.2%   84%   🔴 inflated by chase-end artifact
phase_10_inn2   45  +£184  +82.0%   93%   🔴 inflated by chase-end artifact
─────────────────────────────────────────────────
TOTAL               +£968
inn1-only (4 mkts)  +£602 over 574 trades = +21% ROI  ← honest deployable number
```

**TRADING STANCE (corrected):**
1. **Trade the FULL signal set, not the mid-band.** The earlier "mid-band [0.05,0.30] only" advice was WRONG — mid-band overall is barely positive (+£41) and phase_6 mid-band loses (−6%). Profit lives in the full set including high-confidence signals.
2. **Lead with full_innings (20-over) and phase_15** — the two strongest, most trustworthy edges.
3. **phase_10 decent, phase_6 full-set-only** (weakest, mid-band negative).
4. **inn2 ROI (64%/82%) is inflated** by the chase-end settlement asymmetry (market settles on final total when chase ends early). Real edge is lower; trade small/sceptically.
5. Caveats: LTP-based (no slippage beyond 5% commission), 2.0 odds assumed — real fills shave ROI.

**All models corrected for scoring-inflation lag (the biggest hidden flaw).** full_innings was the worst case: par for the Hampshire test moved 141 → 175 after the fix (market 187, actual 200 — old model would have lost badly backing under). The earlier "+24-44% ROI" full-sample headlines were in-sample inflated; the £5 OOS table above is the honest read.

**The scoring-inflation lag was the single biggest hidden problem.** Every phase model used backward-looking equal-weight priors that systematically underestimated the modern game. The fix (recency-weighted priors + a `league_trend` feature = the league's prior-season average) is deployed everywhere. Re-apply it to any new model.

## Project evolution (what we tried, what worked)

The project went through three distinct strategy iterations before landing on
Innings Runs Line markets.

### Phase 1 — Chase win probability on Match Odds (abandoned)

Original goal was a second-innings chase win-probability model trading IPL
Match Odds. Built a clean walk-forward LightGBM with hierarchical player
shrinkage. **Result: no consistent cross-league edge.** The market for Match
Odds in T20 is too efficient; our model didn't move the needle versus the
market-implied probability.

Code: still present under `src/features/`, `src/models/` — useful for state
representation but no longer the primary path.

### Phase 2 — Multi-runner "X Runs or more" ladders (abandoned)

Switched to ladder markets where each rung is a different threshold ("180 Runs
or more", "200 Runs or more", etc.). Backtest showed real edge (~+15% ROI),
but lay-side liquidity was so thin you couldn't actually execute the trades.
The model was right, the market wasn't tradeable.

Code: `src/live/signals.py:evaluate_market` still scores ladder markets; not in
active use.

### Phase 3 — Innings Runs Line markets (CURRENT)

These are single-line Over/Under markets at ~2.0 odds with real two-way
liquidity. Backtested four phases:

| Phase | Target balls | Backtest ROI (2025+ OOS) | Confidence |
|---|---|---|---|
| 6 Overs Line | 36 (powerplay) | +24% (n=244) | Good |
| 10 Overs Line | 60 | +25% (n=148) | Good |
| 15 Overs Line | 90 | +44% (n=107) | High (but smaller sample) |
| Full Innings Line (20 over) | 120 | included in above | Good |

These are **innings-1 only**; innings-2 phase markets have separate models —
see "Innings-2 model" below.

## Architecture

Respect the four-layer boundary documented in CLAUDE.md (ingestion, features,
model, execution). Cross-layer leakage is a code smell.

Key code paths:

```
data/processed/                  # Cricsheet parquet, partitioned by league
data/raw/betfair/                # historical Betfair .dat archives
                                 # (51MB + 57MB, downloaded from Google Drive
                                 # by user — see chat history for links)
models/
  full_innings_*                 # 1st-innings 20-over line model
  phase_{6,10,15}_*              # 1st-innings phase line models
  phase_{6,10}_inn2_*            # 2nd-innings phase line models (target-aware)

src/live/
  signals.py                     # FullInningsModel class, load_models(), 
                                 # model_for_market_name() (innings-aware)
  chat_runner.py                 # Telegram bot driving the screenshot flow
  vision.py                      # Anthropic vision -> structured market extraction
  config.py                      # env-loaded LiveConfig (no Betfair creds)
  telegram.py                    # Telegram client

scripts/
  train_full_innings_model.py    # original full-innings trainer
  train_phase_models.py          # inn1 phase model trainer
  train_phase_models_inn2.py     # inn2 phase model trainer (target-aware)
  evaluate_phase_lines.py        # inn1 backtest
  evaluate_phase_lines_inn2.py   # inn2 backtest (walk-forward)
  evaluate_line_combined.py      # combined backtest across all phases
  diagnose_inn2_signals.py       # confidence-band + distribution diagnostics
  diagnose_betfair_access.py     # checks if Hetzner is IP-blocked from Betfair

deploy/
  README.md                      # Hetzner setup guide (screenshot bot only)
  live.env.example               # TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                                 # ANTHROPIC_API_KEY (no Betfair creds)
```

## Validation methodology

Walk-forward only. CLAUDE.md mandates this; don't deviate.

- **Inn1 phase models**: priors are strictly-prior-season (`season < S`), 
  booster is fit on all data with 90/10 split for early stopping. Match-level
  leakage is impossible because the only match-specific input is the per-season
  team prior, which excludes that match's own season.
- **Inn2 phase models**: walk-forward training (cutoff = season 2023). Critical
  because `target` is now a feature; the model could otherwise memorize 
  `(target, team, venue, season) → outcome`. Eval restricts to season > 2023.

When testing a new feature, **always** confirm the backtest holds up under the
strict OOS slice (2025+ matches only). The "all-time" backtest can be inflated
by training-set leakage even with the prior structure.

### Backtest realism

Every backtest must include:
- 5% Betfair commission on net winnings
- LINE_ODDS = 2.00 (confirmed live by user, not 1.92)
- Filter to T-1 minute pre-inplay line value (no in-play leakage)
- Walk-forward season cutoff where appropriate

## Live trading workflow (current production)

User screenshots a Betfair market in Telegram, bot extracts structured fields
via Anthropic vision, model scores it, bot replies with a `BACK_OVER` /
`BACK_UNDER` / `SKIP` recommendation. No Betfair API calls from the server.

Why: original Betfair API attempts from Hetzner returned 403, attributed
(perhaps incorrectly) to data-center IP blocking. The screenshot workaround
sidesteps that entirely.

Configuration: `deploy/live.env.example` documents the env vars. Run via
systemd unit (see `deploy/README.md`).

## Innings-2 model (target-aware)

Built only for phase_6 and phase_10. Reasoning:

- **inn2 phase totals settle on the final score** if the chase ends or the team
  is bowled out — Betfair confirmed this via market rules (see chat history,
  user screenshot of rules page). Markets void **only** on rain-shortened
  innings reduced below the stipulated overs.
- **phase_15 inn2 and full-innings inn2** suffer from too much chase-end
  selection (target hit before phase boundary → settles low) for the model to
  separate signal from artifact reliably.
- The new model adds `target`, `phase_par_from_target`, and `x_minus_target_par`
  features. The booster can split on `target × bowl_prior` to learn pressure
  dynamics.

Walk-forward eval (trained ≤2023, evaluated 2024+):
```
phase_6  OOS: n=56  ROI=+63.7%  win=83.9%
phase_10 OOS: n=45  ROI=+77.7%  win=91.1%
```

**These headline numbers are not credible at face value.** The diagnostic
(`scripts/diagnose_inn2_signals.py`) revealed bimodal model_p distributions —
roughly 60% of trades sit at p<0.1 or p>0.9. Restricting to the credible
confidence band `|p − 0.5| ∈ [0.05, 0.30]` gives:

```
band [0.05, 0.30] phase_6: n=21  ROI=+39%  win=71%
band [0.05, 0.30] phase_10: n=8   ROI=+22%  win=63%
```

This is what real edge looks like and matches the live trading outcomes so far.

## Strategy B — Test cricket lay-the-draw

**Status**: validated on historical data, NOT yet traded live. First live attempts will be the ENG vs NZ 3-test series.

### The structural finding

Across 137 tests with Betfair Match Odds data (2022-2026):

| Market implied draw % | Actual draw rate | n |
|---|---|---|
| 5% | 0% | 28 |
| 10% | 4% | 27 |
| 15% | 7% | 27 |
| 23% | 20% | 30 |
| 40% | 20% | 25 |
| **Overall: 18%** | **10%** | **137** |

The market systematically overestimates test draw probability by ~8pp in the post-Bazball era. Naive "lay every draw at T-60s" returned +£695 P&L (+50.7% ROI on stake) across the 137 tests; capped-liability + filter to implied ≥15% returned +£280 P&L at +11.7% ROI on capital with max £75 drawdown.

### v1 trading rule (ENG vs NZ)

```
For each test, ~60 min before scheduled start:
- Check Betfair Match Odds market for "The Draw" runner
- If implied draw probability ≥ 15% (lay price ≤ 6.7): LAY
- Stake = £25 / (lay_price - 1), capped at £50 max
- Cap liability at £25 per trade
- If implied < 15%: SKIP
```

### Code

- `scripts/phase0_lay_the_draw.py` — Cricsheet outcome distribution + oddspapi probe
- `scripts/extract_test_draw_data.py` — pulls T-60s pre-inplay draw prices from `data/raw/betfair/betfair_all_markets.dat`, joins with Cricsheet outcomes. Output: `data/processed/test_draws.parquet`
- `scripts/simulate_lay_the_draw.py` — strategy comparison + sizing calculator
- `docs/LAY_THE_DRAW_SCOPE.md` — full scope document with phase plan

### Why no model (v1)

- Naive "implied ≥15%" filter already captures most of the structural edge
- Only 14 draws across 137 trades — training data too thin for a meaningful binary classifier
- ENG-NZ is just 3 trades; better to validate the strategy live first
- Build a model later (Strategy B v2) when we have 200+ test trades to learn from

### Inn2 models now trend-corrected (2026-05-26)

`phase_6_inn2` and `phase_10_inn2` were the last models on stale priors. Now
rebuilt with recency-weighted priors (half-life 2) + `league_trend` feature,
keeping their target-awareness. Companions: `phase_{6,10}_inn2_league_trend.json`.
The fix corrected the par-line direction: for Essex chasing 201 at Rose Bowl,
the 6-over inn2 par moved 51 → 55 (was implausibly below the inn1 par; now matches
it), and 10-over moved 89.5 → 97 (chase acceleration now reflected). The full-signal
OOS backtest was unchanged (+63%/+82%) because it's dominated by the chase-end
settlement-asymmetry exploit, not the mid-band where the lag mattered — so the
par-line direction is the better evidence here. Trained on all seasons for production.

### Pending for Strategy B

1. **Forward-validate on ENG-NZ** (3 tests). Paper-trade if uncertain about live execution.
2. **Confirm Betfair Match Odds settlement on rain-abandoned tests** — likely "The Draw" wins, making rain a real risk. Skip matches with bad forecasts.
3. **Persistent paper-trade log** — capture lay price, stake, liability, outcome
4. **Consider a model in v2** after 200+ live tests have accumulated

## Live P&L log

Real and paper trades made via the screenshot bot:

| Trade | Market | Stake | Result | Notes |
|---|---|---|---|---|
| GT v CSK inn1 6-over | UNDER | £5 (live) | LOST -£5.00 | extreme-confidence signal |
| GT v CSK inn1 20-over | OVER | £5 (live) | WON +£4.75 | |
| GT v CSK inn2 6-over | UNDER | (skipped) | would-win | model gave +33pp on inn1 model — correctly skipped because inn1 model has no target awareness |
| SRH v RCB inn2 6-over | UNDER @66.5 | £5 (live) | LOST -£5.00 | extreme-confidence (30pp edge), wrong direction |
| Glam v Glouc 20-over | UNDER @171.5 | £5 (paper) | WON +£4.75 | mid-confidence (7pp edge), as backtest predicted |
| Hampshire v Essex 6-over | UNDER @55.5 | £5 (live) | WON +£4.75 | hot/dry conditions argued OVER; model UNDER right (51/2). Market over-corrected for weather. |

Running: **+£4.25 over 5 settled trades.**

**Caution on the "extreme-confidence loses" pattern**: it held for the first 4
trades but Hampshire (an extreme +32pp signal) won. With 5 trades the live
sample is statistically meaningless. Trust the OOS backtest (180+ trades) over
the live sample: the honest signal is that phase_6/10 mid-band had no robust
edge until the trend-feature fix, and phase_15 is the reliable phase.

## Known pitfalls / lessons

1. **Always wait for toss before signalling.** Pre-toss the model has to
   guess which team bats first. KKR v MI example in chat history: pre-toss
   call flipped from lay to back when MI won the toss and chose to bat.

2. **Watch for line-extraction artifacts.** During backtest cleanup, ~33% of
   inn2 rows had implausible line values (e.g. 989.5 for a 6-over PP line).
   `LINE_BOUNDS` dict in `evaluate_phase_lines_inn2.py` filters these.

3. **Bookmaker spread + commission.** Default 5% commission baked in. Real
   spreads on Line markets are usually 1 run; tighter than expected. But
   illiquid markets can show 10–100 run spreads — skip those.

4. **Innings-2 selection bias.** Cricsheet only captures innings-2 phase
   totals when the chase reached the phase boundary. The training data must
   include all chases (using final inn2 total when chase ended early) to
   avoid biasing toward chases that batted out the phase.

5. **Don't trust extreme-confidence signals.** Backtest diagnostics revealed
   bimodal `model_p` distribution. Live trades confirmed: the extreme bets
   lose more than they should. Restrict to `|p − 0.5| ∈ [0.05, 0.30]`.

6. **Plausibility checks on lines.** PP lines outside [25, 130] are usually
   defaults (zero matched volume). Skip them.

## Dead ends — do not re-investigate without strong cause

- **Match Odds model** (Phase 1): no consistent edge, abandoned.
- **Multi-runner ladder markets** (Phase 2): real edge but no lay-side 
  liquidity. Don't try to lay these.
- **oddspapi.io as data source**: confirmed Betfair Exchange coverage is
  *only* Match Odds, 1X2, Tied, Coin Toss, Completed. **No phase Lines, no
  innings totals.** Diagnostic in chat history; their generic market schema
  has thousands of "Over Under" definitions but none populated for Betfair
  cricket.
- **Innings-2 full-innings and phase_15 models**: chase-end selection bias
  is too strong; the inn2-trained phase_15 backtest showed 100% win rate on
  17 trades which is clearly an artifact.

## Pending work / next steps

### Immediate

1. **Deploy the trend-aware phase_6 model.** Validated in `scripts/trend_aware_phase6.py`
   (recency+trend recovers OOS from +9.5%/-7.3% to +14.9%/-0.2%). To ship:
   - Add `league_trend` + `x_minus_trend` features to `FullInningsModel.predict_p`
     in `src/live/signals.py` (needs the league's prior-season PP/phase average
     saved alongside the model)
   - Retrain production phase_6 with recency-weighted priors + trend feature
   - Re-run OOS to confirm before live use

2. **Test the trend feature on phase_10.** Same scoring-inflation lag likely
   applies. If it recovers phase_10 the way it did phase_6, revive phase_10.

3. **Adopt the weather model for phase_15.** Validated +7.7pp OOS in mid-band.
   `scripts/train_phase_models_with_weather.py` produces `phase_15_wx_*`. Needs
   weather features added to `FullInningsModel.predict_p` and live weather
   lookup (already built in `src/live/weather.py`) wired into `chat_runner.py`.

4. **Wire confidence-band + line-plausibility filters into `chat_runner.py`**
   so the live bot stops emitting untrustworthy signals. Spec:
   - `0.05 ≤ |p − 0.5| ≤ 0.30` → emit (but note: even mid-band only robust for phase_15)
   - PP line outside [25, 130] → skip
   - 10-over line outside [50, 175] → skip
   - 15-over line outside [80, 250] → skip
   - Full innings line outside [120, 280] → skip

5. **Add weather context to every signal** regardless of model — the live bot
   should show conditions so the human can apply judgement (the Hampshire trade
   showed the market can over-correct for weather; human eyes catch this).

6. **Run `scripts/diagnose_betfair_access.py` on Hetzner** to confirm whether
   the 403 was IP-block or just an expired session token. If reachable, build
   a streaming Betfair client and ditch the screenshot path.

### Weather pipeline (built 2026-05-26)

- `src/live/weather.py` — Open-Meteo geocoding + lookup, ~50 curated venue coords
- `scripts/backfill_weather.py` — historical weather for training data → `data/processed/match_weather.parquet` (2,902 rows, 95/166 venues, 75% match coverage)
- `scripts/train_phase_models_with_weather.py` — trains `phase_{N}_wx` models
- `scripts/compare_phase_models_with_without_weather.py` — v1-vs-v2 backtest
- `scripts/oos_weather_check.py` — strict 2025+ OOS isolating weather contribution
- `scripts/anomaly_weather_oos.py` — per-league OOS comparing baseline vs absolute vs anomaly weather. Showed absolute weather is **inert for cool leagues** (BBL/NTB identical to baseline) because hot-weather data is 98% IPL/PSL. Anomaly features lift cool-league ROI +22%→+30% but on tiny samples (n=15).
- **71 venues failed geocoding** (minor English county grounds). Add their coords
  to `VENUE_OVERRIDES` to lift coverage 75% → ~95%.

### Production models deployed (2026-05-26)

`scripts/train_production_models.py` trains and deploys:
- **phase_6, phase_10, full_innings**: recency-weighted priors (half-life 2 seasons) + `league_trend` feature. Companion: `{prefix}_league_trend.json`. Corrects the scoring-inflation lag.
- **phase_15**: anomaly-weather features (deviation from venue climatology). Companion: `phase_15_venue_climo.json`. `weather_mode: anomaly` in meta.
- NOTE: full_innings now uses `_bat_prior.json` not the old `_bat_pp.json` — the old `_pp` files were removed so the loader picks up the new trend priors. `signals.py` tries `_bat_pp` first then `_bat_prior`.
- `scripts/trend_aware_phase6.py` validates the trend fix per-phase (env `TREND_PHASE=phase_6|phase_10`).

`FullInningsModel.predict_p` (in `src/live/signals.py`) now supports both: it loads the trend table + venue climatology if present, accepts a `weather=` dict, and computes `league_trend`/`x_minus_trend` and anomaly features automatically. Backward-compatible — models without these companions behave as before.

**Backups of pre-trend/weather models** in `models/_backup_pre_trend_weather/`.

**Weather caveat (important):** the anomaly design is correct and removes the league-proxy confound, but English/NTB weather remains statistically unvalidated (too few hot English matches). Per-match weather predictions for England are directionally untrustworthy — a hot day still nudges P(over) *down* at test points, which is backwards. Use weather as human context for English cricket, not as an acted-on model input.

### Decay investigation (built 2026-05-26)

- `scripts/investigate_phase_decay.py` — year-by-year market sharpness, model
  accuracy, scoring-distribution shift, league mix. This is what diagnosed the
  scoring-inflation lag. Re-run it periodically to monitor for new decay.

### Medium-term

3. **Build a target-aware full-innings inn2 model** — current full-innings is
   inn1-trained and has no target awareness, so inn2 full-innings line trades
   are skipped entirely. With the diagnostic-confirmed mid-confidence band as
   the trading regime, the chase-end bias should be more manageable.

4. **Persistent paper-trading log.** Currently the P&L lives in this document
   and chat history. Add `data/paper_trades.parquet` that the bot appends to
   on every signal, with one row per emitted signal + a settlement column
   updated post-match.

5. **Per-league prior adjustment.** Match Odds backtest showed cross-league
   transfer is real but noisy. T20 Blast (NTB) priors are particularly thin —
   only ~14 games per team per season. Consider league-weighted shrinkage.

### Long-term (only if previous steps confirm continued edge)

6. Move from £5 flat stake to **quarter-Kelly with a 5% cap** once 30+ trades
   have been logged. Kelly analysis from earlier chat history is in
   `data/processed/kelly_sizing_analysis.txt` (or recompute).

7. **Replace screenshot path with Betfair streaming** (only if step 2 confirms
   Hetzner can reach the API). The screenshot path is robust but slow — by
   the time you've captured + sent + received, the price can have moved.

## Important context for whoever picks this up

- The user is **trading real money** — bias toward conservatism over feature
  breadth. Every backtest result reported should include realistic costs.
- Telegram bot identity, ANTHROPIC_API_KEY etc. are stored on the Hetzner box
  in `/etc/cricket/live.env` (not in the repo).
- The Betfair historical .dat archives in `data/raw/betfair/` were downloaded
  by the user via Google Drive — links in chat history. They're not in git
  (too large) but the backtest scripts need them.
- All paper-trade signals go through the same `Signal` dataclass in
  `src/live/signals.py`. Don't add a parallel signal path.
- The Hetzner deployment has been live since the screenshot pivot. Don't break
  it with refactors unless asked.

## How to continue from here

1. Read `CLAUDE.md` (architectural principles, validation requirements).
2. Read this document.
3. Check the most recent commits on `claude/ipl-win-probability-model-DNUwf`
   for the latest state.
4. If the user asks about a market, default to: ask for toss/batting team if
   not provided → score with the appropriate phase model → report only if 
   confidence band is `[0.05, 0.30]` and line is plausible → otherwise SKIP.
5. When in doubt, **flag a backtest result as suspect** rather than trusting
   the headline number. The bimodal-distribution lesson taught us this is the
   default failure mode of this strategy class.
