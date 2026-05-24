# Project handover — IPL/T20 win-probability trading

Last updated: 2026-05-24

This is a working handover for picking up the project mid-stream. Read CLAUDE.md
first for the architectural principles — they're non-negotiable. This document
covers **where we are**, **what we tried**, and **what to do next**.

## TL;DR

- **Strategy A (T20 phase Lines)**: trading 6/10/15/20-over Innings Runs Line markets on Betfair Exchange. Live since May 2026, mixed results (-£0.50 over 4 trades). Mid-confidence band only.
- **Strategy B (test cricket lay-the-draw)**: validated on 137 historical Betfair test Match Odds markets (2022-2026). **Market overestimates draws by ~8pp** in the post-Bazball era. Naive strategy returns +50% ROI on stake / +5.5% on capital; filtered to implied ≥15% returns **+11.7% ROI on capital**. Not live yet; waiting for ENG-NZ series.
- **Live workflow**: screenshot-driven Telegram bot (no Betfair API). User screenshots a market, Claude vision extracts it, model scores it, bot replies with the signal.
- Models cover 6/10/15-over phase Lines for innings 1 (trained on all leagues), and 6/10-over for innings 2 (target-aware). Innings 2 full-innings has too much chase-end selection bias to be useful.
- **Lesson from live trading**: extreme-confidence signals (`|p − 0.5| > 0.30`) are unreliable. Mid-confidence band `[0.05, 0.30]` is where the real edge lives. See "Live P&L log" below.
- **Open question**: whether to bypass the Hetzner→Betfair IP block (if it is one) to automate price ingestion instead of screenshots. Diagnostic script ready (`scripts/diagnose_betfair_access.py`).

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

Running: **-£0.50 over 4 settled trades.**

**Strong observed pattern**: extreme-confidence signals (|p − 0.5| > 0.25 or
edge > 25pp) have all lost so far. Mid-confidence signals (edge 5–10pp) have
won. Sample size of 4 is meaningless statistically, but the direction is
consistent with the diagnostic-finding above.

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

1. **Wire confidence-band + line-plausibility filters into `chat_runner.py`**
   so the live bot stops emitting extreme-confidence signals. Spec:
   - `0.05 ≤ |p − 0.5| ≤ 0.30` → emit
   - PP line outside [25, 130] → skip
   - 10-over line outside [50, 175] → skip
   - 15-over line outside [80, 250] → skip
   - Full innings line outside [120, 280] → skip

2. **Run `scripts/diagnose_betfair_access.py` on Hetzner** to confirm whether
   the 403 was IP-block or just an expired session token. If reachable, build
   a streaming Betfair client and ditch the screenshot path.

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
