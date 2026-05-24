# Lay-the-Draw test cricket — project scope

## Goal

Build a binary classifier that estimates P(test match draws), compare to
Betfair Exchange draw price, lay the draw when model says it's overpriced.

Target: positive EV with ~5–10% capital ROI per series. Validate on the
upcoming ENG v NZ 3-test series via paper trading before going live.

## What we know upfront

- Cricsheet has ball-by-ball + match outcome data for ~800 tests since 2008
- Outcome field (`info.outcome`) tells us draw vs win directly — no derivation needed
- oddspapi.io covers Betfair Exchange 1X2 (market ID 273) which includes Draw
- We have 250 free oddspapi calls/month — easily enough for test scoring
- Betfair Match Odds liquidity on major test matches is ~£100k+ matched

## Open questions to resolve in Phase 0 (must answer before committing)

| Question | How to resolve | Kills project if |
|---|---|---|
| Are historical Betfair test market archives available? | Check Google Drive / Betfair data downloads | We can't backtest against real Betfair prices; have to use Pinnacle as proxy |
| What is the average draw rate in Cricsheet's test data overall? | Quick pandas script | <10% (too rare to model reliably) or >35% (market knows this; little edge) |
| Per-team-pair draw rate variance — are some pairs ~5% draws and others ~40%? | Group-by analysis | All teams ~25% (no team-level signal) |
| Does oddspapi return the draw runner separately on the 1X2 market for tests? | One test API call against an upcoming test | Schema doesn't expose draw as its own outcome |
| Is the upcoming ENG v NZ series already on Betfair? | Check the Match Odds market exists | We can't paper-trade it |

These are 2 hours of work. Don't skip — historically the project pivots when
one of these answers is unexpected.

## Phase plan (5 phases, ~3 days total once Phase 0 is clean)

### Phase 0 — De-risk (2 hours, before committing further)

1. Verify Cricsheet test outcome distribution:
   ```
   draws / wins / ties / no result, by year, by team-pair
   ```
2. Probe oddspapi.io for a current/upcoming test match — confirm draw runner
   exists and price/liquidity look reasonable
3. Search for historical Betfair test archives (similar to the .dat files we
   have for T20). If unavailable, decide whether Pinnacle proxy is acceptable.

**Go/no-go gate**: if Phase 0 reveals one of the killers above, stop or pivot.

### Phase 1 — Ingestion (half day)

- Download `https://cricsheet.org/downloads/tests_json.zip`
- New script `scripts/ingest_cricsheet_tests.py` that produces ONE row per match
- Don't bother with ball-by-ball; we only need match-level outcomes
- Output: `data/processed/tests/matches.parquet` with columns:
  ```
  match_id, date, season, venue, team1, team2, toss_winner, toss_decision,
  winner (or 'draw' / 'tie' / 'no result'), match_type ('Test')
  ```
- Tag match_type so we never confuse test rows with T20 rows
- ~30 minutes of code, ~10 minutes of validation

### Phase 2 — Features + priors (half day)

New file `src/features/test_draw.py` — pure functions, fully testable. Build:

| Feature | Definition | Why |
|---|---|---|
| `bat_team_draw_rate_prior` | (team1's draw rate in tests with season < S) | Some teams play attacking, some defensive |
| `bowl_team_draw_rate_prior` | (team2's draw rate, season < S) | Same as above for opposition |
| `pair_draw_rate_prior` | (this team-pair's draw rate, season < S) | Captures specific rivalries (England-NZ are aggressive) |
| `venue_draw_rate_prior` | (venue's draw rate, season < S) | Pitch deterioration profile varies hugely |
| `month_of_year` | int 1-12 | Weather/conditions proxy |
| `is_neutral_venue` | bool | Higher draw rate at neutral venues (no home advantage urgency) |
| `toss_winner_chose_bat` | bool | Bat-first teams in tests slightly more draw-prone |

All priors built with strict season cutoff (`season < S`) — same approach as
T20 phase models. **No leakage**.

Walk-forward in this case: use seasons 2008–2022 for training priors, 
hold out 2023–2025 for evaluation.

### Phase 3 — Model + calibration (half day)

`scripts/train_test_draw_model.py`:

- LightGBM binary classifier, `objective="binary"`
- ~7 features, ~600 training matches → small model, no risk of overfit
- Walk-forward: train on `season < S`, predict on season `S`, roll forward
- Save model + meta JSON in `models/test_draw_*`

Calibration step (mandatory for trading):
- Isotonic regression on held-out predictions
- Check reliability diagram — does predicted 30% actually happen 30% of the time?
- ECE (expected calibration error) should be <5pp on held-out data

If calibration is bad, the binary classifier is not safe to bet on — fix it
before backtesting.

### Phase 4 — Backtest + sizing (1 day)

`scripts/backtest_lay_the_draw.py`:

- For each held-out test, score with model and compare to actual draw price
- Compute lay strategy:
  - Lay only when `model_p_draw < market_implied_p_draw - edge_threshold`
  - Test edge thresholds: 3pp, 5pp, 8pp
- Track per-trade P&L with 5% commission
- Report:
  - Total P&L
  - ROI on stake
  - ROI on capital tied up (liability + stake)
  - Maximum drawdown
  - Distribution of trade outcomes
  - Win/loss/draw breakdown

**Sizing analysis** (critical for high-variance lay markets):
- What stake size gives reasonable EV without ruin risk?
- Kelly fraction calculation (likely a quarter-Kelly with hard cap)
- "What if 3 draws in a row" stress test
- Compare flat stake vs Kelly across the backtest

**Go/no-go gate**: backtest must show:
- Positive ROI at 5pp edge threshold
- No more than 30% drawdown in any walk-forward slice
- Calibration ECE < 5pp
- At least 50 trades across the backtest (otherwise too noisy to trust)

### Phase 5 — Live integration (half day)

If Phase 4 passes:

1. Add `src/live/test_draw_signal.py` — equivalent of `signals.py` but for the
   single binary draw market
2. Add a pre-match polling script that calls oddspapi every ~6 hours from
   24h-before to 1h-before scheduled start, scoring the draw price each time
3. Telegram alert when edge exceeds threshold + 5 minutes warm-up
4. Persistent log: `data/test_draw_signals.parquet`

Free tier of oddspapi: 250 calls/month. For ENG v NZ, that's:
- 3 tests × 4 polls/day × 5 days = 60 calls per series at worst
- Well within budget

## What we explicitly will NOT do in v1

- **In-play trading** — too many edge cases (declarations, weather mid-session).
  Pre-match only. Once a match starts, we're stuck with the trade.
- **Cash-out / green-up logic** — increases complexity, marginal EV gain
- **Multi-strategy stacking** (back-the-draw at high prices, lay at low prices) —
  one direction first, prove it works
- **Weather forecasts** — manual judgment for now. Skip matches where the 
  forecast shows >2 days of rain. Maybe v2 adds an automated weather feed.
- **Series context features** (dead rubber, must-win) — leave for v2

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Model badly miscalibrated | Medium | Mandatory calibration check + reliability diagram before betting |
| Single rain-out wipes out edge | Medium | Skip matches with bad forecasts; small stakes per trade |
| Test markets too thin during play | Low (high-profile series) | Confirm pre-match only |
| oddspapi.io draw runner unreliable | Low-medium | Phase 0 sanity check |
| 3 ENG-NZ draws in a row | Low (historical ENG-NZ draw rate ~15%) | Position sizing + Kelly cap |
| Lay liability mismanagement | Medium | Hard cap on liability per match; stake calculator |

## Estimated total effort

- Phase 0: 2 hours
- Phase 1: 4 hours
- Phase 2: 4 hours
- Phase 3: 4 hours
- Phase 4: 8 hours
- Phase 5: 4 hours

**Total: ~3 days of focused work** if Phase 0 doesn't surface a killer.

## Decision points

1. **After Phase 0**: do the numbers (draw rates, oddspapi coverage, archive
   availability) justify continuing? If draws are <10% across the board, the
   market is too skewed and the model probably won't find edge. If oddspapi
   doesn't expose the draw runner, we need a different price source.

2. **After Phase 3 (calibration)**: is the model's reliability good enough to
   trade on? If not, debug — don't proceed to Phase 4.

3. **After Phase 4 (backtest)**: does the simulated ROI clear the
   no-go thresholds? If yes, build live; if no, decide whether to extend
   features (weather, recent form) or shelve.

4. **After ENG v NZ paper trading**: does paper P&L match backtest expectations?
   3 matches is too few to be conclusive, but a -£100 paper outcome warns of
   issues that need diagnosing before any real money.

## Concrete first action

Run Phase 0 — 2 hours. Specifically:

```bash
# A. Cricsheet outcome distribution
python -c "..."  # download + parse + group-by per pair

# B. oddspapi draw runner verification
curl ... | jq ...

# C. Search for historical Betfair test market archives  
```

I can write the Phase 0 script now if you give the OK. It's a 30-line script
that gives us all three answers and tells us whether to commit to the full
build.
