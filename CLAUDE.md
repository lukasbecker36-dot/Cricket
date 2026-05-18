# IPL Chase Win Probability Model

## Project goal

Build a live win probability model for IPL second-innings chases, used to identify trading opportunities on Betfair Exchange match odds. The model outputs a probability; trades fire when the market-implied probability disagrees with the model by more than a configurable threshold.

This is a real-money trading system. Bias toward correctness, conservatism, and honest evaluation over speed or feature breadth.

## Core principles

**Honest validation over impressive features.** A simple model with rigorous validation beats a complex model with sloppy validation. Always prefer walk-forward testing over random splits. Never report a backtest result without realistic costs included (commission, slippage, bid-ask spread).

**Reproducibility.** Every model run must be reproducible from a config file plus a data snapshot. Random seeds set explicitly. No notebook-only logic that affects production.

**Fail loudly.** Silent failures are the worst failure mode in trading systems. Missing data, stale prices, model output outside [0,1], unexpected match states — all should raise exceptions, not return defaults.

**No premature optimisation.** Build the simplest version of each component first. Profile before optimising. The model accuracy almost always matters more than runtime.

**Small commits, working code.** Each commit should leave the system in a working state. Don't refactor and add features in the same commit.

## Architecture

The system has four distinct layers; respect the boundaries:

1. **Ingestion** — pulls historical and live data into a normalised internal format. Should be the only layer that knows about external data source quirks.
2. **Features** — converts match state into model inputs. Pure functions, no I/O, fully testable.
3. **Model** — takes feature vectors, outputs win probability. Training and inference paths separated.
4. **Execution** — takes model output and market prices, decides on trades. Has its own risk limits and sanity checks independent of the model.

Cross-layer leakage (e.g., feature engineering inside the model module, or execution logic that re-fetches data) is a code smell.

## Data

### Historical
- Source: Cricsheet (https://cricsheet.org), JSON format preferred over YAML for parsing speed.
- Scope: all IPL matches from 2008 onwards, plus optionally other T20 leagues for player-quality estimation.
- Storage: parquet files in `data/processed/`, partitioned by season.
- Schema: every ball is one row, with columns for match_id, innings, over, ball, batsman, bowler, runs, wicket, etc. Define the schema explicitly in `src/ingestion/schema.py`.

### Live
- Live data source TBD; abstract behind a `LiveFeedAdapter` interface so the implementation can change without affecting downstream code.
- Live feed must be polled, not push-based, for development simplicity. Polling interval: 2-5 seconds.
- Cache every poll response with timestamp for post-match analysis and debugging.
- Expect 5-15 second lag vs actual play; never assume freshness.

### Data quality checks
Every ingested match should pass validation before use:
- Total balls per innings reasonable (115-130 for legal completions)
- No duplicate ball IDs
- Wicket events have valid dismissal types
- Batsman/bowler identifiers resolvable to player records
- Final score matches sum of ball-by-ball runs

Failed validation: log the match ID and exclude, don't silently proceed.

## Modelling approach

### Target
Predict P(chasing team wins) given match state at any point during the second innings.

### State representation
At any point in the chase, state includes:
- Balls remaining
- Wickets remaining
- Runs required
- Current batsmen (striker and non-striker) with quality features
- Bowlers used so far and overs remaining for each
- Venue
- Innings 1 score (target)

### Initial feature set (v1)
Start simple. Don't add features without out-of-sample evidence they help.

- Required run rate
- Current run rate
- Wickets in hand
- Balls remaining
- Balls-faced-by-current-batsmen (proxy for "set" vs "new")
- Batsman strike rate (career, T20 only, regularised toward player-type mean)
- Bowler economy (career, T20, regularised)
- Venue-adjusted par score

### Player quality features
Sample sizes get thin quickly. Use hierarchical shrinkage:
- Individual player career stats shrunk toward player-type stats (opener, finisher, spinner, pacer)
- Player-type stats shrunk toward overall league averages
- Use empirical Bayes for shrinkage parameters

Never use a player's stats from the match being predicted (data leakage).

### Model choice
Start with gradient-boosted trees (LightGBM). Justified because:
- Tabular data
- Handles missing values natively (relevant for early-career players)
- Fast training enables walk-forward validation at scale
- Interpretable feature importance

Neural networks not justified until GBM hits a clear ceiling.

### Calibration
Raw GBM probabilities will be miscalibrated, especially at the extremes. Apply isotonic regression on held-out data after training. Always evaluate calibration with reliability diagrams, not just accuracy.

## Validation

### Walk-forward only
Train on seasons N to N+k, test on season N+k+1. Roll forward. Never use random splits — temporal leakage is the most common failure mode in sports modelling.

### Required metrics
For every model version:
- Log loss (primary metric)
- Brier score
- Calibration error (reliability diagram, ECE)
- Accuracy at multiple probability thresholds
- ROI assuming a simulated trading strategy (with costs)
- Maximum drawdown of the simulated equity curve

A single accuracy number is not sufficient.

### Backtesting realism
The backtest must include:
- Betfair commission (use 5% as conservative default; adjust to actual rate)
- Slippage: assume execution at 1 tick worse than the price seen
- Minimum trade size and liquidity constraints (don't simulate £10k bets at prices showing £500 available)
- Trade timing: model output at time T cannot trade on prices at time T; use T+5 seconds as a safety margin

A backtest result that ignores any of these is not trustworthy.

## Execution

### Out of scope for v1
Do not build live execution until the model is validated. v1 produces signals, logs them, and compares to subsequent market movement. Live trading comes in v2 after at least 1 month of paper trading.

### When v2 is built
- Hard limits: maximum stake per trade, maximum exposure per match, maximum daily loss
- Independent sanity checks: never trade if model output is outside [0.01, 0.99], never trade in the last 6 balls (variance too high), never trade if data is more than 30 seconds stale
- All trades logged with full state at decision time for post-hoc analysis
- Kill switch: a simple way to halt all trading immediately

## Code conventions

### Python
- Python 3.11+
- Type hints on all function signatures
- Pydantic for data validation at I/O boundaries
- No bare except clauses
- Logging via `logging` module with structured fields; no print statements in production code

### Testing
- pytest for all tests
- Unit tests for all feature engineering functions (these are pure, easy to test)
- Integration tests for data ingestion pipelines using small sample files
- A backtest is not a test; it's a validation. Both matter, neither replaces the other.

### Dependencies
Keep them minimal. Core stack:
- pandas / polars for data manipulation (prefer polars for the live path)
- lightgbm for modelling
- scikit-learn for calibration and metrics
- pydantic for schemas
- pytest for testing

Justify any addition beyond this list.

### Git
- Main branch always working
- Feature branches for each component
- Commit messages describe *why*, not *what*
- Never commit data files, credentials, or API keys

## What to avoid

- Building features before the data pipeline is solid
- Adding model complexity before establishing a baseline
- Trusting backtest results without slippage and commission
- Optimising for a single metric (accuracy alone is misleading)
- Live trading before paper trading
- Hardcoding match IDs, player names, or magic numbers — use config files
- Logging that depends on the happy path (logs should be useful when things go wrong)

## Working with Claude Code on this project

When asked to build a component, prefer:
- Writing the schema or interface first, then the implementation
- Adding tests alongside new code, not after
- Asking for clarification on ambiguous requirements rather than guessing
- Flagging when a request would violate the architecture principles above

When evaluating model results, always report:
- The validation methodology used
- All cost assumptions in any ROI number
- Confidence intervals or sample sizes
- Failure cases and edge conditions, not just the headline metric

If a request seems to skip a step (e.g., "let's add a neural network" before the baseline is built and validated), push back and suggest the more disciplined path. The goal is a working trading system, not a feature checklist.
