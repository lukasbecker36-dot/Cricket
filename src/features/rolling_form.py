"""Rolling-form features per batter, computed leakage-free.

For each (batter, match), aggregate runs and balls faced over the batter's last
N strictly-prior matches. Cricket-realistic: 10 matches is ~150-200 legal balls
for a regular top-order batter, fewer for the tail. Shrinkage handles the
variable-sample-size problem the same way as career stats.
"""
from __future__ import annotations

import pandas as pd

ROLLING_PRIOR_BALLS = 60
ROLLING_LEAGUE_SR = 130.0


def compute_rolling_batter_form(
    balls: pd.DataFrame,
    matches: pd.DataFrame,
    window_matches: int = 10,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Return a dict keyed by (match_id, batter) -> (recent_runs, recent_balls)
    over the batter's last `window_matches` strictly-prior matches.
    """
    legal = balls[balls["is_legal_delivery"]]
    per_match = (
        legal.groupby(["match_id", "striker"], sort=False)
        .agg(runs=("runs_batter", "sum"), balls=("striker", "size"))
        .reset_index()
    )
    if "date" in matches.columns:
        dates = matches[["match_id", "date"]].drop_duplicates("match_id")
        per_match = per_match.merge(dates, on="match_id", how="left")
        order_col = "date"
    else:
        order_col = "match_id"
    per_match = per_match.sort_values(["striker", order_col]).reset_index(drop=True)

    per_match["recent_runs"] = (
        per_match.groupby("striker")["runs"]
        .transform(lambda s: s.shift(1).rolling(window_matches, min_periods=1).sum())
    )
    per_match["recent_balls"] = (
        per_match.groupby("striker")["balls"]
        .transform(lambda s: s.shift(1).rolling(window_matches, min_periods=1).sum())
    )

    out: dict[tuple[str, str], tuple[float, float]] = {}
    for row in per_match.itertuples(index=False):
        rr = 0.0 if pd.isna(row.recent_runs) else float(row.recent_runs)
        rb = 0.0 if pd.isna(row.recent_balls) else float(row.recent_balls)
        out[(row.match_id, row.striker)] = (rr, rb)
    return out


def recent_sr(
    rolling_form: dict[tuple[str, str], tuple[float, float]],
    match_id: str,
    batter: str,
) -> float:
    """Empirical-Bayes shrunk strike rate from rolling form. Defaults to league mean."""
    rr, rb = rolling_form.get((match_id, batter), (0.0, 0.0))
    if rb == 0:
        return ROLLING_LEAGUE_SR
    return (rr * 100.0 + ROLLING_PRIOR_BALLS * ROLLING_LEAGUE_SR) / (rb + ROLLING_PRIOR_BALLS)
