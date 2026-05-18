"""Feature engineering for v1. Pure functions on ChaseState + cached stats."""
from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from .player_quality import (
    LEAGUE_ECON_MEAN,
    PlayerStats,
    shrunk_economy,
    shrunk_strike_rate,
)
from .rolling_form import recent_sr
from .state import ChaseState

FEATURE_COLUMNS: list[str] = [
    "required_run_rate",
    "current_run_rate",
    "wickets_remaining",
    "balls_remaining",
    "runs_required",
    "striker_balls_faced",
    "non_striker_balls_faced",
    "striker_sr",
    "non_striker_sr",
    "bowler_econ",
    "venue_par",
    "target",
    # recent-form features
    "runs_last_12_balls",
    "wickets_last_18_balls",
    "boundaries_last_over",
    "phase",
    "recent_run_rate",
    # rolling-form features (last 10 prior matches per batter)
    "striker_recent_sr",
    "non_striker_recent_sr",
    # remaining-bowling-resource features
    "remaining_econ",
    "best_bowler_overs_left",
    "n_bowlers_with_overs_left",
]


def venue_par_score(venue_pars: pd.Series, venue: str, league_mean: float = 165.0) -> float:
    val = venue_pars.get(venue)
    if val is None or pd.isna(val):
        return league_mean
    return float(val)


def remaining_bowling_quality(
    bowlers_used: dict[str, int],
    legal_balls_bowled: int,
    stats: PlayerStats,
) -> tuple[float, float, int]:
    """Estimate quality of bowling overs still to be bowled.

    Returns (expected_econ, best_bowler_overs_left, n_bowlers_with_overs_left).

    Logic: each bowler can bowl max 24 legal balls (4 overs). For each bowler who
    has bowled, remaining = 24 - balls_bowled. Total innings has 120 legal balls;
    "unknown" remaining overs (= total - sum of known remaining) get filled at
    the league-mean economy. Final econ = ball-weighted average across known and
    unknown remaining balls.
    """
    total_remaining = max(0, 120 - legal_balls_bowled)
    if total_remaining == 0:
        return LEAGUE_ECON_MEAN, 0.0, 0

    known_remaining_balls = 0
    known_runs_per_ball_weighted = 0.0  # sum of (balls_left * econ/6)
    best_left = 0
    n_with_left = 0
    for bowler, used in bowlers_used.items():
        left = max(0, 24 - used)
        if left == 0:
            continue
        n_with_left += 1
        if left > best_left:
            best_left = left
        known_remaining_balls += left
        econ = shrunk_economy(stats, bowler)
        known_runs_per_ball_weighted += left * (econ / 6.0)

    unknown_balls = max(0, total_remaining - known_remaining_balls)
    unknown_runs = unknown_balls * (LEAGUE_ECON_MEAN / 6.0)
    expected_runs = known_runs_per_ball_weighted + unknown_runs
    expected_econ = expected_runs / total_remaining * 6.0
    return expected_econ, best_left / 6.0, n_with_left


def features_from_state(
    state: ChaseState,
    stats: PlayerStats,
    venue_pars: pd.Series,
    rolling_form: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Build the v1 feature dict from a ChaseState. Fully testable."""
    rrr = state.required_run_rate
    if rrr == float("inf"):
        rrr = 36.0  # cap: > any realistic value, model treats as "lost"
    rf = rolling_form or {}
    rem_econ, best_left, n_left = remaining_bowling_quality(
        state.bowlers_used, state.legal_balls_bowled, stats
    )
    return {
        "required_run_rate": rrr,
        "current_run_rate": state.current_run_rate,
        "wickets_remaining": float(state.wickets_remaining),
        "balls_remaining": float(state.balls_remaining),
        "runs_required": float(state.runs_required),
        "striker_balls_faced": float(state.striker_balls_faced),
        "non_striker_balls_faced": float(state.non_striker_balls_faced),
        "striker_sr": shrunk_strike_rate(stats, state.striker),
        "non_striker_sr": shrunk_strike_rate(stats, state.non_striker),
        "bowler_econ": shrunk_economy(stats, state.bowler),
        "venue_par": venue_par_score(venue_pars, state.venue),
        "target": float(state.target),
        "runs_last_12_balls": float(state.runs_last_12_balls),
        "wickets_last_18_balls": float(state.wickets_last_18_balls),
        "boundaries_last_over": float(state.boundaries_last_over),
        "phase": float(state.phase),
        "recent_run_rate": state.runs_last_12_balls / 12.0 * 6.0,
        "striker_recent_sr": recent_sr(rf, state.match_id, state.striker),
        "non_striker_recent_sr": recent_sr(rf, state.match_id, state.non_striker),
        "remaining_econ": rem_econ,
        "best_bowler_overs_left": best_left,
        "n_bowlers_with_overs_left": float(n_left),
    }


def compute_venue_pars(balls: pd.DataFrame, up_to_season: int) -> pd.Series:
    """Mean innings-1 total by venue, using only strictly-prior seasons."""
    past = balls[(balls["season"] < up_to_season) & (balls["innings"] == 1)]
    if past.empty:
        return pd.Series(dtype=float)
    totals = past.groupby(["match_id", "venue"])["runs_total"].sum().reset_index()
    return totals.groupby("venue")["runs_total"].mean()


def replay_chase(match_balls: pd.DataFrame, label: int) -> Iterator[ChaseState]:
    """Walk a single match's innings-2 balls and yield ChaseState BEFORE each ball.

    The label is the chasing-team outcome and is copied onto every state.
    """
    inn2 = match_balls[match_balls["innings"] == 2].sort_values(
        ["over", "ball", "is_legal_delivery"], ascending=[True, True, False]
    )
    if inn2.empty:
        return
    target = int(inn2["target"].dropna().iloc[0]) if "target" in inn2 and inn2["target"].notna().any() else 0
    venue = str(inn2["venue"].iloc[0])
    season = int(inn2["season"].iloc[0])
    match_id = str(inn2["match_id"].iloc[0])

    runs = 0
    wickets = 0
    legal = 0
    balls_faced: dict[str, int] = {}
    bowlers_used: dict[str, int] = {}

    # history: one entry per delivery (legal or not). Each is (runs, wicket, is_legal, is_boundary)
    history: list[tuple[int, bool, bool, bool]] = []

    def window_stats(legal_window: int) -> tuple[int, int]:
        """Sum runs and wickets over the last `legal_window` legal deliveries."""
        if legal_window <= 0:
            return 0, 0
        legal_seen = 0
        r = 0
        w = 0
        for entry in reversed(history):
            er, ew, eleg, _ = entry
            r += er
            if ew:
                w += 1
            if eleg:
                legal_seen += 1
                if legal_seen >= legal_window:
                    break
        return r, w

    def boundaries_in_last_over() -> int:
        """Count 4s + 6s among the last 6 legal deliveries."""
        legal_seen = 0
        b = 0
        for entry in reversed(history):
            _, _, eleg, is_boundary = entry
            if eleg:
                legal_seen += 1
                if is_boundary:
                    b += 1
                if legal_seen >= 6:
                    break
        return b

    for _, ball in inn2.iterrows():
        striker = str(ball["striker"])
        non_striker = str(ball["non_striker"])
        bowler = str(ball["bowler"])

        runs_12, _ = window_stats(12)
        _, wkts_18 = window_stats(18)
        boundaries = boundaries_in_last_over()

        yield ChaseState(
            match_id=match_id,
            season=season,
            venue=venue,
            target=target,
            runs_scored=runs,
            wickets_lost=wickets,
            legal_balls_bowled=legal,
            striker=striker,
            non_striker=non_striker,
            striker_balls_faced=balls_faced.get(striker, 0),
            non_striker_balls_faced=balls_faced.get(non_striker, 0),
            bowler=bowler,
            runs_last_12_balls=runs_12,
            wickets_last_18_balls=wkts_18,
            boundaries_last_over=boundaries,
            bowlers_used=dict(bowlers_used),
            label=label,
        )

        r = int(ball["runs_total"])
        rb = int(ball["runs_batter"])
        is_legal = bool(ball["is_legal_delivery"])
        is_wicket = bool(ball["wicket"])
        is_boundary = is_legal and rb in (4, 6)
        history.append((r, is_wicket, is_legal, is_boundary))

        runs += r
        if is_legal:
            legal += 1
            balls_faced[striker] = balls_faced.get(striker, 0) + 1
            bowlers_used[bowler] = bowlers_used.get(bowler, 0) + 1
        if is_wicket:
            wickets += 1
