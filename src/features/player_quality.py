"""Empirical-Bayes shrinkage of player stats. Hierarchical: player -> player-type -> league.

Never include the current match in the stats. Callers pass `up_to_season` and the
estimator only looks at strictly-earlier seasons.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PlayerStats:
    """Per-player rolling stats, all from strictly-prior matches."""

    batting_strike_rate: pd.Series  # indexed by batter name
    batting_balls_faced: pd.Series
    batting_boundary_rate: pd.Series  # boundaries (4s+6s) per legal ball
    bowler_economy: pd.Series  # career economy, runs per over
    bowler_balls: pd.Series
    # Phase-specific bowler stats: dicts keyed by (bowler, phase) where phase
    # is 0=powerplay, 1=middle, 2=death. Missing entries fall back to league mean.
    bowler_phase_runs: dict
    bowler_phase_balls: dict
    league_phase_economy: dict  # phase -> league-wide economy from same window


SHRINK_PRIOR_BALLS = 60  # ~10 overs faced; conservative
SHRINK_PRIOR_BOWLER_BALLS = 60
SHRINK_PRIOR_PHASE_BALLS = 30  # phase samples are sparser; smaller prior
LEAGUE_SR_MEAN = 130.0
LEAGUE_ECON_MEAN = 8.2
LEAGUE_BOUNDARY_RATE = 0.135  # ~13.5% of legal balls go for 4 or 6 in T20


def over_to_phase(over: int) -> int:
    if over < 6:
        return 0
    if over < 15:
        return 1
    return 2


def compute_player_stats(balls: pd.DataFrame, up_to_season: int) -> PlayerStats:
    """Aggregate batting/bowling stats from seasons < up_to_season."""
    past = balls[balls["season"] < up_to_season]
    if past.empty:
        empty_s = pd.Series(dtype=float)
        return PlayerStats(
            batting_strike_rate=empty_s,
            batting_balls_faced=empty_s,
            batting_boundary_rate=empty_s,
            bowler_economy=empty_s,
            bowler_balls=empty_s,
            bowler_phase_runs={},
            bowler_phase_balls={},
            league_phase_economy={0: LEAGUE_ECON_MEAN, 1: LEAGUE_ECON_MEAN, 2: LEAGUE_ECON_MEAN},
        )

    bat_legal = past[past["is_legal_delivery"]]
    bat_runs = bat_legal.groupby("striker")["runs_batter"].sum()
    bat_balls = bat_legal.groupby("striker").size()
    bat_sr = (bat_runs / bat_balls * 100.0).fillna(0.0)
    bat_boundaries = bat_legal.assign(is_b=bat_legal["runs_batter"].isin([4, 6])).groupby(
        "striker"
    )["is_b"].sum()
    bat_boundary_rate = (bat_boundaries / bat_balls).fillna(0.0)

    bowl = past[past["is_legal_delivery"]].copy()
    bowl_runs = bowl.groupby("bowler")["runs_total"].sum()
    bowl_balls = bowl.groupby("bowler").size()
    bowl_econ = (bowl_runs / bowl_balls * 6.0).fillna(0.0)

    bowl["phase"] = bowl["over"].apply(over_to_phase)
    phase_runs = bowl.groupby(["bowler", "phase"])["runs_total"].sum().to_dict()
    phase_balls = bowl.groupby(["bowler", "phase"]).size().to_dict()
    league_phase = (
        bowl.groupby("phase").apply(lambda g: g["runs_total"].sum() / len(g) * 6.0).to_dict()
    )
    league_phase = {int(k): float(v) for k, v in league_phase.items()}
    for p in (0, 1, 2):
        league_phase.setdefault(p, LEAGUE_ECON_MEAN)

    return PlayerStats(
        batting_strike_rate=bat_sr,
        batting_balls_faced=bat_balls,
        batting_boundary_rate=bat_boundary_rate,
        bowler_economy=bowl_econ,
        bowler_balls=bowl_balls,
        bowler_phase_runs=phase_runs,
        bowler_phase_balls=phase_balls,
        league_phase_economy=league_phase,
    )


def phase_shrunk_economy(stats: PlayerStats, bowler: str, phase: int) -> float:
    league = stats.league_phase_economy.get(phase, LEAGUE_ECON_MEAN)
    balls = stats.bowler_phase_balls.get((bowler, phase), 0)
    if balls == 0:
        return league
    runs = stats.bowler_phase_runs.get((bowler, phase), 0)
    econ = runs / balls * 6.0
    return (balls * econ + SHRINK_PRIOR_PHASE_BALLS * league) / (balls + SHRINK_PRIOR_PHASE_BALLS)


def shrunk_strike_rate(stats: PlayerStats, player: str, league_mean: float = 130.0) -> float:
    sr = stats.batting_strike_rate.get(player)
    n = stats.batting_balls_faced.get(player, 0)
    if sr is None or n == 0:
        return league_mean
    return (n * sr + SHRINK_PRIOR_BALLS * league_mean) / (n + SHRINK_PRIOR_BALLS)


def shrunk_boundary_rate(
    stats: PlayerStats, player: str, league_mean: float = LEAGUE_BOUNDARY_RATE
) -> float:
    rate = stats.batting_boundary_rate.get(player)
    n = stats.batting_balls_faced.get(player, 0)
    if rate is None or n == 0:
        return league_mean
    return (n * rate + SHRINK_PRIOR_BALLS * league_mean) / (n + SHRINK_PRIOR_BALLS)


def shrunk_economy(stats: PlayerStats, player: str, league_mean: float = 8.2) -> float:
    econ = stats.bowler_economy.get(player)
    n = stats.bowler_balls.get(player, 0)
    if econ is None or n == 0:
        return league_mean
    return (n * econ + SHRINK_PRIOR_BOWLER_BALLS * league_mean) / (n + SHRINK_PRIOR_BOWLER_BALLS)
