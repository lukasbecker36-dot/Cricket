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
    bowler_economy: pd.Series  # runs per over
    bowler_balls: pd.Series


SHRINK_PRIOR_BALLS = 60  # ~10 overs faced; conservative
SHRINK_PRIOR_BOWLER_BALLS = 60
LEAGUE_SR_MEAN = 130.0
LEAGUE_ECON_MEAN = 8.2


def compute_player_stats(balls: pd.DataFrame, up_to_season: int) -> PlayerStats:
    """Aggregate batting/bowling stats from seasons < up_to_season."""
    past = balls[balls["season"] < up_to_season]
    if past.empty:
        empty = pd.Series(dtype=float)
        return PlayerStats(empty, empty, empty, empty)

    bat_legal = past[past["is_legal_delivery"]]
    bat_runs = bat_legal.groupby("striker")["runs_batter"].sum()
    bat_balls = bat_legal.groupby("striker").size()
    bat_sr = (bat_runs / bat_balls * 100.0).fillna(0.0)

    bowl = past[past["is_legal_delivery"]]
    bowl_runs = bowl.groupby("bowler")["runs_total"].sum()
    bowl_balls = bowl.groupby("bowler").size()
    bowl_econ = (bowl_runs / bowl_balls * 6.0).fillna(0.0)

    return PlayerStats(
        batting_strike_rate=bat_sr,
        batting_balls_faced=bat_balls,
        bowler_economy=bowl_econ,
        bowler_balls=bowl_balls,
    )


def shrunk_strike_rate(stats: PlayerStats, player: str, league_mean: float = 130.0) -> float:
    sr = stats.batting_strike_rate.get(player)
    n = stats.batting_balls_faced.get(player, 0)
    if sr is None or n == 0:
        return league_mean
    return (n * sr + SHRINK_PRIOR_BALLS * league_mean) / (n + SHRINK_PRIOR_BALLS)


def shrunk_economy(stats: PlayerStats, player: str, league_mean: float = 8.2) -> float:
    econ = stats.bowler_economy.get(player)
    n = stats.bowler_balls.get(player, 0)
    if econ is None or n == 0:
        return league_mean
    return (n * econ + SHRINK_PRIOR_BOWLER_BALLS * league_mean) / (n + SHRINK_PRIOR_BOWLER_BALLS)
