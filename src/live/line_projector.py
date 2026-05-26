"""Project the model's par lines for an innings.

A 'par line' is the threshold X where the model says P(total > X) = 0.5 —
i.e. the model's fair Over/Under line. These are MODEL fair values, not
market lines; the user compares them to Betfair's offered lines to spot edge.

Used by the Telegram bot to push inn1 lines on toss detection and inn2 lines
once the inn1 score (target) is known.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.live.signals import FullInningsModel, load_models


@dataclass(frozen=True)
class PhaseProjection:
    phase: str          # '6 over' / '10 over' / '15 over' / '20 over (full)'
    par_line: float     # model's fair Over/Under line (X where P(over)=0.5)
    trustworthy: bool   # False for markets we don't trust (e.g. inn2 15/20)
    note: str = ""


# (label, registry key inn1, registry key inn2, search lo, search hi)
PHASES = [
    ("6 over",        "6",    "6_inn2",  20, 110),
    ("10 over",       "10",   "10_inn2", 40, 175),
    ("15 over",       "15",   None,      60, 250),
    ("20 over (full)", "full", None,     80, 280),
]


def _par_line(model: FullInningsModel, *, batting_team: str, bowling_team: str,
              venue: str, season: int, innings: int, league: str | None,
              target: float | None, weather: dict | None, lo: int, hi: int) -> float:
    """Search X in [lo, hi] for the point where P(over) crosses 0.5."""
    best_x, best_d = float(lo), 1e9
    for X in range(lo, hi + 1):
        p = model.predict_p(
            threshold_X=X, implied_open=0.5,
            batting_team=batting_team, bowling_team=bowling_team, venue=venue,
            season=season, innings=innings, league=league,
            target=target, weather=weather,
        )
        d = abs(p - 0.5)
        if d < best_d:
            best_d, best_x = d, float(X)
    return best_x


def project_innings(
    *, registry: dict[str, FullInningsModel],
    batting_team: str, bowling_team: str, venue: str, season: int,
    innings: int, league: str | None,
    target: float | None = None, weather: dict | None = None,
) -> list[PhaseProjection]:
    """Return par-line projections for every phase of one innings.

    innings=1 uses the inn1 models for all four phases.
    innings=2 uses the target-aware inn2 models for 6/10 overs; 15/20-over
    inn2 markets have no trustworthy model (chase-end selection bias) and are
    flagged trustworthy=False.
    """
    out: list[PhaseProjection] = []
    for label, key1, key2, lo, hi in PHASES:
        if innings == 1:
            model = registry.get(key1)
            trustworthy = True
            note = ""
        else:
            model = registry.get(key2) if key2 else None
            if model is None:
                # no inn2 model for this phase
                out.append(PhaseProjection(label, float("nan"), False,
                                           "no inn2 model (chase-end bias)"))
                continue
            trustworthy = True
            note = ""
        if model is None:
            continue
        par = _par_line(model, batting_team=batting_team, bowling_team=bowling_team,
                        venue=venue, season=season, innings=innings, league=league,
                        target=target, weather=weather, lo=lo, hi=hi)
        out.append(PhaseProjection(label, par, trustworthy, note))
    return out


def format_projections(projs: list[PhaseProjection], header: str) -> str:
    lines = [f"*{header}*", "_model fair lines — compare to Betfair to find edge_", ""]
    for p in projs:
        if not p.trustworthy or (isinstance(p.par_line, float) and np.isnan(p.par_line)):
            lines.append(f"  {p.phase}: — ({p.note})")
        else:
            lines.append(f"  {p.phase}: *{p.par_line:.0f}*")
    return "\n".join(lines)


if __name__ == "__main__":
    reg = load_models(Path("models"))
    print("--- inn1: Hampshire batting vs Essex @ Rose Bowl ---")
    p1 = project_innings(registry=reg, batting_team="Hampshire", bowling_team="Essex",
                         venue="The Rose Bowl", season=2026, innings=1, league="ntb")
    print(format_projections(p1, "1st Innings par lines"))
    print("\n--- inn2: Essex chasing 201 @ Rose Bowl ---")
    p2 = project_innings(registry=reg, batting_team="Essex", bowling_team="Hampshire",
                         venue="The Rose Bowl", season=2026, innings=2, league="ntb", target=201.0)
    print(format_projections(p2, "2nd Innings par lines"))
