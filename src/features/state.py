"""Match state at a single ball during a chase. Pure data, no I/O."""
from __future__ import annotations

from pydantic import BaseModel


class ChaseState(BaseModel):
    """Snapshot of an in-progress second-innings chase."""

    match_id: str
    season: int
    venue: str
    target: int
    runs_scored: int
    wickets_lost: int
    legal_balls_bowled: int
    striker: str
    non_striker: str
    striker_balls_faced: int
    non_striker_balls_faced: int
    bowler: str
    # recent-form windows (computed by replay_chase)
    runs_last_12_balls: int = 0
    wickets_last_18_balls: int = 0
    boundaries_last_over: int = 0
    # bowler -> legal balls already bowled by them in this innings
    bowlers_used: dict[str, int] = {}
    # Legal balls the current striker+non-striker pair has played together. Resets on wicket.
    partnership_balls: int = 0
    label: int | None = None  # 1 if chasing team won, 0 if not; None for live

    @property
    def phase(self) -> int:
        """0=powerplay (overs 0-5), 1=middle (6-14), 2=death (15-19)."""
        over_idx = self.legal_balls_bowled // 6
        if over_idx < 6:
            return 0
        if over_idx < 15:
            return 1
        return 2

    @property
    def runs_required(self) -> int:
        return max(self.target - self.runs_scored, 0)

    @property
    def balls_remaining(self) -> int:
        return max(120 - self.legal_balls_bowled, 0)

    @property
    def wickets_remaining(self) -> int:
        return max(10 - self.wickets_lost, 0)

    @property
    def required_run_rate(self) -> float:
        if self.balls_remaining <= 0:
            return float("inf") if self.runs_required > 0 else 0.0
        return self.runs_required / self.balls_remaining * 6.0

    @property
    def current_run_rate(self) -> float:
        if self.legal_balls_bowled <= 0:
            return 0.0
        return self.runs_scored / self.legal_balls_bowled * 6.0
