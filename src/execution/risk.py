"""Independent sanity checks. Run BEFORE any trade decision. Fail loud, default deny."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_stake: float = 200.0
    max_exposure_per_match: float = 1000.0
    max_daily_loss: float = 500.0
    min_prob: float = 0.01
    max_prob: float = 0.99
    min_balls_remaining: int = 6
    max_staleness_seconds: float = 30.0


class RiskRejection(Exception):
    pass


def check_trade(
    model_p: float,
    stake: float,
    exposure_in_match: float,
    daily_loss: float,
    balls_remaining: int,
    staleness_seconds: float | None,
    limits: RiskLimits = RiskLimits(),
) -> None:
    """Raise RiskRejection if any guardrail trips. Silence = approval."""
    if not (limits.min_prob <= model_p <= limits.max_prob):
        raise RiskRejection(f"model_p {model_p:.4f} outside [{limits.min_prob}, {limits.max_prob}]")
    if stake > limits.max_stake:
        raise RiskRejection(f"stake {stake} > max {limits.max_stake}")
    if exposure_in_match + stake > limits.max_exposure_per_match:
        raise RiskRejection("would exceed per-match exposure")
    if daily_loss >= limits.max_daily_loss:
        raise RiskRejection("daily loss limit hit")
    if balls_remaining < limits.min_balls_remaining:
        raise RiskRejection(f"only {balls_remaining} balls remaining; too volatile")
    if staleness_seconds is None or staleness_seconds > limits.max_staleness_seconds:
        raise RiskRejection(f"data stale: {staleness_seconds}s")
