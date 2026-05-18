"""Data quality checks. Fail loudly: bad matches are excluded, not silently passed through."""
from __future__ import annotations

import logging
from collections.abc import Iterable

from .schema import Ball, MatchMeta

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    pass


def validate_match(meta: MatchMeta, balls: list[Ball]) -> None:
    """Raise ValidationError if the match fails any data quality check."""
    if not balls:
        raise ValidationError(f"{meta.match_id}: no balls")

    for innings in (1, 2):
        legal = [b for b in balls if b.innings == innings and b.is_legal_delivery]
        if not legal:
            # Some innings end before any legal delivery only in pathological cases.
            continue
        n = len(legal)
        if innings == 1 and not (n <= 130):
            raise ValidationError(f"{meta.match_id}: innings 1 has {n} legal balls (>130)")
        if innings == 2 and not (n <= 130):
            raise ValidationError(f"{meta.match_id}: innings 2 has {n} legal balls (>130)")

    ids = [(b.match_id, b.innings, b.over, b.ball, b.is_legal_delivery) for b in balls]
    seen: set[tuple] = set()
    for key in ids:
        if not key[-1]:  # extras can repeat ball=0 within an over; skip
            continue
        if key in seen:
            raise ValidationError(f"{meta.match_id}: duplicate ball id {key}")
        seen.add(key)

    for b in balls:
        if b.wicket and b.dismissal_kind is None:
            raise ValidationError(f"{meta.match_id}: wicket without dismissal_kind")

    for innings in (1, 2):
        inn_balls = [b for b in balls if b.innings == innings]
        if not inn_balls:
            continue
        total = sum(b.runs_total for b in inn_balls)
        if total < 0 or total > 350:
            raise ValidationError(f"{meta.match_id}: innings {innings} total {total} out of range")


def filter_valid(
    matches: Iterable[tuple[MatchMeta, list[Ball]]],
) -> Iterable[tuple[MatchMeta, list[Ball]]]:
    """Yield only matches that pass validation; log and skip the rest."""
    for meta, balls in matches:
        try:
            validate_match(meta, balls)
        except ValidationError as e:
            logger.warning("excluding match: %s", e)
            continue
        yield meta, balls
