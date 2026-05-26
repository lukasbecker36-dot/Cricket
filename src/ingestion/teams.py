"""Franchise name canonicalisation.

T20 franchises rebrand (Kings XI Punjab -> Punjab Kings) or get spelled
differently across seasons (Bangalore/Bengaluru). Cricsheet records whatever
name was used at the time, so a franchise's history ends up fragmented across
keys. That breaks per-(team, season) prior lookups: the current-name event
misses its own history and falls through to default_par.

Canonicalise every variant to the franchise's CURRENT name, applied both when
building priors and at inference, so all of a franchise's history accrues to
one key.

We do NOT merge genuinely distinct franchises (e.g. Deccan Chargers was
terminated and Sunrisers Hyderabad is a different ownership; Gujarat Lions is
not Gujarat Titans). Only true rebrands / spelling variants are merged.
"""
from __future__ import annotations

_CANONICAL: dict[str, str] = {
    # Royal Challengers Bangalore -> Bengaluru (rebrand 2024)
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    # Kings XI Punjab -> Punjab Kings (rebrand 2021)
    "Kings XI Punjab": "Punjab Kings",
    # Delhi Daredevils -> Delhi Capitals (rebrand 2019)
    "Delhi Daredevils": "Delhi Capitals",
    # spelling variant
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}


def canonical_team(name: str | None) -> str:
    if not name:
        return ""
    return _CANONICAL.get(name, name)
