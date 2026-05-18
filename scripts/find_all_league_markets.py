"""Scan a Betfair BASIC tar archive, bucket markets by T20 league.

Outputs one index file per league found: data/processed/betfair_<league>_index.json.
Each entry: {archive_path, event_id, market_id, market_time, event_name, runners}.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.ingestion.league_rosters import LEAGUE_TEAMS, detect_league
from src.logging_setup import configure_logging
from scripts.find_ipl_markets import first_market_definition
import tarfile

logger = logging.getLogger(__name__)


def scan(tar_path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {league: [] for league in LEAGUE_TEAMS}
    n_total = 0
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".bz2"):
                continue
            n_total += 1
            f = tar.extractfile(member)
            if f is None:
                continue
            md = first_market_definition(f.read())
            if md is None:
                continue
            runners = md.get("runners", []) or []
            runner_names = [r.get("name", "") for r in runners]
            league = detect_league(runner_names)
            if league is None:
                continue
            if md.get("marketType") != "MATCH_ODDS":
                continue
            out[league].append({
                "archive_path": member.name,
                "event_id": str(md.get("eventId", "")),
                "market_id": member.name.rsplit("/", 1)[-1].replace(".bz2", ""),
                "market_type": md.get("marketType"),
                "market_time": md.get("marketTime"),
                "event_name": md.get("eventName"),
                "runners": [(r.get("id"), r.get("name")) for r in runners],
            })
    logger.info("scanned %d markets total", n_total)
    return out


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("data/processed"), type=Path)
    args = parser.parse_args()

    by_league = scan(args.archive)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for league, matches in by_league.items():
        if not matches:
            continue
        path = args.out_dir / f"betfair_{league}_index.json"
        path.write_text(json.dumps(matches, indent=2))
        seasons: dict[str, int] = {}
        for m in matches:
            yr = (m.get("market_time") or "")[:4]
            if yr:
                seasons[yr] = seasons.get(yr, 0) + 1
        breakdown = " ".join(f"{y}:{seasons[y]}" for y in sorted(seasons))
        logger.info("%-6s %4d markets   %s", league.upper(), len(matches), breakdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
