"""Build a Betfair-IPL parquet store joined to Cricsheet match IDs.

Reads:
  data/raw/betfair/<archive>.dat          (tar of BASIC .bz2 files)
  data/processed/betfair_ipl_index.json   (output of find_ipl_markets)
  data/processed/matches.parquet          (Cricsheet match metas)

Writes:
  data/processed/betfair/<season>/<match_id>.parquet
  data/processed/betfair_join.parquet     (which Betfair markets matched to which Cricsheet match)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.ingestion.betfair_loader import (
    implied_market_probability,
    load_from_tar,
    match_betfair_to_cricsheet,
)
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)


# Cricsheet uses different team-name conventions in different seasons; map
# Betfair's canonical 2024+ names to whatever Cricsheet expects per season.
# We map both directions in the joiner so just normalising the Betfair side is enough.
TEAM_ALIASES = {
    # Betfair (2024+ form) -> commonly Cricsheet form
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
}


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=Path("data/raw/betfair/betfair_raw.dat"), type=Path)
    parser.add_argument("--league", default="ipl", help="league code: ipl, bbl, psl, cpl, ntb, ...")
    parser.add_argument("--index", default=None, type=Path)
    parser.add_argument("--matches", default=None, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    # Default paths derived from league
    league = args.league
    cricsheet_dir = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
    if args.index is None:
        args.index = Path(f"data/processed/betfair_{league}_index.json")
    if args.matches is None:
        args.matches = cricsheet_dir / "matches.parquet"
    if args.out is None:
        args.out = cricsheet_dir / "betfair"

    index = json.loads(args.index.read_text())
    cricsheet = pd.read_parquet(args.matches)
    logger.info("loaded %d Betfair markets and %d Cricsheet matches", len(index), len(cricsheet))

    joined = match_betfair_to_cricsheet(index, cricsheet, TEAM_ALIASES)
    args.out.mkdir(parents=True, exist_ok=True)
    join_path = cricsheet_dir / "betfair_join.parquet"
    joined.to_parquet(join_path, index=False)
    logger.info("matched %d / %d Betfair markets to Cricsheet match_ids", len(joined), len(index))

    # Now extract each matched market into a per-match parquet of implied probabilities.
    cs_by_id = cricsheet.set_index("match_id")
    written = 0
    for row in joined.itertuples(index=False):
        market = load_from_tar(args.archive, row.archive_path)
        if market is None:
            continue
        # Identify which Betfair selection_id is the chasing team.
        # We don't yet know who chases until we look at Cricsheet's innings 2 batting team.
        # The matches.parquet currently has team list but not innings ordering -- we look at
        # balls.parquet for that. Simpler: assume both teams could be the chaser and store
        # ticks for both; downstream can pick.
        season = cs_by_id.loc[row.cricsheet_match_id]["season"]
        out_dir = args.out / f"season={season}"
        out_dir.mkdir(parents=True, exist_ok=True)
        market.ticks.to_parquet(out_dir / f"{row.cricsheet_match_id}_ticks.parquet", index=False)
        # Stash the runner-id-to-team map alongside
        runners_path = out_dir / f"{row.cricsheet_match_id}_runners.json"
        runners_path.write_text(json.dumps({str(k): v for k, v in market.runners.items()}))
        written += 1
        if written % 50 == 0:
            logger.info("processed %d markets", written)

    logger.info("wrote ticks for %d matches under %s", written, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
