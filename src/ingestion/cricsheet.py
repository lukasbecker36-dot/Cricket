"""Download IPL matches from Cricsheet and normalise to the canonical schema.

Cricsheet ships a single zip of per-match JSON files. We download it once, cache
the zip, then iterate the entries lazily so we never blow memory.
"""
from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

from .schema import Ball, MatchMeta

logger = logging.getLogger(__name__)


def download_zip(url: str, dest: Path, force: bool = False) -> Path:
    """Download the Cricsheet zip if missing. Returns the local path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        logger.info("cricsheet zip already cached at %s", dest)
        return dest

    logger.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return dest


def iter_match_jsons(zip_path: Path) -> Iterator[tuple[str, dict]]:
    """Yield (match_id, parsed_json) for every match JSON in the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            match_id = Path(name).stem
            with zf.open(name) as f:
                yield match_id, json.load(f)


def _season_from_info(info: dict) -> int:
    season = info.get("season")
    if isinstance(season, int):
        return season
    if isinstance(season, str):
        # Cricsheet uses "2007/08" for cross-year leagues; IPL is single-year but be defensive.
        head = season.split("/")[0]
        try:
            return int(head)
        except ValueError:
            pass
    dates = info.get("dates", [])
    if dates:
        try:
            return datetime.fromisoformat(str(dates[0])).year
        except ValueError:
            return int(str(dates[0])[:4])
    raise ValueError("cannot determine season")


def parse_match(match_id: str, raw: dict) -> tuple[MatchMeta, list[Ball]]:
    """Convert a Cricsheet match JSON into MatchMeta + list[Ball].

    Cricsheet schema: top-level keys `info` and `innings`. Each innings has `overs`,
    each over has `deliveries` with `runs.batter`, `runs.extras`, optional `wickets`,
    and `extras` dict whose keys identify the extras kind.
    """
    info = raw["info"]
    season = _season_from_info(info)
    venue = info.get("venue", "unknown")
    teams = list(info.get("teams", []))
    outcome = info.get("outcome", {}) or {}
    winner = outcome.get("winner")
    result = "no result" if "result" in outcome else ("tie" if "eliminator" in outcome else "normal")

    meta = MatchMeta(
        match_id=match_id,
        season=season,
        date=str(info.get("dates", [""])[0]),
        venue=venue,
        teams=teams,
        toss_winner=info.get("toss", {}).get("winner", ""),
        toss_decision=info.get("toss", {}).get("decision", ""),
        winner=winner,
        result=result,
    )

    innings_runs: list[int] = []
    balls: list[Ball] = []
    innings_blocks = raw.get("innings", []) or []

    for idx, inn in enumerate(innings_blocks, start=1):
        batting_team = inn.get("team", "")
        bowling_team = next((t for t in teams if t != batting_team), "")
        inn_runs = 0
        for over_block in inn.get("overs", []):
            over_idx = int(over_block["over"])
            legal_in_over = 0
            for delivery in over_block.get("deliveries", []):
                runs_block = delivery.get("runs", {})
                runs_batter = int(runs_block.get("batter", 0))
                runs_extras = int(runs_block.get("extras", 0))
                runs_total = int(runs_block.get("total", runs_batter + runs_extras))
                inn_runs += runs_total

                extras = delivery.get("extras", {}) or {}
                extras_kind = next(iter(extras.keys()), None)
                is_legal = extras_kind not in {"wides", "noballs"}
                if is_legal:
                    legal_in_over += 1

                wickets = delivery.get("wickets") or []
                wicket = bool(wickets)
                dismissal_kind = wickets[0].get("kind") if wicket else None
                player_out = wickets[0].get("player_out") if wicket else None

                target = None
                if idx == 2 and innings_runs:
                    target = innings_runs[0] + 1

                balls.append(
                    Ball(
                        match_id=match_id,
                        season=season,
                        venue=venue,
                        innings=idx,
                        over=over_idx,
                        ball=legal_in_over if is_legal else 0,
                        batting_team=batting_team,
                        bowling_team=bowling_team,
                        striker=delivery.get("batter", ""),
                        non_striker=delivery.get("non_striker", ""),
                        bowler=delivery.get("bowler", ""),
                        runs_batter=runs_batter,
                        runs_extras=runs_extras,
                        runs_total=runs_total,
                        extras_kind=extras_kind,
                        wicket=wicket,
                        dismissal_kind=dismissal_kind,
                        player_out=player_out,
                        target=target,
                        is_legal_delivery=is_legal,
                    )
                )
        innings_runs.append(inn_runs)

    if len(innings_runs) >= 1:
        meta.target_innings_2 = innings_runs[0] + 1

    return meta, balls
