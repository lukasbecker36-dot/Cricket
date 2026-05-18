"""Parquet writer: ball-by-ball data partitioned by season."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .schema import Ball, MatchMeta

logger = logging.getLogger(__name__)


def write_balls(balls: list[Ball], out_dir: Path) -> None:
    if not balls:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([b.model_dump() for b in balls])
    for season, group in df.groupby("season"):
        path = out_dir / f"season={season}" / "balls.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, group], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["match_id", "innings", "over", "ball", "is_legal_delivery"],
                keep="last",
            )
            combined.to_parquet(path, index=False)
        else:
            group.to_parquet(path, index=False)
    logger.info("wrote %d balls across %d seasons to %s", len(df), df["season"].nunique(), out_dir)


def write_meta(metas: list[MatchMeta], out_dir: Path) -> None:
    if not metas:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([m.model_dump() for m in metas])
    path = out_dir / "matches.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates("match_id", keep="last")
    df.to_parquet(path, index=False)
    logger.info("wrote %d match metas to %s", len(df), path)


def read_balls(processed_dir: Path, seasons: list[int] | None = None) -> pd.DataFrame:
    """Read ball-by-ball data, optionally filtered to specific seasons."""
    if seasons is None:
        paths = sorted(processed_dir.glob("season=*/balls.parquet"))
    else:
        paths = [processed_dir / f"season={s}" / "balls.parquet" for s in seasons]
        paths = [p for p in paths if p.exists()]
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
