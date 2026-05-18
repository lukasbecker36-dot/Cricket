"""Download Cricsheet IPL data and write the canonical parquet store.

Usage:
    python -m scripts.ingest [--config configs/default.yaml] [--force-download]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import Config
from src.ingestion.cricsheet import download_zip, iter_match_jsons, parse_match
from src.ingestion.storage import write_balls, write_meta
from src.ingestion.validate import filter_valid
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    configure_logging()
    cfg = Config.load(args.config)

    zip_path = cfg.data.cache_dir / "ipl_json.zip"
    download_zip(cfg.data.cricsheet_url, zip_path, force=args.force_download)

    metas: list = []
    balls_buffer: list = []
    parsed = (parse_match(mid, raw) for mid, raw in iter_match_jsons(zip_path))

    for meta, balls in filter_valid(parsed):
        metas.append(meta)
        balls_buffer.extend(balls)
        if len(balls_buffer) >= 200_000:
            write_balls(balls_buffer, cfg.data.processed_dir)
            balls_buffer.clear()

    if balls_buffer:
        write_balls(balls_buffer, cfg.data.processed_dir)
    write_meta(metas, cfg.data.processed_dir)
    logger.info("ingest complete: %d matches", len(metas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
