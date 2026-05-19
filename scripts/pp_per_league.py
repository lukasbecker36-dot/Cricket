"""Per-league replication of PP threshold mispricing.

Reads the three-snapshot parquet, joins each match to its league via the
Cricsheet match metas, then re-runs the calibration analysis per league.
If the +/-pp pattern at the tails is similar across leagues, the signal
is real. If it's driven by one league, it's specific.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    configure_logging()
    df = pd.read_parquet("data/processed/pp_three_snapshots.parquet")
    logger.info("loaded %d rows", len(df))

    # Build match_id -> league mapping
    leagues = ["ipl", "bbl", "psl", "cpl", "ntb"]
    league_by_match: dict[str, str] = {}
    for league in leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        m = pd.read_parquet(d / "matches.parquet")
        for mid in m["match_id"].astype(str).unique():
            league_by_match[mid] = league
    df["league"] = df["match_id"].astype(str).map(league_by_match)
    df = df.dropna(subset=["league"])
    logger.info("matches by league: %s", df.groupby("league")["match_id"].nunique().to_dict())

    for col, label in [
        ("p_open",     "OPENING"),
        ("p_last",     "LAST OBSERVED"),
    ]:
        for league in ["ALL"] + leagues:
            sub = df if league == "ALL" else df[df["league"] == league]
            sub = sub.dropna(subset=[col]).copy()
            sub = sub[sub[col] > 1.0]
            if sub.empty:
                continue
            sub["implied"] = 1.0 / sub[col]
            sub["bin"] = pd.cut(sub["implied"], bins=np.arange(0, 1.05, 0.1), include_lowest=True)
            cal = sub.groupby("bin").agg(
                n=("implied", "size"),
                mean_implied=("implied", "mean"),
                actual_rate=("actual_over_X", "mean"),
            ).dropna()
            cal["edge_pp"] = (cal["actual_rate"] - cal["mean_implied"]) * 100
            print(f"\n=== {label} -- {league.upper()} (n={len(sub):,}) ===")
            print(cal.round(4).to_string())

    # Compact summary: edge_pp at the two extreme bins by league
    print("\n=== Compact: edge_pp at extreme bins (LAST OBSERVED) ===")
    sub = df.dropna(subset=["p_last"]).copy()
    sub = sub[sub["p_last"] > 1.0]
    sub["implied"] = 1.0 / sub["p_last"]
    out = []
    for league in leagues:
        league_sub = sub[sub["league"] == league]
        if league_sub.empty:
            continue
        low = league_sub[league_sub["implied"] < 0.1]
        mid = league_sub[(league_sub["implied"] >= 0.4) & (league_sub["implied"] < 0.5)]
        high = league_sub[league_sub["implied"] >= 0.9]
        out.append({
            "league": league.upper(),
            "n_low_tail":  len(low),  "edge_low_tail":  round((low["actual_over_X"].mean() - low["implied"].mean())*100, 1) if len(low) else None,
            "n_mid":       len(mid),  "edge_mid":       round((mid["actual_over_X"].mean() - mid["implied"].mean())*100, 1) if len(mid) else None,
            "n_high_tail": len(high), "edge_high_tail": round((high["actual_over_X"].mean() - high["implied"].mean())*100, 1) if len(high) else None,
        })
    print(pd.DataFrame(out).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
