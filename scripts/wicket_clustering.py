"""Wicket clustering: does a wicket falling change the next over's distribution?

Hypothesis (Markovian naive vs reality):
  P(wicket in over | wicket in last over) > P(wicket in over) overall
  E[runs in over | wicket in last over] < E[runs in over] overall

If markets price 'runs in next over' or similar event markets using the
unconditional distribution, then a model that conditions on recent wicket
events finds systematic mispricing.

This script reports the conditional distributions across all leagues' inn-1
and inn-2 data, then quantifies the magnitude of the effect.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def build_per_over_table(balls: pd.DataFrame) -> pd.DataFrame:
    """One row per (match_id, innings, over_index). Records the over's runs
    and wickets, plus whether wickets fell in the recent past."""
    rows = []
    for (match_id, innings), group in balls.groupby(["match_id", "innings"]):
        group = group.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False])
        legal_only = group[group["is_legal_delivery"]].reset_index(drop=True)
        if len(legal_only) < 12:
            continue
        legal_only["over_idx"] = legal_only.index // 6
        per_over = legal_only.groupby("over_idx").agg(
            runs=("runs_total", "sum"),
            wickets=("wicket", "sum"),
            over_label=("over", "first"),
            season=("season", "first"),
        ).reset_index()
        per_over = per_over.iloc[:-1] if len(legal_only) % 6 != 0 else per_over
        if len(per_over) < 2:
            continue
        per_over["wkts_prev_over"] = per_over["wickets"].shift(1).fillna(0).astype(int)
        per_over["wkts_prev2_overs"] = (
            per_over["wickets"].shift(1).fillna(0) + per_over["wickets"].shift(2).fillna(0)
        ).astype(int)
        per_over["match_id"] = match_id
        per_over["innings"] = innings
        rows.append(per_over)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def conditional_stats(label: str, mask: pd.Series, df: pd.DataFrame) -> dict:
    sub = df[mask]
    if sub.empty:
        return {"condition": label, "n": 0}
    return {
        "condition": label,
        "n": int(len(sub)),
        "mean_runs": round(float(sub["runs"].mean()), 3),
        "std_runs": round(float(sub["runs"].std()), 3),
        "p_wkt_in_over": round(float((sub["wickets"] > 0).mean()), 4),
        "mean_wickets": round(float(sub["wickets"].mean()), 4),
    }


def welch_p(a: np.ndarray, b: np.ndarray) -> float:
    res = scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(res.pvalue)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="+", default=["ipl", "bbl", "psl", "cpl", "ntb"])
    parser.add_argument("--phase-min-over", type=int, default=3)
    parser.add_argument("--phase-max-over", type=int, default=15)
    args = parser.parse_args()

    all_balls = []
    for league in args.leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
    balls = pd.concat(all_balls, ignore_index=True)
    logger.info("loaded %d balls across %d leagues", len(balls), len(all_balls))

    table = build_per_over_table(balls)
    logger.info("per-over rows: %d", len(table))

    phase_mask = (table["over_label"] >= args.phase_min_over) & (table["over_label"] <= args.phase_max_over)
    df = table[phase_mask].reset_index(drop=True)
    logger.info("middle-overs rows (over %d-%d): %d",
                args.phase_min_over, args.phase_max_over, len(df))

    base_runs = df["runs"].mean()
    base_pwkt = (df["wickets"] > 0).mean()
    print(f"\n=== Unconditional baseline (middle overs only, n={len(df):,}) ===")
    print(f"  mean runs/over     : {base_runs:.3f}")
    print(f"  P(wicket in over)  : {base_pwkt:.4f}")
    print(f"  mean wickets/over  : {df['wickets'].mean():.4f}")

    conds = [
        conditional_stats("no wkt last over",  df["wkts_prev_over"] == 0, df),
        conditional_stats("1+ wkt last over",  df["wkts_prev_over"] >= 1, df),
        conditional_stats("2+ wkts last over", df["wkts_prev_over"] >= 2, df),
        conditional_stats("no wkt last 2 overs",  df["wkts_prev2_overs"] == 0, df),
        conditional_stats("1+ wkt last 2 overs",  df["wkts_prev2_overs"] >= 1, df),
        conditional_stats("2+ wkts last 2 overs", df["wkts_prev2_overs"] >= 2, df),
    ]
    print("\n=== Conditional on recent wickets ===")
    print(pd.DataFrame(conds).to_string(index=False))

    print("\n=== Significance (Welch t-test on runs/over) ===")
    a = df[df["wkts_prev_over"] == 0]["runs"].to_numpy(dtype=float)
    b = df[df["wkts_prev_over"] >= 1]["runs"].to_numpy(dtype=float)
    p_runs = welch_p(a, b)
    diff_runs = b.mean() - a.mean()
    print(f"  runs/over: with-wkt={b.mean():.2f}  no-wkt={a.mean():.2f}  diff={diff_runs:+.2f}  p={p_runs:.2e}")

    a_wkt = df[df["wkts_prev_over"] == 0]["wickets"].to_numpy(dtype=float)
    b_wkt = df[df["wkts_prev_over"] >= 1]["wickets"].to_numpy(dtype=float)
    p_wkt = welch_p(a_wkt, b_wkt)
    diff_wkt = b_wkt.mean() - a_wkt.mean()
    print(f"  wickets/over: with-wkt={b_wkt.mean():.4f}  no-wkt={a_wkt.mean():.4f}  diff={diff_wkt:+.4f}  p={p_wkt:.2e}")

    print("\n=== Translation to 'runs in next over' over/under markets ===")
    print(f"If the market priced 'runs in next over' at the unconditional mean ({base_runs:.2f}),")
    print(f"  - after a wicket, true expected runs = {b.mean():.2f}  (mkt over-bias of {base_runs - b.mean():+.2f})")
    print(f"  - after no wicket, true expected runs = {a.mean():.2f}  (mkt under-bias of {a.mean() - base_runs:+.2f})")
    print(f"Edge magnitude: {abs(diff_runs):.2f} runs / over = about {abs(diff_runs)/base_runs*100:.1f}% of the mean.")

    print("\n=== Per-league replication ===")
    for league in args.leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if b.empty:
            continue
        t = build_per_over_table(b)
        t = t[(t["over_label"] >= args.phase_min_over) & (t["over_label"] <= args.phase_max_over)]
        if t.empty:
            continue
        with_w = t[t["wkts_prev_over"] >= 1]["runs"].mean()
        no_w = t[t["wkts_prev_over"] == 0]["runs"].mean()
        with_w_wkt = t[t["wkts_prev_over"] >= 1]["wickets"].mean()
        no_w_wkt = t[t["wkts_prev_over"] == 0]["wickets"].mean()
        print(f"  {league.upper():5s} n={len(t):6d}  runs: no-wkt={no_w:.2f}  with-wkt={with_w:.2f}  diff={with_w - no_w:+.2f}   "
              f"wkts: {no_w_wkt:.3f} -> {with_w_wkt:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
