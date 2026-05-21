"""Diagnostics for the inn2 phase models: signal direction, model_p
distribution, and per-match drill-down. Also re-scores the GT v CSK PP
market that was skipped live (target 230, line 60.5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.live.signals import load_models


def main() -> int:
    df = pd.read_parquet("data/processed/eval_phase_lines_inn2.parquet")
    max_train = 2023
    df = df[df["season"] > max_train].copy()
    print(f"OOS rows: {len(df)}")

    print("\n=== Signal direction balance ===")
    EDGE = 0.05
    df["signal"] = np.where(df["model_p"] >= 0.5 + EDGE, "back_over",
                  np.where(df["model_p"] <= 0.5 - EDGE, "back_under", "skip"))
    print(df.groupby(["phase", "signal"]).size().unstack(fill_value=0))

    print("\n=== model_p distribution (active trades only) ===")
    active = df[df["signal"] != "skip"]
    for phase in ["phase_6", "phase_10"]:
        sub = active[active["phase"] == phase]["model_p"]
        if sub.empty: continue
        print(f"\n  {phase} (n={len(sub)})")
        print(f"    quantiles: 5%={sub.quantile(.05):.3f}  25%={sub.quantile(.25):.3f}  "
              f"50%={sub.quantile(.50):.3f}  75%={sub.quantile(.75):.3f}  95%={sub.quantile(.95):.3f}")
        bins = [0, .05, .1, .2, .3, .4, .45, .55, .6, .7, .8, .9, .95, 1.001]
        cats = pd.cut(sub, bins=bins, right=False)
        print(cats.value_counts().sort_index().to_string())

    print("\n=== Highest-confidence trades (top 10) ===")
    top = active.assign(conf=np.abs(active["model_p"] - 0.5)).sort_values("conf", ascending=False).head(10)
    cols = ["match_id", "season", "league", "phase", "batting_team", "bowling_team",
            "target", "line_t_minus_1", "actual_total", "model_p", "signal"]
    print(top[cols].to_string(index=False))

    print("\n=== Lowest-confidence trades (just over edge) ===")
    lo = active.assign(conf=np.abs(active["model_p"] - 0.5)).sort_values("conf").head(10)
    print(lo[cols].to_string(index=False))

    print("\n=== Win-rate by model_p bucket ===")
    won = np.where(active["signal"] == "back_over",
                   (active["actual_total"] > active["line_t_minus_1"]).astype(int),
                   (active["actual_total"] < active["line_t_minus_1"]).astype(int))
    active = active.assign(won=won, p_bucket=pd.cut(active["model_p"], bins=[0, .2, .35, .5, .65, .8, 1.0]))
    print(active.groupby(["phase", "p_bucket"], observed=True)["won"].agg(["mean", "size"]).to_string())

    print("\n=== Confidence-band backtest (only take trades with |p-0.5| in band) ===")
    LINE_ODDS = 2.00
    for lo_edge, hi_edge in [(0.05, 0.20), (0.05, 0.30), (0.05, 0.40), (0.10, 0.30), (0.05, 1.0)]:
        for phase in ["phase_6", "phase_10"]:
            sub = active[active["phase"] == phase].copy()
            conf = np.abs(sub["model_p"] - 0.5)
            sub = sub[(conf >= lo_edge) & (conf <= hi_edge)]
            if sub.empty:
                print(f"  band [{lo_edge:.2f}, {hi_edge:.2f}] {phase}: 0 trades")
                continue
            won_sub = sub["won"].astype(int).to_numpy()
            pnl = np.where(won_sub == 1, 100 * (LINE_ODDS - 1) * 0.95, -100.0)
            roi = pnl.sum() / (len(sub) * 100)
            print(f"  band [{lo_edge:.2f}, {hi_edge:.2f}] {phase}: n={len(sub):3d}  ROI={roi:+.2%}  win={won_sub.mean():.1%}")

    print("\n=== Are wins concentrated in low-actual outcomes? ===")
    for phase in ["phase_6", "phase_10"]:
        sub = active[active["phase"] == phase]
        if sub.empty: continue
        print(f"  {phase}:")
        print(f"    actual_total quantiles among wins:   "
              f"25%={sub[sub['won']==1]['actual_total'].quantile(.25):.0f}  "
              f"50%={sub[sub['won']==1]['actual_total'].quantile(.50):.0f}  "
              f"75%={sub[sub['won']==1]['actual_total'].quantile(.75):.0f}")
        print(f"    actual_total quantiles among losses: "
              f"25%={sub[sub['won']==0]['actual_total'].quantile(.25):.0f}  "
              f"50%={sub[sub['won']==0]['actual_total'].quantile(.50):.0f}  "
              f"75%={sub[sub['won']==0]['actual_total'].quantile(.75):.0f}")
        chase_end = sub["actual_total"] < sub["target"] + 0  # final inn2 below target = chase didn't finish OR was bowled out
        all_out_or_blown_out = (sub["actual_total"] < 0.7 * sub["line_t_minus_1"])  # actual MUCH lower than line
        print(f"    fraction of wins where actual < 70% of line: "
              f"{(all_out_or_blown_out & (sub['won']==1)).sum() / max(1, (sub['won']==1).sum()):.1%}")

    # ----- GT v CSK 2026-05-21 lookup -----
    print("\n=== GT v CSK 2026-05-21 inn2 PP scenario (live skipped) ===")
    registry = load_models(Path("models"))
    m6 = registry["6_inn2"]
    p = m6.predict_p(
        threshold_X=61, implied_open=0.5,
        batting_team="Chennai Super Kings", bowling_team="Gujarat Titans",
        venue="Narendra Modi Stadium, Ahmedabad", season=2026, innings=2,
        league="ipl", target=230.0,
    )
    print(f"  PP under 60.5, target 230, CSK chasing GT at Ahmedabad:")
    print(f"  inn2 model_p (over) = {p:.3f}  -> signal {'BACK_OVER' if p > 0.55 else ('BACK_UNDER' if p < 0.45 else 'SKIP')}")
    print(f"  Old inn1 model said model_p=0.833 (under)  -- which we now know was wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
