"""Investigate why phase_6 / phase_10 edge decayed in 2025.

Hypotheses tested:
  H1. Market got sharper -- the line sits closer to the actual outcome over time
  H2. Lines drifted to fair -- actual over-rate converged to 50%
  H3. Model directional accuracy degraded by year
  H4. League mix shifted (e.g. more noisy NTB in 2025)
  H5. Scoring distribution shifted (priors lag the modern game)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.storage import read_balls
from src.live.signals import load_models


def main() -> int:
    eval_df = pd.read_parquet("data/processed/eval_phase_lines.parquet")
    registry = load_models(Path("models"))

    # Recompute model_p with current (no-weather) models for consistency
    def score(row):
        m = registry.get(row["phase"].replace("phase_", ""))
        if m is None:
            return np.nan
        return m.predict_p(
            threshold_X=int(round(row["line_t_minus_1"])), implied_open=0.5,
            batting_team=row["batting_team"], bowling_team=row["bowling_team"],
            venue=row["venue"], season=int(row["season"]), innings=int(row["innings"]),
            league=row["league"])

    eval_df["model_p"] = eval_df.apply(score, axis=1)
    eval_df = eval_df.dropna(subset=["model_p"])
    eval_df["line_minus_actual"] = eval_df["line_t_minus_1"] - eval_df["actual_total"]
    eval_df["abs_line_error"] = eval_df["line_minus_actual"].abs()
    eval_df["actual_over"] = (eval_df["actual_total"] > eval_df["line_t_minus_1"]).astype(int)
    # model directional correctness: did the side the model favoured win?
    eval_df["model_side_over"] = (eval_df["model_p"] >= 0.5).astype(int)
    eval_df["model_correct"] = (eval_df["model_side_over"] == eval_df["actual_over"]).astype(int)
    # mid-band flag
    eval_df["is_mid"] = (eval_df["model_p"] - 0.5).abs().between(0.05, 0.30)

    for phase in ["phase_6", "phase_10", "phase_15"]:
        sub = eval_df[eval_df["phase"] == phase]
        print(f"\n{'='*70}\n{phase.upper()}\n{'='*70}")
        print("\nH1/H2 — market sharpness by year (lower abs_line_error = sharper):")
        g = sub.groupby("season").agg(
            n=("actual_total", "size"),
            abs_line_err=("abs_line_error", "mean"),
            line_bias=("line_minus_actual", "mean"),   # +ve = line set above actual (market too high)
            actual_over_rate=("actual_over", "mean"),  # should be ~0.5 if line fair
        ).round(2)
        print(g.to_string())

        print("\nH3 — model directional accuracy by year (all signals & mid-band):")
        g2 = sub.groupby("season").agg(
            n=("model_correct", "size"),
            model_acc_all=("model_correct", "mean"),
        ).round(3)
        mid = sub[sub["is_mid"]]
        g2_mid = mid.groupby("season").agg(
            n_mid=("model_correct", "size"),
            model_acc_mid=("model_correct", "mean"),
        ).round(3)
        print(g2.join(g2_mid, how="left").to_string())

        print("\nH4 — league mix by year (row counts):")
        print(sub.groupby(["season", "league"]).size().unstack(fill_value=0).to_string())

    # H5 — scoring distribution shift (use raw balls, innings-1 phase totals)
    print(f"\n{'='*70}\nH5 — innings-1 phase total distribution by season (all leagues)\n{'='*70}")
    LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]
    all_balls = []
    for league in LEAGUES:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
    balls = pd.concat(all_balls, ignore_index=True)

    def phase_total(g, tb):
        g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
        li = np.where(g["is_legal_delivery"].values)[0]
        if len(li) < tb:
            return None
        return int(g.iloc[:li[tb-1]+1]["runs_total"].sum())

    for phase, tb in [("phase_6", 36), ("phase_10", 60), ("phase_15", 90)]:
        rows = []
        for (mid, inn), g in balls.groupby(["match_id", "innings"]):
            if int(inn) != 1:
                continue
            t = phase_total(g, tb)
            if t is not None:
                rows.append({"season": int(g["season"].iloc[0]), "total": t})
        pt = pd.DataFrame(rows)
        recent = pt[pt["season"] >= 2020]
        print(f"\n{phase} mean phase total by season (2020+):")
        print(recent.groupby("season")["total"].agg(["mean", "std", "size"]).round(1).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
