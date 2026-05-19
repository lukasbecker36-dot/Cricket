"""Does the wicket-in-last-over effect survive sophisticated conditioning?

If a multi-feature model that knows the bowler, the score state, the phase,
and the historical bowler economy by phase STILL benefits substantially
from adding wkts_prev_over, then sophisticated modelers also need to
include this indicator -- meaning markets that don't are leaving edge on
the table.

If wkts_prev_over's marginal value collapses once we add proper features,
the effect is mechanically captured by anyone running a competent model
and no edge remains.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.features.player_quality import compute_player_stats, phase_shrunk_economy
from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def over_phase(over_label: int) -> int:
    if over_label < 6:
        return 0
    if over_label < 15:
        return 1
    return 2


def build_table(balls: pd.DataFrame, stats_cache: dict) -> pd.DataFrame:
    """Per-over rows with state at start of over plus the bowler's phase econ."""
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
            bowler=("bowler", "first"),
        ).reset_index()
        per_over = per_over.iloc[:-1] if len(legal_only) % 6 != 0 else per_over
        if len(per_over) < 2:
            continue
        per_over["wkts_prev_over"] = per_over["wickets"].shift(1).fillna(0).astype(int)
        per_over["runs_so_far"] = per_over["runs"].cumsum().shift(1).fillna(0).astype(int)
        per_over["wkts_so_far"] = per_over["wickets"].cumsum().shift(1).fillna(0).astype(int)
        per_over["balls_so_far"] = (per_over.index * 6).astype(int)
        per_over["phase"] = per_over["over_label"].apply(over_phase)
        per_over["innings"] = innings
        per_over["match_id"] = match_id
        rows.append(per_over)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Bowler phase economy from leak-free per-season cache
    out["bowler_phase_econ"] = [
        phase_shrunk_economy(stats_cache[s], b, p)
        for s, b, p in zip(out["season"], out["bowler"], out["phase"])
    ]
    return out


def fit_and_eval(df: pd.DataFrame, feature_cols: list[str], label: str) -> tuple[lgb.Booster, float]:
    train_idx = df["season"] <= df["season"].quantile(0.85)
    train = df[train_idx]
    test = df[~train_idx]
    x_train = train[feature_cols].to_numpy(dtype=np.float32)
    y_train = train["runs"].to_numpy(dtype=np.float32)
    x_test = test[feature_cols].to_numpy(dtype=np.float32)
    y_test = test["runs"].to_numpy(dtype=np.float32)
    booster = lgb.train(
        params={
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 200,
            "verbose": -1,
            "seed": 17,
            "deterministic": True,
        },
        train_set=lgb.Dataset(x_train, label=y_train, feature_name=feature_cols),
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50, verbose=False)],
        valid_sets=[lgb.Dataset(x_test, label=y_test, feature_name=feature_cols)],
    )
    pred = booster.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    logger.info("%s RMSE=%.4f (n_train=%d, n_test=%d)", label, rmse, len(train), len(test))
    return booster, rmse


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
    logger.info("loaded %d balls", len(balls))

    seasons = sorted(balls["season"].unique())
    stats_cache = {s: compute_player_stats(balls, up_to_season=s) for s in seasons}
    logger.info("computed per-season stats for %d seasons", len(seasons))

    table = build_table(balls, stats_cache)
    mask = (table["over_label"] >= args.phase_min_over) & (table["over_label"] <= args.phase_max_over)
    df = table[mask].dropna().reset_index(drop=True)
    logger.info("middle-overs rows: %d", len(df))

    base_features = [
        "runs_so_far", "wkts_so_far", "balls_so_far",
        "over_label", "phase", "innings", "bowler_phase_econ",
    ]
    full_features = base_features + ["wkts_prev_over"]

    print("\n=== Conditional-on-rich-features check ===")
    print(f"Base features: {base_features}")
    print(f"With added 'wkts_prev_over' indicator: {full_features}")
    print()

    _, rmse_base = fit_and_eval(df, base_features, "no-wkt-feature ")
    booster_full, rmse_full = fit_and_eval(df, full_features, "with-wkt-feature")

    print(f"\nRMSE base (no wkts_prev_over):   {rmse_base:.4f}")
    print(f"RMSE with wkts_prev_over added:  {rmse_full:.4f}")
    print(f"Marginal RMSE improvement:       {rmse_base - rmse_full:+.4f}  ({(rmse_base-rmse_full)/rmse_base*100:+.2f}%)")

    # Feature importance
    print(f"\nFeature importance (gain) with wkts_prev_over included:")
    imps = sorted(zip(full_features, booster_full.feature_importance(importance_type="gain")), key=lambda x: -x[1])
    total = sum(g for _, g in imps)
    for name, gain in imps:
        print(f"  {name:25s} gain={gain:10.0f}  ({gain/total*100:5.1f}%)")

    # Residual analysis: take the no-wkt-feature model's predictions on test,
    # then check whether residuals are systematically different for the two groups
    train_idx = df["season"] <= df["season"].quantile(0.85)
    test = df[~train_idx].reset_index(drop=True)
    booster_base, _ = fit_and_eval(df, base_features, "(re-fit for residuals)")
    test_pred = booster_base.predict(test[base_features].to_numpy(dtype=np.float32))
    residuals = test["runs"].to_numpy(dtype=float) - test_pred
    mask_no = (test["wkts_prev_over"] == 0).to_numpy()
    mask_yes = (test["wkts_prev_over"] >= 1).to_numpy()
    print(f"\nResiduals from BASE model (no wkts_prev_over feature) on test set:")
    print(f"  after no wicket: mean residual = {residuals[mask_no].mean():+.3f}  (n={mask_no.sum()})")
    print(f"  after a wicket:  mean residual = {residuals[mask_yes].mean():+.3f}  (n={mask_yes.sum()})")
    print(f"  systematic gap:  {residuals[mask_no].mean() - residuals[mask_yes].mean():+.3f} runs/over")
    print()
    print("If the systematic gap above is close to zero, the rich features absorb the")
    print("wicket-clustering effect. If it stays at ~1 run, the simple wicket indicator")
    print("carries information the rich features miss => independent edge survives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
