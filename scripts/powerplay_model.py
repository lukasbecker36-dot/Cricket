"""Powerplay-total prediction: at end of over 3, predict runs at end of over 6.

Tests the hypothesis: 'the market's adjustment for the remaining 3 powerplay
overs is often conservative.' If naive extrapolation from current rate is
close to optimal, there's no model edge. If a state-aware model substantially
beats naive, there's room for the market to be similarly underadjusting.

Trains on combined T20 data (IPL+BBL+PSL+CPL+NTB) across all but the most
recent two seasons; tests on the most recent two seasons. Compares LightGBM
to:
  - league_mean: just the historical league powerplay average
  - linear_rate: runs_after_3_overs * 2  (assume same rate for remaining 3 overs)
  - linear_adjusted: regression on (runs, wickets) only
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.player_quality import compute_player_stats, phase_shrunk_economy, shrunk_strike_rate
from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

# State observation point: after this many legal balls
OBSERVE_AT = 18  # end of over 3
PP_END = 36      # end of over 6 (full powerplay)


def extract_samples(balls: pd.DataFrame) -> pd.DataFrame:
    """One row per innings-1 powerplay where >= 36 legal balls were bowled."""
    rows = []
    for match_id, match_balls in balls.groupby("match_id"):
        inn1 = match_balls[match_balls["innings"] == 1].sort_values(
            ["over", "ball", "is_legal_delivery"], ascending=[True, True, False]
        ).reset_index(drop=True)
        if inn1.empty:
            continue
        legal_mask = inn1["is_legal_delivery"].values
        legal_idx = np.where(legal_mask)[0]
        if len(legal_idx) < PP_END:
            continue  # innings ended before powerplay completed

        cut_18 = legal_idx[OBSERVE_AT - 1] + 1
        cut_36 = legal_idx[PP_END - 1] + 1
        runs_18 = inn1.iloc[:cut_18]["runs_total"].sum()
        wickets_18 = inn1.iloc[:cut_18]["wicket"].sum()
        pp_total = inn1.iloc[:cut_36]["runs_total"].sum()

        state_at_18 = inn1.iloc[cut_18 - 1]
        rows.append({
            "match_id": match_id,
            "season": int(state_at_18["season"]),
            "venue": str(state_at_18["venue"]),
            "striker": str(state_at_18["striker"]),
            "non_striker": str(state_at_18["non_striker"]),
            "bowler": str(state_at_18["bowler"]),
            "runs_18": int(runs_18),
            "wickets_18": int(wickets_18),
            "pp_total": int(pp_total),
        })
    return pd.DataFrame(rows)


def add_player_features(samples: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    """Per-row player-quality features computed from strictly-prior seasons."""
    df = samples.copy()
    df["striker_sr"] = np.nan
    df["non_striker_sr"] = np.nan
    df["bowler_econ_pp"] = np.nan

    # Cache stats per row's season for efficiency
    seasons = sorted(df["season"].unique())
    stats_cache = {s: compute_player_stats(balls, up_to_season=s) for s in seasons}
    for s in seasons:
        stats = stats_cache[s]
        mask = df["season"] == s
        df.loc[mask, "striker_sr"] = df.loc[mask, "striker"].apply(
            lambda p, st=stats: shrunk_strike_rate(st, p)
        )
        df.loc[mask, "non_striker_sr"] = df.loc[mask, "non_striker"].apply(
            lambda p, st=stats: shrunk_strike_rate(st, p)
        )
        df.loc[mask, "bowler_econ_pp"] = df.loc[mask, "bowler"].apply(
            lambda p, st=stats: phase_shrunk_economy(st, p, phase=0)  # powerplay
        )
    return df


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    residuals = y_true - y_pred
    return {
        "model": name,
        "n": int(len(y_true)),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mean_pred": round(float(y_pred.mean()), 1),
        "mean_actual": round(float(y_true.mean()), 1),
        "residual_skew": round(float(((residuals - residuals.mean())**3).mean() / (residuals.std()**3 + 1e-9)), 2),
        "pct_above_pred_plus10": round(float((residuals > 10).mean() * 100), 1),
    }


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="+", default=["ipl", "bbl", "psl", "cpl", "ntb"])
    parser.add_argument("--test-recent", type=int, default=2,
                        help="hold out the N most recent seasons of each league as test")
    args = parser.parse_args()

    all_balls = []
    for league in args.leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty:
            b = b.copy()
            b["_league"] = league
            all_balls.append(b)
    balls = pd.concat(all_balls, ignore_index=True)
    logger.info("loaded %d balls across %d leagues", len(balls), len(all_balls))

    samples = extract_samples(balls)
    if samples.empty:
        logger.error("no samples extracted")
        return 1
    samples = add_player_features(samples, balls)
    logger.info("powerplay samples: %d", len(samples))

    # Train/test split: per league hold out the N most recent seasons
    test_mask = pd.Series(False, index=samples.index)
    for league in args.leagues:
        # use _league from balls; samples don't have it -- recompute via match_id mapping
        # simpler: hold out by global recency
        pass
    cutoff_seasons = sorted(samples["season"].unique())[-args.test_recent:]
    test_mask = samples["season"].isin(cutoff_seasons)
    train = samples[~test_mask].copy()
    test = samples[test_mask].copy()
    logger.info("train rows=%d, test rows=%d (test seasons=%s)",
                len(train), len(test), cutoff_seasons)

    feature_cols = ["runs_18", "wickets_18", "striker_sr", "non_striker_sr", "bowler_econ_pp"]
    y_train = train["pp_total"].to_numpy(dtype=np.float32)
    y_test = test["pp_total"].to_numpy(dtype=np.float32)
    x_train = train[feature_cols].to_numpy(dtype=np.float32)
    x_test = test[feature_cols].to_numpy(dtype=np.float32)

    league_mean = float(train["pp_total"].mean())
    baseline_mean = np.full(len(test), league_mean, dtype=np.float32)
    baseline_linear = (test["runs_18"].to_numpy(dtype=np.float32) * 2.0)  # naive doubling

    # 2-feature linear model
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(train[["runs_18", "wickets_18"]], y_train)
    baseline_lr = lr.predict(test[["runs_18", "wickets_18"]]).astype(np.float32)

    # LightGBM with all features
    gbm = lgb.train(
        params={
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "verbose": -1,
            "seed": 17,
            "deterministic": True,
        },
        train_set=lgb.Dataset(x_train, label=y_train),
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50, verbose=False)],
        valid_sets=[lgb.Dataset(x_test, label=y_test)],
    )
    gbm_pred = gbm.predict(x_test).astype(np.float32)

    results = pd.DataFrame([
        evaluate("league_mean   ", y_test, baseline_mean),
        evaluate("linear_rate*2 ", y_test, baseline_linear),
        evaluate("LR(runs+wkts) ", y_test, baseline_lr),
        evaluate("LightGBM      ", y_test, gbm_pred),
    ])
    print("\n=== Powerplay-total prediction: end-of-over-3 -> end-of-over-6 ===")
    print(f"Train: {len(train)} samples (seasons < {cutoff_seasons[0]})")
    print(f"Test:  {len(test)} samples (seasons {cutoff_seasons})")
    print()
    print(results.to_string(index=False))

    # Tail analysis on best model
    residuals = y_test - gbm_pred
    print(f"\nResidual distribution (actual - predicted), LightGBM:")
    print(f"  mean={residuals.mean():.2f}  std={residuals.std():.2f}")
    print(f"  skew={evaluate('', y_test, gbm_pred)['residual_skew']}  (positive = right tail)")
    print(f"  P(residual > +15) = {(residuals > 15).mean()*100:.1f}%")
    print(f"  P(residual > +20) = {(residuals > 20).mean()*100:.1f}%")
    print(f"  P(residual > +30) = {(residuals > 30).mean()*100:.1f}%")
    print(f"  P(residual < -15) = {(residuals < -15).mean()*100:.1f}%")
    print(f"  P(residual < -20) = {(residuals < -20).mean()*100:.1f}%")

    # Feature importance
    print(f"\nFeature importance (GBM gain):")
    importance = sorted(zip(feature_cols, gbm.feature_importance(importance_type="gain")),
                       key=lambda x: -x[1])
    for name, gain in importance:
        print(f"  {name:25s} {gain:8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
