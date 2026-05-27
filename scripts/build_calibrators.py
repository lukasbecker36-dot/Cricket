"""Fit Platt-scaling calibrators for phase_6 and phase_10.

The audit showed these two under-predict the over-rate (residual scoring-
inflation lag), ECE ~0.12. We correct it with a 2-parameter logistic map
fit on WALK-FORWARD OOS predictions: for each recent season S, train a model
on seasons <= S-1 and predict season S across the threshold sweep, giving
leak-free (raw_p, actual_over) pairs. Pool, fit logistic(y ~ logit(raw_p)).

Saves models/phase_{6,10}_platt.json = {"A": slope, "B": intercept}.
signals.py applies calibrated_p = sigmoid(A*logit(raw_p) + B) when present.
phase_15 / full_innings are already calibrated and get no calibrator.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ingestion.storage import read_balls
from src.ingestion.teams import canonical_team
from src.ingestion.venues import canonical_venue
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]
HALF_LIFE = 2.0
CALIB_SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]  # walk-forward eval seasons
PHASES = [("phase_6", 36, 20, 110, 5), ("phase_10", 60, 40, 175, 5)]


def phase_total(g, tb):
    g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    li = np.where(g["is_legal_delivery"].values)[0]
    if len(li) < tb:
        return None
    return int(g.iloc[:li[tb-1]+1]["runs_total"].sum())


def train_booster(df, feats):
    df = df.dropna(subset=feats + ["actual_over_X"])
    x = df[feats].to_numpy(dtype=np.float32); y = df["actual_over_X"].to_numpy(dtype=np.float32)
    cut = int(len(x) * 0.9)
    return lgb.train(params={"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
        "num_leaves": 31, "min_data_in_leaf": 200, "verbose": -1, "seed": 17, "deterministic": True},
        train_set=lgb.Dataset(x[:cut], label=y[:cut], feature_name=feats), num_boost_round=400,
        valid_sets=[lgb.Dataset(x[cut:], label=y[cut:], feature_name=feats)],
        callbacks=[lgb.early_stopping(40, verbose=False)])


def ece(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return e


def main() -> int:
    configure_logging()
    all_balls, league_by_match = [], {}
    for lg in LEAGUES:
        d = Path("data/processed") if lg == "ipl" else Path(f"data/processed/{lg}")
        b = read_balls(d)
        if not b.empty: all_balls.append(b)
        m = pd.read_parquet(d / "matches.parquet")
        for _, r in m.iterrows(): league_by_match[str(r["match_id"])] = lg
    balls = pd.concat(all_balls, ignore_index=True)
    model_dir = Path("models")

    for plabel, tb, xmn, xmx, xst in PHASES:
        # phase data (inn1, canonical)
        rows = []
        for (mid, inn), g in balls.groupby(["match_id", "innings"]):
            if int(inn) != 1: continue
            t = phase_total(g, tb)
            if t is None: continue
            rows.append({"match_id": str(mid), "season": int(g["season"].iloc[0]),
                "batting_team": canonical_team(g["batting_team"].iloc[0]),
                "bowling_team": canonical_team(g["bowling_team"].iloc[0]),
                "venue": canonical_venue(g["venue"].iloc[0]), "total": t})
        pp = pd.DataFrame(rows); default_par = float(pp["total"].mean())
        # leak-free priors (per-season keys use strictly prior data)
        bat, bowl, ven, trend = {}, {}, {}, {}
        for s in sorted(pp["season"].unique()):
            prior = pp[pp["season"] < s]
            if prior.empty: continue
            w = np.power(0.5, (s - prior["season"]) / HALF_LIFE); prior = prior.assign(_w=w)
            for col, tbl in [("batting_team", bat), ("bowling_team", bowl), ("venue", ven)]:
                for key, gg in prior.groupby(col):
                    tbl[f"{key}|{int(s)}"] = float(np.average(gg["total"], weights=gg["_w"]))
            prev = pp[pp["season"] == s - 1]
            if not prev.empty: trend[str(int(s))] = float(prev["total"].mean())
        # full training-row set (all seasons, X-sweep)
        rr = []
        for r in pp.itertuples(index=False):
            s = int(r.season)
            bp = bat.get(f"{r.batting_team}|{s}", default_par); wp = bowl.get(f"{r.bowling_team}|{s}", default_par)
            vp = ven.get(f"{r.venue}|{s}", default_par); lt = trend.get(str(s), default_par)
            league = league_by_match.get(r.match_id, "unknown")
            for X in range(xmn, xmx + 1, xst):
                rr.append({"season": s, "league": league, "threshold_X": X, "bat_prior": bp,
                    "bowl_prior": wp, "venue_par": vp, "x_minus_par": X - vp, "x_minus_bat": X - bp,
                    "x_minus_bowl": X - wp, "innings": 1, "league_trend": lt, "x_minus_trend": X - lt,
                    "actual_over_X": int(r.total >= X)})
        allrows = pd.DataFrame(rr)
        for L in LEAGUES: allrows[f"is_{L}"] = (allrows["league"] == L).astype(int)
        feats = ["threshold_X", "bat_prior", "bowl_prior", "venue_par", "x_minus_par", "x_minus_bat",
                 "x_minus_bowl", "innings", "league_trend", "x_minus_trend"] + [f"is_{L}" for L in LEAGUES]

        # walk-forward OOS predictions
        wf_p, wf_y = [], []
        for S in CALIB_SEASONS:
            tr = allrows[allrows["season"] <= S - 1]
            te = allrows[allrows["season"] == S]
            if len(tr) < 1000 or te.empty: continue
            booster = train_booster(tr, feats)
            wf_p.append(booster.predict(te[feats].to_numpy(dtype=np.float32)))
            wf_y.append(te["actual_over_X"].to_numpy())
        p = np.clip(np.concatenate(wf_p), 1e-6, 1 - 1e-6); y = np.concatenate(wf_y)

        # fit Platt: y ~ logit(p)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression()
        lr.fit(logit, y)
        A = float(lr.coef_[0][0]); B = float(lr.intercept_[0])
        cal = lr.predict_proba(logit)[:, 1]
        logger.info("%s: walk-forward n=%d  ECE raw=%.3f -> calibrated=%.3f  (A=%.3f B=%.3f)",
                    plabel, len(p), ece(p, y), ece(cal, y), A, B)
        (model_dir / f"{plabel}_platt.json").write_text(json.dumps({"A": A, "B": B}))
        print(f"{plabel}: saved Platt (A={A:.3f}, B={B:.3f}); "
              f"mean raw_p={p.mean():.3f} -> calibrated={cal.mean():.3f} (base over-rate={y.mean():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
