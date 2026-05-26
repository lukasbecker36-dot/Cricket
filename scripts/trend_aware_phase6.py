"""Trend-aware phase_6 model + strict OOS test.

The baseline phase_6 prior is an equal-weight average of ALL prior seasons,
which lags the fast-rising powerplay scoring trend. This script tests three
prior variants against the equal-weight baseline, all trained <=2024 and
evaluated on 2025+ Line markets:

  - baseline:   equal-weight all prior seasons (current production)
  - recency:    exponential decay, half-life 2 seasons
  - last2:      only the 2 most recent prior seasons
  - recency+trend: recency priors + a league-prev-season PP average feature

The winning variant (if any beats baseline OOS) becomes the new phase_6.
"""
from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

import os

LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]
# Phase configurable via env so the same script validates phase_6 and phase_10.
_PHASE = os.environ.get("TREND_PHASE", "phase_6")
_CFG = {"phase_6": (36, 20, 110, 5), "phase_10": (60, 40, 175, 5)}[_PHASE]
TARGET_BALLS, X_MIN, X_MAX, X_STEP = _CFG
MAX_TRAIN_SEASON = 2024
HALF_LIFE = 2.0  # seasons


def phase_total(g, tb):
    g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    li = np.where(g["is_legal_delivery"].values)[0]
    if len(li) < tb:
        return None
    return int(g.iloc[:li[tb-1]+1]["runs_total"].sum())


def collect_phase_df(balls):
    rows = []
    for (mid, inn), g in balls.groupby(["match_id", "innings"]):
        if int(inn) != 1:
            continue
        t = phase_total(g, TARGET_BALLS)
        if t is None:
            continue
        rows.append({"match_id": mid, "season": int(g["season"].iloc[0]),
                     "batting_team": g["batting_team"].iloc[0],
                     "bowling_team": g["bowling_team"].iloc[0],
                     "venue": g["venue"].iloc[0], "total": t})
    return pd.DataFrame(rows)


def build_priors(pp_df, mode):
    """mode in {'baseline','recency','last2'}. Returns (bat, bowl, ven, league_trend)."""
    bat, bowl, ven, league_trend = {}, {}, {}, {}
    seasons = sorted(pp_df["season"].unique())
    # league trend: need league mapping; computed separately and passed in via column
    for s in seasons:
        if mode == "last2":
            prior = pp_df[(pp_df["season"] < s) & (pp_df["season"] >= s - 2)]
        else:
            prior = pp_df[pp_df["season"] < s]
        if prior.empty:
            continue
        if mode == "recency":
            w = np.power(0.5, (s - prior["season"]) / HALF_LIFE)
            prior = prior.assign(_w=w)
            def wmean(grpcol):
                out = {}
                for key, gg in prior.groupby(grpcol):
                    out[f"{key}|{int(s)}"] = float(np.average(gg["total"], weights=gg["_w"]))
                return out
            bat.update(wmean("batting_team"))
            bowl.update(wmean("bowling_team"))
            ven.update(wmean("venue"))
        else:
            for t, m in prior.groupby("batting_team")["total"].mean().items():
                bat[f"{t}|{int(s)}"] = float(m)
            for t, m in prior.groupby("bowling_team")["total"].mean().items():
                bowl[f"{t}|{int(s)}"] = float(m)
            for v, m in prior.groupby("venue")["total"].mean().items():
                ven[f"{v}|{int(s)}"] = float(m)
        # league trend = mean PP total in the single previous season
        prev = pp_df[pp_df["season"] == s - 1]
        if not prev.empty:
            league_trend[int(s)] = float(prev["total"].mean())
    return bat, bowl, ven, league_trend


def make_train_rows(pp_df, league_by_match, bat, bowl, ven, league_trend, default_par, use_trend):
    rows = []
    for r in pp_df.itertuples(index=False):
        s = int(r.season)
        bp = bat.get(f"{r.batting_team}|{s}", default_par)
        wp = bowl.get(f"{r.bowling_team}|{s}", default_par)
        vp = ven.get(f"{r.venue}|{s}", default_par)
        lt = league_trend.get(s, default_par)
        league = league_by_match.get(str(r.match_id), "unknown")
        for X in range(X_MIN, X_MAX + 1, X_STEP):
            row = {"season": s, "league": league, "threshold_X": X,
                   "bat_prior": bp, "bowl_prior": wp, "venue_par": vp,
                   "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp,
                   "innings": 1, "actual_over_X": int(r.total >= X)}
            if use_trend:
                row["league_trend"] = lt
                row["x_minus_trend"] = X - lt
            rows.append(row)
    df = pd.DataFrame(rows)
    for L in LEAGUES:
        df[f"is_{L}"] = (df["league"] == L).astype(int)
    return df


def train_and_eval(mode, use_trend, pp_all, pp_train, league_by_match, eval_df, default_par):
    bat, bowl, ven, league_trend = build_priors(pp_all, mode)
    feats = ["threshold_X", "bat_prior", "bowl_prior", "venue_par",
             "x_minus_par", "x_minus_bat", "x_minus_bowl", "innings"]
    if use_trend:
        feats += ["league_trend", "x_minus_trend"]
    feats += [f"is_{L}" for L in LEAGUES]

    train_df = make_train_rows(pp_train, league_by_match, bat, bowl, ven, league_trend, default_par, use_trend)
    train_df = train_df.dropna(subset=feats + ["actual_over_X"])
    x = train_df[feats].to_numpy(dtype=np.float32)
    y = train_df["actual_over_X"].to_numpy(dtype=np.float32)
    cut = int(len(x) * 0.9)
    booster = lgb.train(
        params={"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
                "num_leaves": 31, "min_data_in_leaf": 200, "verbose": -1, "seed": 17, "deterministic": True},
        train_set=lgb.Dataset(x[:cut], label=y[:cut], feature_name=feats),
        num_boost_round=500, valid_sets=[lgb.Dataset(x[cut:], label=y[cut:], feature_name=feats)],
        callbacks=[lgb.early_stopping(50, verbose=False)])

    def score(row):
        s = int(row["season"])
        bp = bat.get(f"{row['batting_team']}|{s}", default_par)
        wp = bowl.get(f"{row['bowling_team']}|{s}", default_par)
        vp = ven.get(f"{row['venue']}|{s}", default_par)
        lt = league_trend.get(s, default_par)
        X = int(round(row["line_t_minus_1"]))
        r = {"threshold_X": float(X), "bat_prior": bp, "bowl_prior": wp, "venue_par": vp,
             "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp, "innings": int(row["innings"])}
        if use_trend:
            r["league_trend"] = lt
            r["x_minus_trend"] = X - lt
        for L in LEAGUES:
            r[f"is_{L}"] = 1.0 if row["league"] == L else 0.0
        xv = np.array([[r.get(f, 0.0) for f in feats]], dtype=np.float32)
        return float(booster.predict(xv)[0])

    sub = eval_df.copy()
    sub["p"] = sub.apply(score, axis=1)
    sub["actual_over"] = (sub["actual_total"] > sub["line_t_minus_1"]).astype(int)
    sub["actual_under"] = (sub["actual_total"] < sub["line_t_minus_1"]).astype(int)

    def bt(df_in, mid_only):
        if mid_only:
            df_in = df_in[(df_in["p"] - 0.5).abs().between(0.05, 0.30)]
        sig = np.where(df_in["p"] >= 0.55, "over", np.where(df_in["p"] <= 0.45, "under", "skip"))
        mask = sig != "skip"
        active, s = df_in[mask], sig[mask]
        if active.empty:
            return None
        won = np.where(s == "over", active["actual_over"] == 1, active["actual_under"] == 1)
        pnl = np.where(won, 100 * (2.0 - 1) * 0.95, -100.0)
        return {"n": len(active), "roi": pnl.sum() / (len(active) * 100), "win": won.mean()}

    return bt(sub, False), bt(sub, True)


def main() -> int:
    configure_logging()
    all_balls, league_by_match = [], {}
    for league in LEAGUES:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
        m = pd.read_parquet(d / "matches.parquet")
        for _, r in m.iterrows():
            league_by_match[str(r["match_id"])] = league
    balls = pd.concat(all_balls, ignore_index=True)

    pp_all = collect_phase_df(balls)
    pp_train = pp_all[pp_all["season"] <= MAX_TRAIN_SEASON]
    default_par = float(pp_all[pp_all["season"] <= MAX_TRAIN_SEASON]["total"].mean())

    eval_df = pd.read_parquet("data/processed/eval_phase_lines.parquet")
    eval_df = eval_df[(eval_df["phase"] == _PHASE) & (eval_df["season"] >= 2025)].copy()
    logger.info("2025+ %s eval rows: %d", _PHASE, len(eval_df))

    print(f"\n{'variant':<22} {'all n':>6} {'all ROI':>9} {'all win':>8} | {'mid n':>6} {'mid ROI':>9} {'mid win':>8}")
    print("-" * 80)
    configs = [
        ("baseline", "baseline", False),
        ("recency(hl=2)", "recency", False),
        ("last2", "last2", False),
        ("recency+trend", "recency", True),
        ("last2+trend", "last2", True),
    ]
    for label, mode, use_trend in configs:
        full, mid = train_and_eval(mode, use_trend, pp_all, pp_train, league_by_match, eval_df, default_par)
        fs = f"{full['n']:>6} {full['roi']:>+8.2%} {full['win']:>+7.1%}" if full else f"{'-':>6} {'-':>9} {'-':>8}"
        ms = f"{mid['n']:>6} {mid['roi']:>+8.2%} {mid['win']:>+7.1%}" if mid else f"{'-':>6} {'-':>9} {'-':>8}"
        print(f"{label:<22} {fs} | {ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
