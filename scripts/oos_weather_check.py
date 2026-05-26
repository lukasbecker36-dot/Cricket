"""Strict OOS check for weather features.

Trains TWO models on seasons <= 2024 only:
  - baseline (no weather)
  - weather (with temp/humidity/wind/precip/cloud)
Then evaluates BOTH on 2025+ Line markets (truly unseen). Reports whether
the weather model's edge survives out-of-sample.

Nothing is saved to models/ -- this is a pure validation run.
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

LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]
WEATHER_FEATURES = ["temp_c", "humidity_pct", "wind_kph", "precip_mm", "cloud_pct"]
MAX_TRAIN_SEASON = 2024
PHASES = [("phase_6", 36, 20, 110, 5), ("phase_10", 60, 40, 175, 5), ("phase_15", 90, 60, 250, 5)]


def phase_total(g, target_balls):
    g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    legal_idx = np.where(g["is_legal_delivery"].values)[0]
    if len(legal_idx) < target_balls:
        return None
    cut = legal_idx[target_balls - 1] + 1
    return int(g.iloc[:cut]["runs_total"].sum())


def build_priors(pp_df):
    bat, bowl, ven = {}, {}, {}
    for s in sorted(pp_df["season"].unique()):
        prior = pp_df[pp_df["season"] < s]
        if prior.empty:
            continue
        for t, m in prior.groupby("batting_team")["total"].mean().items():
            bat[f"{t}|{int(s)}"] = float(m)
        for t, m in prior.groupby("bowling_team")["total"].mean().items():
            bowl[f"{t}|{int(s)}"] = float(m)
        for v, m in prior.groupby("venue")["total"].mean().items():
            ven[f"{v}|{int(s)}"] = float(m)
    return bat, bowl, ven


def collect_phase_df(balls, target_balls):
    rows = []
    for (mid, innings), g in balls.groupby(["match_id", "innings"]):
        if int(innings) != 1:
            continue
        total = phase_total(g, target_balls)
        if total is None:
            continue
        rows.append({"match_id": mid, "season": int(g["season"].iloc[0]),
                     "batting_team": g["batting_team"].iloc[0],
                     "bowling_team": g["bowling_team"].iloc[0],
                     "venue": g["venue"].iloc[0], "total": total})
    return pd.DataFrame(rows)


def build_train_rows(pp_df, league_by_match, match_dates, weather, bat_pp, bowl_pp, ven_par,
                     default_par, x_min, x_max, x_step, medians):
    rows = []
    for r in pp_df.itertuples(index=False):
        season = int(r.season)
        bat_prior = bat_pp.get(f"{r.batting_team}|{season}", default_par)
        bowl_prior = bowl_pp.get(f"{r.bowling_team}|{season}", default_par)
        venue_par = ven_par.get(f"{r.venue}|{season}", default_par)
        league = league_by_match.get(str(r.match_id), "unknown")
        mdate = match_dates.get(str(r.match_id))
        wx = weather[(weather["venue"] == r.venue) & (weather["date"] == mdate)]
        wx_feats = {f: (wx[f].iloc[0] if not wx.empty and not pd.isna(wx[f].iloc[0]) else medians[f])
                    for f in WEATHER_FEATURES}
        for X in range(x_min, x_max + 1, x_step):
            base = {"season": season, "league": league, "threshold_X": X,
                    "bat_prior": bat_prior, "bowl_prior": bowl_prior, "venue_par": venue_par,
                    "x_minus_par": X - venue_par, "x_minus_bat": X - bat_prior,
                    "x_minus_bowl": X - bowl_prior, "innings": 1,
                    "actual_over_X": int(r.total >= X)}
            base.update(wx_feats)
            rows.append(base)
    df = pd.DataFrame(rows)
    for L in LEAGUES:
        df[f"is_{L}"] = (df["league"] == L).astype(int)
    return df


def train_booster(df, features):
    df = df.dropna(subset=features + ["actual_over_X"])
    x = df[features].to_numpy(dtype=np.float32)
    y = df["actual_over_X"].to_numpy(dtype=np.float32)
    cut = int(len(x) * 0.9)
    return lgb.train(
        params={"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
                "num_leaves": 31, "min_data_in_leaf": 200, "verbose": -1, "seed": 17, "deterministic": True},
        train_set=lgb.Dataset(x[:cut], label=y[:cut], feature_name=features),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(x[cut:], label=y[cut:], feature_name=features)],
        callbacks=[lgb.early_stopping(50, verbose=False)])


def main() -> int:
    configure_logging()
    all_balls, match_dates, league_by_match = [], {}, {}
    for league in LEAGUES:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
        m = pd.read_parquet(d / "matches.parquet")
        for _, r in m.iterrows():
            league_by_match[str(r["match_id"])] = league
            match_dates[str(r["match_id"])] = str(r["date"])[:10]
    balls = pd.concat(all_balls, ignore_index=True)
    weather = pd.read_parquet("data/processed/match_weather.parquet")
    medians = {f: weather[f].median() for f in WEATHER_FEATURES}

    eval_df_all = pd.read_parquet("data/processed/eval_phase_lines.parquet")
    eval_df_all["date"] = eval_df_all["match_id"].astype(str).map(match_dates)
    eval_df_all = eval_df_all.merge(weather, on=["venue", "date"], how="left")

    base_feats = ["threshold_X", "bat_prior", "bowl_prior", "venue_par",
                  "x_minus_par", "x_minus_bat", "x_minus_bowl", "innings"] + [f"is_{L}" for L in LEAGUES]
    wx_feats = base_feats[:8] + WEATHER_FEATURES + [f"is_{L}" for L in LEAGUES]

    LINE_ODDS, EDGE = 2.0, 0.05
    print(f"\n{'phase':<10} {'model':<10} {'n':>5} {'ROI':>9} {'win':>8}  (2025+ OOS, trained <=2024)")
    print("-" * 60)

    for phase_label, target_balls, x_min, x_max, x_step in PHASES:
        pp_all = collect_phase_df(balls, target_balls)
        # priors built on full history (strict prior-season, leak-free)
        bat_pp, bowl_pp, ven_par = build_priors(pp_all)
        default_par = float(np.mean(list(ven_par.values()))) if ven_par else 50.0
        # train only on <= 2024
        pp_train = pp_all[pp_all["season"] <= MAX_TRAIN_SEASON]
        train_rows = build_train_rows(pp_train, league_by_match, match_dates, weather,
                                      bat_pp, bowl_pp, ven_par, default_par, x_min, x_max, x_step, medians)
        b_base = train_booster(train_rows, base_feats)
        b_wx = train_booster(train_rows, wx_feats)

        # eval on 2025+ for this phase
        sub = eval_df_all[(eval_df_all["phase"] == phase_label) & (eval_df_all["season"] >= 2025)].copy()
        if sub.empty:
            print(f"{phase_label:<10} (no 2025+ eval rows)")
            continue

        def score(row, booster, feats):
            season = int(row["season"])
            bp = bat_pp.get(f"{row['batting_team']}|{season}", default_par)
            wp = bowl_pp.get(f"{row['bowling_team']}|{season}", default_par)
            vp = ven_par.get(f"{row['venue']}|{season}", default_par)
            X = int(round(row["line_t_minus_1"]))
            r = {"threshold_X": float(X), "bat_prior": bp, "bowl_prior": wp, "venue_par": vp,
                 "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp, "innings": int(row["innings"])}
            for f in WEATHER_FEATURES:
                v = row.get(f)
                r[f] = medians[f] if (v is None or pd.isna(v)) else v
            for L in LEAGUES:
                r[f"is_{L}"] = 1.0 if row["league"] == L else 0.0
            x = np.array([[r.get(f, 0.0) for f in feats]], dtype=np.float32)
            return float(booster.predict(x)[0])

        sub["p_base"] = sub.apply(lambda r: score(r, b_base, base_feats), axis=1)
        sub["p_wx"] = sub.apply(lambda r: score(r, b_wx, wx_feats), axis=1)
        sub["actual_over"] = (sub["actual_total"] > sub["line_t_minus_1"]).astype(int)
        sub["actual_under"] = (sub["actual_total"] < sub["line_t_minus_1"]).astype(int)

        def bt(df_in, col, mid_only=False):
            if mid_only:
                df_in = df_in[(df_in[col] - 0.5).abs().between(0.05, 0.30)]
            sig = np.where(df_in[col] >= 0.55, "over", np.where(df_in[col] <= 0.45, "under", "skip"))
            mask = sig != "skip"
            active = df_in[mask]
            s = sig[mask]
            if active.empty:
                return None
            won = np.where(s == "over", active["actual_over"] == 1, active["actual_under"] == 1)
            pnl = np.where(won, 100 * (LINE_ODDS - 1) * 0.95, -100.0)
            return {"n": len(active), "roi": pnl.sum() / (len(active) * 100), "win": won.mean()}

        for label, col in [("baseline", "p_base"), ("weather", "p_wx")]:
            r = bt(sub, col)
            if r:
                print(f"{phase_label:<10} {label:<10} {r['n']:>5} {r['roi']:>+8.2%} {r['win']:>+7.1%}")
        for label, col in [("base-mid", "p_base"), ("wx-mid", "p_wx")]:
            r = bt(sub, col, mid_only=True)
            if r:
                print(f"{phase_label:<10} {label:<10} {r['n']:>5} {r['roi']:>+8.2%} {r['win']:>+7.1%}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
