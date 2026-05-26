"""Anomaly-based weather features + per-league OOS test.

Instead of absolute temp/humidity/wind/cloud, feed deviations from each
venue's climatological normal (mean of its match-day conditions). This makes
'hot' mean the same thing in England and India, removing the league-proxy
confound.

  temp_anom    = temp_c     - venue_mean(temp_c)
  humid_anom   = humidity   - venue_mean(humidity)
  wind_anom    = wind_kph   - venue_mean(wind_kph)
  cloud_anom   = cloud_pct  - venue_mean(cloud_pct)
  precip_mm    = kept absolute (0=dry is meaningful everywhere)

Trains phase_15 on <=2024, evaluates 2025+ per league, comparing:
  baseline (no weather) vs absolute-weather vs anomaly-weather.
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
TB, X_MIN, X_MAX, X_STEP = 90, 60, 250, 5
MAX_TRAIN_SEASON = 2024
ABS_W = ["temp_c", "humidity_pct", "wind_kph", "precip_mm", "cloud_pct"]
ANOM_W = ["temp_anom", "humid_anom", "wind_anom", "precip_mm", "cloud_anom"]


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
        t = phase_total(g, TB)
        if t is None:
            continue
        rows.append({"match_id": mid, "season": int(g["season"].iloc[0]),
                     "batting_team": g["batting_team"].iloc[0],
                     "bowling_team": g["bowling_team"].iloc[0],
                     "venue": g["venue"].iloc[0], "total": t})
    return pd.DataFrame(rows)


def build_priors(pp):
    bat, bowl, ven = {}, {}, {}
    for s in sorted(pp["season"].unique()):
        prior = pp[pp["season"] < s]
        if prior.empty:
            continue
        for t, m in prior.groupby("batting_team")["total"].mean().items():
            bat[f"{t}|{int(s)}"] = float(m)
        for t, m in prior.groupby("bowling_team")["total"].mean().items():
            bowl[f"{t}|{int(s)}"] = float(m)
        for v, m in prior.groupby("venue")["total"].mean().items():
            ven[f"{v}|{int(s)}"] = float(m)
    return bat, bowl, ven


def train_booster(df, feats):
    df = df.dropna(subset=feats + ["actual_over_X"])
    x = df[feats].to_numpy(dtype=np.float32)
    y = df["actual_over_X"].to_numpy(dtype=np.float32)
    cut = int(len(x) * 0.9)
    return lgb.train(
        params={"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
                "num_leaves": 31, "min_data_in_leaf": 200, "verbose": -1, "seed": 17, "deterministic": True},
        train_set=lgb.Dataset(x[:cut], label=y[:cut], feature_name=feats),
        num_boost_round=500, valid_sets=[lgb.Dataset(x[cut:], label=y[cut:], feature_name=feats)],
        callbacks=[lgb.early_stopping(50, verbose=False)])


def main() -> int:
    configure_logging()
    all_balls, league_by_match, match_dates = [], {}, {}
    for lg in LEAGUES:
        d = Path("data/processed") if lg == "ipl" else Path(f"data/processed/{lg}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
        m = pd.read_parquet(d / "matches.parquet")
        for _, r in m.iterrows():
            league_by_match[str(r["match_id"])] = lg
            match_dates[str(r["match_id"])] = str(r["date"])[:10]
    balls = pd.concat(all_balls, ignore_index=True)
    weather = pd.read_parquet("data/processed/match_weather.parquet")

    # venue climatology (mean of match-day conditions per venue)
    climo = weather.groupby("venue")[["temp_c", "humidity_pct", "wind_kph", "cloud_pct"]].mean()
    global_climo = weather[["temp_c", "humidity_pct", "wind_kph", "cloud_pct"]].mean()

    def anom(venue, var, val):
        if pd.isna(val):
            return None
        base = climo[var].get(venue, global_climo[var]) if venue in climo.index else global_climo[var]
        return val - base

    pp = collect_phase_df(balls)
    bat, bowl, ven = build_priors(pp)
    default_par = float(pp[pp["season"] <= MAX_TRAIN_SEASON]["total"].mean())
    medians_abs = {f: float(weather[f].median()) for f in ABS_W}

    # Build full row set with both absolute and anomaly weather
    def make_rows(pp_sub):
        rows = []
        for r in pp_sub.itertuples(index=False):
            s = int(r.season)
            bp = bat.get(f"{r.batting_team}|{s}", default_par)
            wp = bowl.get(f"{r.bowling_team}|{s}", default_par)
            vp = ven.get(f"{r.venue}|{s}", default_par)
            league = league_by_match.get(str(r.match_id), "unknown")
            mdate = match_dates.get(str(r.match_id))
            wx = weather[(weather["venue"] == r.venue) & (weather["date"] == mdate)]
            if not wx.empty:
                w = wx.iloc[0]
                absf = {f: (w[f] if not pd.isna(w[f]) else medians_abs[f]) for f in ABS_W}
            else:
                absf = dict(medians_abs)
            anomf = {
                "temp_anom": anom(r.venue, "temp_c", absf["temp_c"]) or 0.0,
                "humid_anom": anom(r.venue, "humidity_pct", absf["humidity_pct"]) or 0.0,
                "wind_anom": anom(r.venue, "wind_kph", absf["wind_kph"]) or 0.0,
                "cloud_anom": anom(r.venue, "cloud_pct", absf["cloud_pct"]) or 0.0,
                "precip_mm": absf["precip_mm"],
            }
            for X in range(X_MIN, X_MAX + 1, X_STEP):
                row = {"season": s, "league": league, "threshold_X": X,
                       "bat_prior": bp, "bowl_prior": wp, "venue_par": vp,
                       "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp,
                       "innings": 1, "actual_over_X": int(r.total >= X)}
                row.update(absf)
                row.update(anomf)
                rows.append(row)
        df = pd.DataFrame(rows)
        for L in LEAGUES:
            df[f"is_{L}"] = (df["league"] == L).astype(int)
        return df

    train_rows = make_rows(pp[pp["season"] <= MAX_TRAIN_SEASON])
    base_feats = ["threshold_X", "bat_prior", "bowl_prior", "venue_par",
                  "x_minus_par", "x_minus_bat", "x_minus_bowl", "innings"] + [f"is_{L}" for L in LEAGUES]
    abs_feats = base_feats[:8] + ABS_W + [f"is_{L}" for L in LEAGUES]
    anom_feats = base_feats[:8] + ANOM_W + [f"is_{L}" for L in LEAGUES]

    b_base = train_booster(train_rows, base_feats)
    b_abs = train_booster(train_rows, abs_feats)
    b_anom = train_booster(train_rows, anom_feats)

    # Eval on 2025+ line markets
    eval_df = pd.read_parquet("data/processed/eval_phase_lines.parquet")
    eval_df = eval_df[(eval_df["phase"] == "phase_15") & (eval_df["season"] >= 2025)].copy()
    eval_df["date"] = eval_df["match_id"].astype(str).map(match_dates)
    eval_df = eval_df.merge(weather, on=["venue", "date"], how="left")
    eval_df["actual_over"] = (eval_df["actual_total"] > eval_df["line_t_minus_1"]).astype(int)
    eval_df["actual_under"] = (eval_df["actual_total"] < eval_df["line_t_minus_1"]).astype(int)

    def score(row, booster, feats):
        s = int(row["season"])
        bp = bat.get(f"{row['batting_team']}|{s}", default_par)
        wp = bowl.get(f"{row['bowling_team']}|{s}", default_par)
        vp = ven.get(f"{row['venue']}|{s}", default_par)
        X = int(round(row["line_t_minus_1"]))
        r = {"threshold_X": float(X), "bat_prior": bp, "bowl_prior": wp, "venue_par": vp,
             "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp, "innings": int(row["innings"])}
        absf = {f: (row[f] if (f in row and not pd.isna(row[f])) else medians_abs[f]) for f in ABS_W}
        r.update(absf)
        r["temp_anom"] = anom(row["venue"], "temp_c", absf["temp_c"]) or 0.0
        r["humid_anom"] = anom(row["venue"], "humidity_pct", absf["humidity_pct"]) or 0.0
        r["wind_anom"] = anom(row["venue"], "wind_kph", absf["wind_kph"]) or 0.0
        r["cloud_anom"] = anom(row["venue"], "cloud_pct", absf["cloud_pct"]) or 0.0
        for L in LEAGUES:
            r[f"is_{L}"] = 1.0 if row["league"] == L else 0.0
        xv = np.array([[r.get(f, 0.0) for f in feats]], dtype=np.float32)
        return float(booster.predict(xv)[0])

    eval_df["p_base"] = eval_df.apply(lambda r: score(r, b_base, base_feats), axis=1)
    eval_df["p_abs"] = eval_df.apply(lambda r: score(r, b_abs, abs_feats), axis=1)
    eval_df["p_anom"] = eval_df.apply(lambda r: score(r, b_anom, anom_feats), axis=1)

    def bt(df_in, col):
        sig = np.where(df_in[col] >= 0.55, "over", np.where(df_in[col] <= 0.45, "under", "skip"))
        mask = sig != "skip"
        active, s = df_in[mask], sig[mask]
        if active.empty:
            return None
        won = np.where(s == "over", active["actual_over"] == 1, active["actual_under"] == 1)
        pnl = np.where(won, 95.0, -100.0)
        return {"n": len(active), "roi": pnl.sum() / (len(active) * 100), "win": won.mean()}

    print(f"\n{'segment':<14} {'base':>20} {'absolute-wx':>20} {'anomaly-wx':>20}")
    print("-" * 78)

    def fmt(r):
        return f"n={r['n']:>3} {r['roi']:>+7.1%} {r['win']:>5.0%}" if r else "        -          "

    # Overall
    print(f"{'ALL':<14} {fmt(bt(eval_df,'p_base')):>20} {fmt(bt(eval_df,'p_abs')):>20} {fmt(bt(eval_df,'p_anom')):>20}")
    # Per league
    for lg in LEAGUES:
        sub = eval_df[eval_df["league"] == lg]
        if len(sub) < 3:
            continue
        print(f"{lg.upper():<14} {fmt(bt(sub,'p_base')):>20} {fmt(bt(sub,'p_abs')):>20} {fmt(bt(sub,'p_anom')):>20}")
    # England-relevant grouping: NTB + BBL (cooler leagues)
    cool = eval_df[eval_df["league"].isin(["ntb", "bbl"])]
    if len(cool) >= 3:
        print(f"{'NTB+BBL(cool)':<14} {fmt(bt(cool,'p_base')):>20} {fmt(bt(cool,'p_abs')):>20} {fmt(bt(cool,'p_anom')):>20}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
