"""Does adding player-strength features improve OOS edge?

For phase_6/10/15/full: train baseline (current production features) vs
+player (baseline + bat_strength + bowl_strength), both on seasons <=2024,
evaluate on 2025+ Line markets. £5 stake. Player features merged on
(match_id, innings) from innings_player_strength.parquet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ingestion.storage import read_balls
from src.ingestion.teams import canonical_team
from src.ingestion.venues import canonical_venue
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]
HALF_LIFE, MAX_TRAIN, STAKE, ODDS, COMM = 2.0, 2024, 5.0, 2.0, 0.05
PHASES = [("phase_6", "phase_6", 36, 20, 110, 5), ("phase_10", "phase_10", 60, 40, 175, 5),
          ("phase_15", "phase_15", 90, 60, 250, 5), ("full_innings", "full_innings", 120, 80, 280, 5)]


def phase_total(g, tb):
    g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    li = np.where(g["is_legal_delivery"].values)[0]
    if tb >= 120:
        wk = int(g["wicket"].fillna(False).astype(bool).sum())
        return int(g["runs_total"].sum()) if (len(li) >= 118 or wk >= 10) else None
    if len(li) < tb:
        return None
    return int(g.iloc[:li[tb-1]+1]["runs_total"].sum())


def train_booster(df, feats):
    df = df.dropna(subset=feats + ["actual_over_X"])
    x = df[feats].to_numpy(dtype=np.float32); y = df["actual_over_X"].to_numpy(dtype=np.float32)
    cut = int(len(x) * 0.9)
    return lgb.train(params={"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
        "num_leaves": 31, "min_data_in_leaf": 200, "verbose": -1, "seed": 17, "deterministic": True},
        train_set=lgb.Dataset(x[:cut], label=y[:cut], feature_name=feats), num_boost_round=500,
        valid_sets=[lgb.Dataset(x[cut:], label=y[cut:], feature_name=feats)],
        callbacks=[lgb.early_stopping(50, verbose=False)])


def bt(df, col):
    sig = np.where(df[col] >= 0.55, "over", np.where(df[col] <= 0.45, "under", "skip"))
    mask = sig != "skip"; active, s = df[mask], sig[mask]
    if active.empty: return None
    won = np.where(s == "over", active["actual_over"] == 1, active["actual_under"] == 1)
    pnl = np.where(won, STAKE * (ODDS - 1) * (1 - COMM), -STAKE)
    return {"n": len(active), "pnl": pnl.sum(), "roi": pnl.sum() / (len(active) * STAKE), "win": won.mean()}


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
    strength = pd.read_parquet("data/processed/innings_player_strength.parquet")
    str_map = {(str(r.match_id), int(r.innings)): (r.bat_strength, r.bowl_strength)
               for r in strength.itertuples(index=False)}
    bat_med = float(strength["bat_strength"].median()); bowl_med = float(strength["bowl_strength"].median())

    eval_all = pd.read_parquet("data/processed/eval_phase_lines.parquet")
    eval_full = pd.read_parquet("data/processed/eval_line_combined.parquet")
    eval_full = eval_full[eval_full["innings"] == 1].copy(); eval_full["phase"] = "full_innings"

    print(f"\n{'phase':<14} {'baseline ROI':>14} {'+player ROI':>14}   (2025+ OOS, £5, full signal)")
    print("-" * 60)
    for plabel, ekey, tb, xmn, xmx, xst in PHASES:
        # phase df (inn1) with canonical names
        rows = []
        for (mid, inn), g in balls.groupby(["match_id", "innings"]):
            if int(inn) != 1: continue
            t = phase_total(g, tb)
            if t is None: continue
            rows.append({"match_id": str(mid), "season": int(g["season"].iloc[0]),
                "batting_team": canonical_team(g["batting_team"].iloc[0]),
                "bowling_team": canonical_team(g["bowling_team"].iloc[0]),
                "venue": canonical_venue(g["venue"].iloc[0]), "total": t})
        pp = pd.DataFrame(rows)
        default_par = float(pp["total"].mean())
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

        tr_rows = []
        for r in pp[pp["season"] <= MAX_TRAIN].itertuples(index=False):
            s = int(r.season)
            bp = bat.get(f"{r.batting_team}|{s}", default_par); wp = bowl.get(f"{r.bowling_team}|{s}", default_par)
            vp = ven.get(f"{r.venue}|{s}", default_par); lt = trend.get(str(s), default_par)
            bs, ws = str_map.get((r.match_id, 1), (bat_med, bowl_med))
            league = league_by_match.get(r.match_id, "unknown")
            for X in range(xmn, xmx + 1, xst):
                tr_rows.append({"season": s, "league": league, "threshold_X": X, "bat_prior": bp,
                    "bowl_prior": wp, "venue_par": vp, "x_minus_par": X - vp, "x_minus_bat": X - bp,
                    "x_minus_bowl": X - wp, "innings": 1, "league_trend": lt, "x_minus_trend": X - lt,
                    "bat_strength": bs, "bowl_strength": ws, "actual_over_X": int(r.total >= X)})
        tr = pd.DataFrame(tr_rows)
        for L in LEAGUES: tr[f"is_{L}"] = (tr["league"] == L).astype(int)
        base_feats = ["threshold_X", "bat_prior", "bowl_prior", "venue_par", "x_minus_par",
                      "x_minus_bat", "x_minus_bowl", "innings", "league_trend", "x_minus_trend"] + [f"is_{L}" for L in LEAGUES]
        play_feats = base_feats + ["bat_strength", "bowl_strength"]
        b_base = train_booster(tr, base_feats); b_play = train_booster(tr, play_feats)

        src = eval_full if ekey == "full_innings" else eval_all
        sub = src[(src["phase"] == ekey) & (src["season"] >= 2025)].copy()
        if sub.empty: continue
        sub["match_id"] = sub["match_id"].astype(str)

        def score(rr, booster, feats):
            s = int(rr["season"])
            bt_, bw_ = canonical_team(rr["batting_team"]), canonical_team(rr["bowling_team"])
            vv = canonical_venue(rr["venue"])
            bp = bat.get(f"{bt_}|{s}", default_par); wp = bowl.get(f"{bw_}|{s}", default_par)
            vp = ven.get(f"{vv}|{s}", default_par); lt = trend.get(str(s), default_par)
            X = int(round(rr["line_t_minus_1"]))
            bs, ws = str_map.get((rr["match_id"], 1), (bat_med, bowl_med))
            d = {"threshold_X": float(X), "bat_prior": bp, "bowl_prior": wp, "venue_par": vp,
                 "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp, "innings": 1,
                 "league_trend": lt, "x_minus_trend": X - lt, "bat_strength": bs, "bowl_strength": ws}
            for L in LEAGUES: d[f"is_{L}"] = 1.0 if rr["league"] == L else 0.0
            return float(booster.predict(np.array([[d.get(f, 0.0) for f in feats]], dtype=np.float32))[0])

        sub["actual_over"] = (sub["actual_total"] > sub["line_t_minus_1"]).astype(int)
        sub["actual_under"] = (sub["actual_total"] < sub["line_t_minus_1"]).astype(int)
        sub["p_base"] = sub.apply(lambda r: score(r, b_base, base_feats), axis=1)
        sub["p_play"] = sub.apply(lambda r: score(r, b_play, play_feats), axis=1)
        rb, rp = bt(sub, "p_base"), bt(sub, "p_play")
        fb = f"n={rb['n']:3d} {rb['roi']:+6.1%}" if rb else "—"
        fp = f"n={rp['n']:3d} {rp['roi']:+6.1%}" if rp else "—"
        print(f"{plabel:<14} {fb:>14} {fp:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
