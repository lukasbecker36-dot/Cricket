"""Consolidated OOS backtest of the PRODUCTION model configs at a £5 stake.

For an honest out-of-sample read, each model is trained on seasons <=2024
(2025+ held out) using the same config now deployed:
  - phase_6, phase_10, full_innings: recency priors (half-life 2) + league_trend
  - phase_15: anomaly-weather (deviation from venue climatology)

Reports n / P&L / ROI / win-rate at £5 flat stake, 2.0 odds, 5% commission,
for both the full signal set and the mid-confidence band [0.05, 0.30].
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
HALF_LIFE = 2.0
MAX_TRAIN_SEASON = 2024
STAKE = 5.0
ODDS = 2.0
COMMISSION = 0.05
ANOM_RAW = {"temp_anom": "temp_c", "humid_anom": "humidity_pct",
            "wind_anom": "wind_kph", "cloud_anom": "cloud_pct"}

# phase_label -> (eval_phase_key, target_balls, x_min, x_max, x_step, mode)
PHASES = [
    ("phase_6",      "phase_6",  36, 20, 110, 5, "trend"),
    ("phase_10",     "phase_10", 60, 40, 175, 5, "trend"),
    ("phase_15",     "phase_15", 90, 60, 250, 5, "anomaly"),
    ("full_innings", "full_innings", 120, 80, 280, 5, "trend"),
]


def phase_total(g, tb):
    g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    li = np.where(g["is_legal_delivery"].values)[0]
    if tb >= 120:
        return int(g["runs_total"].sum()) if len(li) >= 30 else None
    if len(li) < tb:
        return None
    return int(g.iloc[:li[tb-1]+1]["runs_total"].sum())


def collect_phase_df(balls, tb):
    rows = []
    for (mid, inn), g in balls.groupby(["match_id", "innings"]):
        if int(inn) != 1:
            continue
        t = phase_total(g, tb)
        if t is None:
            continue
        rows.append({"match_id": mid, "season": int(g["season"].iloc[0]),
                     "batting_team": g["batting_team"].iloc[0],
                     "bowling_team": g["bowling_team"].iloc[0],
                     "venue": g["venue"].iloc[0], "total": t})
    return pd.DataFrame(rows)


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


def bt(df_in, col, stake, mid_only=False):
    if mid_only:
        df_in = df_in[(df_in[col] - 0.5).abs().between(0.05, 0.30)]
    sig = np.where(df_in[col] >= 0.55, "over", np.where(df_in[col] <= 0.45, "under", "skip"))
    mask = sig != "skip"
    active, s = df_in[mask], sig[mask]
    if active.empty:
        return None
    won = np.where(s == "over", active["actual_over"] == 1, active["actual_under"] == 1)
    pnl = np.where(won, stake * (ODDS - 1) * (1 - COMMISSION), -stake)
    return {"n": int(len(active)), "pnl": float(pnl.sum()),
            "roi": float(pnl.sum() / (len(active) * stake)), "win": float(won.mean())}


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
    climo_vars = ["temp_c", "humidity_pct", "wind_kph", "cloud_pct"]
    venue_climo = {v: {k: float(val) for k, val in d.items()}
                   for v, d in weather.groupby("venue")[climo_vars].mean().to_dict(orient="index").items()}
    global_climo = {k: float(weather[k].mean()) for k in climo_vars}
    medians = {f: float(weather[f].median()) for f in ["temp_c", "humidity_pct", "wind_kph", "precip_mm", "cloud_pct"]}

    eval_all = pd.read_parquet("data/processed/eval_phase_lines.parquet")
    eval_all["date"] = eval_all["match_id"].astype(str).map(match_dates)
    eval_all = eval_all.merge(weather, on=["venue", "date"], how="left")

    # full-innings (20-over) line markets live in a different eval file
    eval_full = pd.read_parquet("data/processed/eval_line_combined.parquet")
    eval_full = eval_full[eval_full["innings"] == 1].copy()
    eval_full["phase"] = "full_innings"

    results = []
    for plabel, ekey, tb, xmn, xmx, xst, mode in PHASES:
        pp = collect_phase_df(balls, tb)
        default_par = float(pp["total"].mean())
        # priors (recency for trend mode, equal-weight for anomaly mode)
        bat, bowl, ven, trend = {}, {}, {}, {}
        for s in sorted(pp["season"].unique()):
            prior = pp[pp["season"] < s]
            if prior.empty:
                continue
            if mode == "trend":
                w = np.power(0.5, (s - prior["season"]) / HALF_LIFE)
                prior = prior.assign(_w=w)
                for col, tbl in [("batting_team", bat), ("bowling_team", bowl), ("venue", ven)]:
                    for key, gg in prior.groupby(col):
                        tbl[f"{key}|{int(s)}"] = float(np.average(gg["total"], weights=gg["_w"]))
                prev = pp[pp["season"] == s - 1]
                if not prev.empty:
                    trend[str(int(s))] = float(prev["total"].mean())
            else:
                for t, m in prior.groupby("batting_team")["total"].mean().items():
                    bat[f"{t}|{int(s)}"] = float(m)
                for t, m in prior.groupby("bowling_team")["total"].mean().items():
                    bowl[f"{t}|{int(s)}"] = float(m)
                for v, m in prior.groupby("venue")["total"].mean().items():
                    ven[f"{v}|{int(s)}"] = float(m)

        # build training rows (<=2024)
        pp_tr = pp[pp["season"] <= MAX_TRAIN_SEASON]
        rows = []
        for r in pp_tr.itertuples(index=False):
            s = int(r.season)
            bp = bat.get(f"{r.batting_team}|{s}", default_par)
            wp = bowl.get(f"{r.bowling_team}|{s}", default_par)
            vp = ven.get(f"{r.venue}|{s}", default_par)
            league = league_by_match.get(str(r.match_id), "unknown")
            base = {"season": s, "league": league, "threshold_X": None, "bat_prior": bp,
                    "bowl_prior": wp, "venue_par": vp, "innings": 1}
            if mode == "trend":
                lt = trend.get(str(s), default_par)
            else:
                mdate = match_dates.get(str(r.match_id))
                wx = weather[(weather["venue"] == r.venue) & (weather["date"] == mdate)]
                raw = {f: (wx[f].iloc[0] if not wx.empty and not pd.isna(wx[f].iloc[0]) else medians[f])
                       for f in medians}
                cl = venue_climo.get(r.venue, global_climo)
            for X in range(xmn, xmx + 1, xst):
                row = dict(base, threshold_X=X, x_minus_par=X - vp,
                           x_minus_bat=X - bp, x_minus_bowl=X - wp, actual_over_X=int(r.total >= X))
                if mode == "trend":
                    row["league_trend"] = lt
                    row["x_minus_trend"] = X - lt
                else:
                    for af, rv in ANOM_RAW.items():
                        row[af] = float(raw[rv]) - float(cl.get(rv, global_climo[rv]))
                    row["precip_mm"] = float(raw["precip_mm"])
                rows.append(row)
        tr = pd.DataFrame(rows)
        for L in LEAGUES:
            tr[f"is_{L}"] = (tr["league"] == L).astype(int)

        base_feats = ["threshold_X", "bat_prior", "bowl_prior", "venue_par",
                      "x_minus_par", "x_minus_bat", "x_minus_bowl", "innings"]
        if mode == "trend":
            feats = base_feats + ["league_trend", "x_minus_trend"] + [f"is_{L}" for L in LEAGUES]
        else:
            feats = base_feats + ["temp_anom", "humid_anom", "wind_anom", "precip_mm", "cloud_anom"] + [f"is_{L}" for L in LEAGUES]
        booster = train_booster(tr, feats)

        # eval 2025+ (full_innings uses its own eval file)
        src = eval_full if ekey == "full_innings" else eval_all
        sub = src[(src["phase"] == ekey) & (src["season"] >= 2025)].copy()
        if sub.empty:
            continue

        def score(rr):
            s = int(rr["season"])
            bp = bat.get(f"{rr['batting_team']}|{s}", default_par)
            wp = bowl.get(f"{rr['bowling_team']}|{s}", default_par)
            vp = ven.get(f"{rr['venue']}|{s}", default_par)
            X = int(round(rr["line_t_minus_1"]))
            d = {"threshold_X": float(X), "bat_prior": bp, "bowl_prior": wp, "venue_par": vp,
                 "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp, "innings": 1}
            if mode == "trend":
                lt = trend.get(str(s), default_par)
                d["league_trend"] = lt
                d["x_minus_trend"] = X - lt
            else:
                cl = venue_climo.get(rr["venue"], global_climo)
                for af, rv in ANOM_RAW.items():
                    raw = rr.get(rv)
                    raw = medians[rv] if (raw is None or pd.isna(raw)) else raw
                    d[af] = float(raw) - float(cl.get(rv, global_climo[rv]))
                pr = rr.get("precip_mm")
                d["precip_mm"] = medians["precip_mm"] if (pr is None or pd.isna(pr)) else float(pr)
            for L in LEAGUES:
                d[f"is_{L}"] = 1.0 if rr["league"] == L else 0.0
            xv = np.array([[d.get(f, 0.0) for f in feats]], dtype=np.float32)
            return float(booster.predict(xv)[0])

        sub["p"] = sub.apply(score, axis=1)
        sub["actual_over"] = (sub["actual_total"] > sub["line_t_minus_1"]).astype(int)
        sub["actual_under"] = (sub["actual_total"] < sub["line_t_minus_1"]).astype(int)
        results.append((plabel, bt(sub, "p", STAKE, False), bt(sub, "p", STAKE, True)))

    # ---- inn2 phases (target-aware; trained <=2023, eval 2024+) ----
    inn1_totals = balls[balls["innings"] == 1].groupby("match_id")["runs_total"].sum().astype(int).to_dict()

    def inn2_outcome(g, tb, inn1_total):
        g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
        li = np.where(g["is_legal_delivery"].values)[0]
        if len(li) == 0:
            return None
        if len(li) >= tb:
            return int(g.iloc[:li[tb-1]+1]["runs_total"].sum())
        wk = int(g["wicket"].fillna(False).astype(bool).sum())
        ft = int(g["runs_total"].sum())
        return ft if (wk >= 10 or ft > inn1_total) else None

    eval_inn2 = pd.read_parquet("data/processed/eval_phase_lines_inn2.parquet")
    INN2_CUT = 2023
    for plabel, tb, xmn, xmx, xst in [("phase_6", 36, 20, 110, 5), ("phase_10", 60, 40, 175, 5)]:
        # build inn2 data
        rows_d = []
        for (mid, inn), g in balls.groupby(["match_id", "innings"]):
            if int(inn) != 2:
                continue
            out = inn2_outcome(g, tb, inn1_totals.get(mid, 0))
            if out is None:
                continue
            tgt = g["target"].dropna()
            if tgt.empty:
                continue
            rows_d.append({"match_id": mid, "season": int(g["season"].iloc[0]),
                           "batting_team": g["batting_team"].iloc[0], "bowling_team": g["bowling_team"].iloc[0],
                           "venue": g["venue"].iloc[0], "target": float(tgt.iloc[0]), "phase_total": out})
        dd = pd.DataFrame(rows_d)
        bat, bowl, ven, trend = {}, {}, {}, {}
        for s in sorted(dd["season"].unique()):
            prior = dd[dd["season"] < s]
            if prior.empty:
                continue
            w = np.power(0.5, (s - prior["season"]) / HALF_LIFE)
            prior = prior.assign(_w=w)
            for col, tbl in [("batting_team", bat), ("bowling_team", bowl), ("venue", ven)]:
                for key, gg in prior.groupby(col):
                    tbl[f"{key}|{int(s)}"] = float(np.average(gg["phase_total"], weights=gg["_w"]))
            prev = dd[dd["season"] == s - 1]
            if not prev.empty:
                trend[str(int(s))] = float(prev["phase_total"].mean())
        default_par = float(dd["phase_total"].mean())
        tr_rows = []
        for r in dd[dd["season"] <= INN2_CUT].itertuples(index=False):
            s = int(r.season)
            bp = bat.get(f"{r.batting_team}|{s}", default_par); wp = bowl.get(f"{r.bowling_team}|{s}", default_par)
            vp = ven.get(f"{r.venue}|{s}", default_par); lt = trend.get(str(s), default_par)
            league = league_by_match.get(str(r.match_id), "unknown")
            ppt = float(r.target) * tb / 120.0
            for X in range(xmn, xmx + 1, xst):
                tr_rows.append({"season": s, "league": league, "threshold_X": X, "bat_prior": bp,
                    "bowl_prior": wp, "venue_par": vp, "target": float(r.target), "phase_par_from_target": ppt,
                    "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp,
                    "x_minus_target_par": X - ppt, "league_trend": lt, "x_minus_trend": X - lt,
                    "innings": 2, "actual_over_X": int(r.phase_total >= X)})
        tr = pd.DataFrame(tr_rows)
        for L in LEAGUES:
            tr[f"is_{L}"] = (tr["league"] == L).astype(int)
        feats = ["threshold_X", "bat_prior", "bowl_prior", "venue_par", "target", "phase_par_from_target",
                 "x_minus_par", "x_minus_bat", "x_minus_bowl", "x_minus_target_par",
                 "league_trend", "x_minus_trend", "innings"] + [f"is_{L}" for L in LEAGUES]
        booster = train_booster(tr, feats)

        sub = eval_inn2[(eval_inn2["phase"] == plabel) & (eval_inn2["season"] > INN2_CUT)].copy()
        if sub.empty:
            continue

        def score2(rr):
            s = int(rr["season"])
            bp = bat.get(f"{rr['batting_team']}|{s}", default_par); wp = bowl.get(f"{rr['bowling_team']}|{s}", default_par)
            vp = ven.get(f"{rr['venue']}|{s}", default_par); lt = trend.get(str(s), default_par)
            X = int(round(rr["line_t_minus_1"])); ppt = float(rr["target"]) * tb / 120.0
            d = {"threshold_X": float(X), "bat_prior": bp, "bowl_prior": wp, "venue_par": vp,
                 "target": float(rr["target"]), "phase_par_from_target": ppt,
                 "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp,
                 "x_minus_target_par": X - ppt, "league_trend": lt, "x_minus_trend": X - lt, "innings": 2}
            for L in LEAGUES:
                d[f"is_{L}"] = 1.0 if rr["league"] == L else 0.0
            xv = np.array([[d.get(f, 0.0) for f in feats]], dtype=np.float32)
            return float(booster.predict(xv)[0])

        sub["p"] = sub.apply(score2, axis=1)
        sub["actual_over"] = (sub["actual_total"] > sub["line_t_minus_1"]).astype(int)
        sub["actual_under"] = (sub["actual_total"] < sub["line_t_minus_1"]).astype(int)
        results.append((f"{plabel}_inn2", bt(sub, "p", STAKE, False), bt(sub, "p", STAKE, True)))

    print(f"\n===== PRODUCTION-CONFIG OOS BACKTEST (train <=2024, eval 2025+) =====")
    print(f"£{STAKE:.0f} stake, odds {ODDS}, {COMMISSION:.0%} commission")
    print(f"\n{'phase':<14} {'FULL: n':>8} {'P&L':>9} {'ROI':>8} {'win':>6} | {'MID: n':>8} {'P&L':>9} {'ROI':>8} {'win':>6}")
    print("-" * 92)
    tot_full = tot_mid = 0.0
    for plabel, full, mid in results:
        fs = f"{full['n']:>8} £{full['pnl']:>+7.0f} {full['roi']:>+7.1%} {full['win']:>5.0%}" if full else f"{'-':>8} {'-':>9} {'-':>8} {'-':>6}"
        ms = f"{mid['n']:>8} £{mid['pnl']:>+7.0f} {mid['roi']:>+7.1%} {mid['win']:>5.0%}" if mid else f"{'-':>8} {'-':>9} {'-':>8} {'-':>6}"
        print(f"{plabel:<14} {fs} | {ms}")
        if full: tot_full += full["pnl"]
        if mid: tot_mid += mid["pnl"]
    print("-" * 92)
    print(f"{'TOTAL':<14} {'':>8} £{tot_full:>+7.0f} {'':>8} {'':>6} | {'':>8} £{tot_mid:>+7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
