"""Calibration audit for the phase models.

For each phase: train on seasons <=2024 (production trend config), predict on
2025+ Line markets, and check whether model_p is honest — i.e. when it says
65%, does the over happen ~65% of the time?

Outputs per phase + pooled:
  - reliability table (predicted vs observed over-rate by bin)
  - ECE (expected calibration error) and Brier score
  - post-Platt-scaling ECE via 5-fold CV (does recalibration help?)
  - a reliability-diagram PNG (data/processed/calibration_reliability.png)

Uses the trend feature config (production for 3 of 4 phases); phase_6/15 also
carry player/weather features in production but the calibration *character*
(over/under-confidence) is a property of the GBM+data and is well-represented
here. Small OOS samples => wide bin error bars; pooled read is most robust.
"""
from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.ingestion.storage import read_balls
from src.ingestion.teams import canonical_team
from src.ingestion.venues import canonical_venue
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]
HALF_LIFE, MAX_TRAIN = 2.0, 2024
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


def ece(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    e, rows = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        pred, obs, cnt = p[m].mean(), y[m].mean(), m.sum()
        e += (cnt / len(p)) * abs(pred - obs)
        rows.append((bins[b], bins[b + 1], cnt, pred, obs))
    return e, rows


def platt_cv_ece(p, y):
    """5-fold CV Platt scaling -> honest post-recalibration ECE."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    oof = np.zeros_like(p)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    for tr, te in skf.split(logit, y):
        lr = LogisticRegression()
        lr.fit(logit[tr], y[tr])
        oof[te] = lr.predict_proba(logit[te])[:, 1]
    return ece(oof, y)[0]


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
    eval_all = pd.read_parquet("data/processed/eval_phase_lines.parquet")
    eval_full = pd.read_parquet("data/processed/eval_line_combined.parquet")
    eval_full = eval_full[eval_full["innings"] == 1].copy(); eval_full["phase"] = "full_innings"

    pooled_p, pooled_y = [], []
    per_phase = {}
    for plabel, ekey, tb, xmn, xmx, xst in PHASES:
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
            league = league_by_match.get(r.match_id, "unknown")
            for X in range(xmn, xmx + 1, xst):
                tr_rows.append({"season": s, "league": league, "threshold_X": X, "bat_prior": bp,
                    "bowl_prior": wp, "venue_par": vp, "x_minus_par": X - vp, "x_minus_bat": X - bp,
                    "x_minus_bowl": X - wp, "innings": 1, "league_trend": lt, "x_minus_trend": X - lt,
                    "actual_over_X": int(r.total >= X)})
        tr = pd.DataFrame(tr_rows)
        for L in LEAGUES: tr[f"is_{L}"] = (tr["league"] == L).astype(int)
        feats = ["threshold_X", "bat_prior", "bowl_prior", "venue_par", "x_minus_par", "x_minus_bat",
                 "x_minus_bowl", "innings", "league_trend", "x_minus_trend"] + [f"is_{L}" for L in LEAGUES]
        booster = train_booster(tr, feats)

        src = eval_full if ekey == "full_innings" else eval_all
        sub = src[(src["phase"] == ekey) & (src["season"] >= 2025)].copy()
        if sub.empty: continue
        def score(rr):
            s = int(rr["season"]); X = int(round(rr["line_t_minus_1"]))
            bp = bat.get(f"{canonical_team(rr['batting_team'])}|{s}", default_par)
            wp = bowl.get(f"{canonical_team(rr['bowling_team'])}|{s}", default_par)
            vp = ven.get(f"{canonical_venue(rr['venue'])}|{s}", default_par); lt = trend.get(str(s), default_par)
            d = {"threshold_X": float(X), "bat_prior": bp, "bowl_prior": wp, "venue_par": vp,
                 "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp, "innings": 1,
                 "league_trend": lt, "x_minus_trend": X - lt}
            for L in LEAGUES: d[f"is_{L}"] = 1.0 if rr["league"] == L else 0.0
            return float(booster.predict(np.array([[d.get(f, 0.0) for f in feats]], dtype=np.float32))[0])
        p = sub.apply(score, axis=1).to_numpy()
        y = (sub["actual_total"] > sub["line_t_minus_1"]).astype(int).to_numpy()
        per_phase[plabel] = (p, y)
        pooled_p.append(p); pooled_y.append(y)

    pooled_p = np.concatenate(pooled_p); pooled_y = np.concatenate(pooled_y)

    print("\n===== CALIBRATION AUDIT (2025+ OOS, trained <=2024) =====")
    for label, (p, y) in list(per_phase.items()) + [("POOLED", (pooled_p, pooled_y))]:
        e, _ = ece(p, y, n_bins=5)
        brier = float(np.mean((p - y) ** 2))
        base = float(y.mean())
        post = platt_cv_ece(p, y) if len(p) >= 30 else float("nan")
        print(f"\n{label}: n={len(p)}  base over-rate={base:.2f}  Brier={brier:.3f}")
        print(f"  ECE raw={e:.3f}   ECE after Platt (5-fold CV)={post:.3f}")

    # pooled reliability table
    print("\n=== POOLED reliability (5 bins) ===")
    _, rows = ece(pooled_p, pooled_y, n_bins=5)
    print(f"  {'bin':<12} {'n':>5} {'pred':>7} {'obs':>7}")
    for lo, hi, cnt, pred, obs in rows:
        print(f"  {f'{lo:.1f}-{hi:.1f}':<12} {cnt:>5} {pred:>7.3f} {obs:>7.3f}")

    # reliability diagram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="perfect")
        for label, (p, y) in per_phase.items():
            _, rws = ece(p, y, n_bins=5)
            xs = [(lo + hi) / 2 for lo, hi, *_ in rws]
            ys = [obs for *_, obs in rws]
            ax.plot(xs, ys, "o-", alpha=0.6, label=label)
        _, rws = ece(pooled_p, pooled_y, n_bins=5)
        ax.plot([(lo + hi) / 2 for lo, hi, *_ in rws], [obs for *_, obs in rws],
                "s-", color="black", lw=2, label="POOLED")
        ax.set_xlabel("predicted P(over)"); ax.set_ylabel("observed over-rate")
        ax.set_title("Reliability — phase models, 2025+ OOS"); ax.legend(); ax.grid(alpha=0.3)
        out = Path("data/processed/calibration_reliability.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"\nsaved reliability diagram: {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
