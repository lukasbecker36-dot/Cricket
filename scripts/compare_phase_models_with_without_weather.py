"""Compare phase models v1 (no weather) vs v2 (with weather) on the same
historical Line market backtest. Re-uses extracted line-market data from
data/processed/eval_phase_lines.parquet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.live.signals import load_models
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def load_model_pair(prefix_v1: str, prefix_v2: str, model_dir: Path):
    # Just reuse FullInningsModel since the schema only differs by the additional
    # weather features being present in v2's `features` list.
    from src.live.signals import FullInningsModel
    return FullInningsModel(model_dir, prefix=prefix_v1), FullInningsModel(model_dir, prefix=prefix_v2)


def main() -> int:
    configure_logging()

    eval_df = pd.read_parquet("data/processed/eval_phase_lines.parquet")
    weather = pd.read_parquet("data/processed/match_weather.parquet")

    # Join weather on (venue, date) -- need match dates
    # eval_df doesn't carry match date; pull from balls
    LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]
    md = {}
    for league in LEAGUES:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        m = pd.read_parquet(d / "matches.parquet")
        for _, r in m.iterrows():
            md[str(r["match_id"])] = str(r["date"])[:10]
    eval_df["date"] = eval_df["match_id"].astype(str).map(md)
    eval_df = eval_df.merge(weather, on=["venue", "date"], how="left")

    coverage = eval_df["temp_c"].notna().mean()
    logger.info("Weather coverage on eval set: %.1f%%", coverage * 100)

    LINE_ODDS, IMPLIED, EDGE = 2.0, 0.5, 0.05

    model_dir = Path("models")
    results = {}
    for phase_short, phase_v1, phase_v2 in [
        ("phase_6",  "phase_6",  "phase_6_wx"),
        ("phase_10", "phase_10", "phase_10_wx"),
        ("phase_15", "phase_15", "phase_15_wx"),
    ]:
        if not (model_dir / f"{phase_v2}_meta.json").exists():
            logger.warning("v2 model %s not trained yet, skipping", phase_v2)
            continue
        m_v1, m_v2 = load_model_pair(phase_v1, phase_v2, model_dir)
        sub = eval_df[eval_df["phase"] == phase_short].copy()
        if sub.empty:
            continue
        # v1 predictions (no weather)
        sub["p_v1"] = sub.apply(lambda r: m_v1.predict_p(
            threshold_X=int(round(r["line_t_minus_1"])), implied_open=IMPLIED,
            batting_team=r["batting_team"], bowling_team=r["bowling_team"],
            venue=r["venue"], season=int(r["season"]), innings=int(r["innings"]),
            league=r["league"],
        ), axis=1)
        # v2 predictions (with weather)
        # FullInningsModel.predict_p doesn't accept weather kwargs; bypass by
        # building feature vector manually if weather features are in meta
        v2_meta_features = m_v2.features
        weather_feats = [f for f in v2_meta_features if f in ("temp_c","humidity_pct","wind_kph","precip_mm","cloud_pct")]
        def predict_v2(r):
            bat_prior = m_v2._lookup(m_v2.bat_pp, r["batting_team"], int(r["season"]))
            bowl_prior = m_v2._lookup(m_v2.bowl_pp, r["bowling_team"], int(r["season"]))
            v_par = m_v2._lookup(m_v2.venue_par, r["venue"], int(r["season"]))
            X = int(round(r["line_t_minus_1"]))
            row = {
                "threshold_X": float(X), "implied_open": float(IMPLIED),
                "bat_prior": bat_prior, "bowl_prior": bowl_prior, "venue_par": v_par,
                "x_minus_par": float(X) - v_par,
                "x_minus_bat": float(X) - bat_prior,
                "x_minus_bowl": float(X) - bowl_prior,
                "innings": int(r["innings"]),
                "temp_c": r.get("temp_c"), "humidity_pct": r.get("humidity_pct"),
                "wind_kph": r.get("wind_kph"), "precip_mm": r.get("precip_mm"),
                "cloud_pct": r.get("cloud_pct"),
            }
            # impute missing weather with meta medians
            medians = m_v2.meta.get("weather_medians", {})
            for f in weather_feats:
                if row.get(f) is None or pd.isna(row.get(f)):
                    row[f] = medians.get(f, 0.0)
            for L in m_v2.leagues:
                row[f"is_{L}"] = 1.0 if r["league"] == L else 0.0
            x = np.array([[row.get(f, 0.0) for f in v2_meta_features]], dtype=np.float32)
            return float(m_v2.booster.predict(x)[0])

        sub["p_v2"] = sub.apply(predict_v2, axis=1)
        sub["actual_over"] = (sub["actual_total"] > sub["line_t_minus_1"]).astype(int)
        sub["actual_under"] = (sub["actual_total"] < sub["line_t_minus_1"]).astype(int)

        def backtest_col(df_in, prob_col):
            signal = np.where(df_in[prob_col] >= 0.5 + EDGE, "over",
                              np.where(df_in[prob_col] <= 0.5 - EDGE, "under", "skip"))
            active_mask = signal != "skip"
            active = df_in[active_mask].copy()
            sigs = signal[active_mask]
            if active.empty:
                return None
            won = np.where(sigs == "over", active["actual_over"] == 1, active["actual_under"] == 1)
            pnl = np.where(won, 100 * (LINE_ODDS - 1) * 0.95, -100.0)
            return {"n": int(len(active)), "pnl": float(pnl.sum()),
                    "roi": float(pnl.sum() / (len(active) * 100)),
                    "win_rate": float(np.mean(won))}

        v1 = backtest_col(sub, "p_v1")
        v2 = backtest_col(sub, "p_v2")
        results[phase_short] = (v1, v2)

        # mid-confidence band only
        sub_mid = sub[(sub["p_v2"] - 0.5).abs().between(0.05, 0.30)].copy()
        v2_mid = backtest_col(sub_mid, "p_v2")
        sub_mid_v1 = sub[(sub["p_v1"] - 0.5).abs().between(0.05, 0.30)].copy()
        v1_mid = backtest_col(sub_mid_v1, "p_v1")
        results[phase_short + "_mid"] = (v1_mid, v2_mid)

    print(f"\n{'phase':<12} {'v1 n':>6} {'v1 ROI':>8} {'v1 win':>8} | {'v2 n':>6} {'v2 ROI':>8} {'v2 win':>8}")
    print("-" * 70)
    for phase, (v1, v2) in results.items():
        v1s = f"{v1['n']:>6} {v1['roi']:>+7.2%} {v1['win_rate']:>+7.1%}" if v1 else "       -        -        -"
        v2s = f"{v2['n']:>6} {v2['roi']:>+7.2%} {v2['win_rate']:>+7.1%}" if v2 else "       -        -        -"
        print(f"{phase:<12} {v1s} | {v2s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
