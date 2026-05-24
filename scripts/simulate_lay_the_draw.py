"""Lay-the-draw simulation with capped liability + edge filter.

Pulls data from data/processed/test_draws.parquet. Tests three strategies:
  1. Naive: lay £10 flat on every market
  2. Capped: stake sized to keep liability <= MAX_LIABILITY
  3. Filtered + capped: only lay when implied draw >= MIN_IMPLIED
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAX_LIABILITY = 25.0       # £ at risk per trade
MIN_IMPLIED = 0.15         # only lay markets where draw implied >= 15%
COMMISSION = 0.05
MAX_STAKE = 50.0           # cap stake even if price is very high


def simulate(df: pd.DataFrame, stake_func, label: str) -> dict:
    pnl_total = 0.0
    capital_total = 0.0
    n = 0
    wins = 0
    for _, r in df.iterrows():
        price = r["draw_price_t60"]
        if price is None or price <= 1.0:
            continue
        stake = stake_func(price)
        if stake <= 0:
            continue
        liability = stake * (price - 1.0)
        capital = stake + liability
        is_draw_side = r["outcome"] in ("draw", "tie")
        if r["outcome"] == "no_result":
            pnl = 0.0
        elif is_draw_side:
            pnl = -liability
        else:
            pnl = stake * (1.0 - COMMISSION)
            wins += 1
        pnl_total += pnl
        capital_total += capital
        n += 1
    roi_capital = pnl_total / capital_total if capital_total else 0
    return {
        "label": label, "n": n, "wins": wins,
        "pnl": pnl_total, "capital": capital_total,
        "roi_capital": roi_capital,
    }


def main() -> int:
    df = pd.read_parquet("data/processed/test_draws.parquet")
    df = df.drop_duplicates("match_id_cricsheet").copy()
    df["draw_implied"] = 1.0 / df["draw_price_t60"]

    STAKE_FLAT = 10.0

    print("=" * 70)
    print("STRATEGY COMPARISON — 137 tests, 2022-2026")
    print("=" * 70)

    # Strategy 1: naive flat £10
    s1 = simulate(df, lambda price: STAKE_FLAT, "1. Naive flat £10")

    # Strategy 2: capped liability, no filter
    def capped(price):
        if price <= 1.0:
            return 0
        return min(MAX_STAKE, MAX_LIABILITY / (price - 1.0))
    s2 = simulate(df, capped, f"2. Liability cap £{MAX_LIABILITY:.0f}")

    # Strategy 3: capped + filter to implied >= MIN_IMPLIED
    filtered = df[df["draw_implied"] >= MIN_IMPLIED].copy()
    s3 = simulate(filtered, capped, f"3. Capped + filter implied >= {MIN_IMPLIED:.0%}")

    # Strategy 4: capped + filter to mid band (e.g., 10% to 30% implied)
    mid = df[(df["draw_implied"] >= 0.10) & (df["draw_implied"] <= 0.30)].copy()
    s4 = simulate(mid, capped, "4. Capped + implied in [10%, 30%]")

    # Strategy 5: capped + low-implied only (lay extreme outsiders for high-stakes)
    low = df[df["draw_implied"] < 0.10].copy()
    s5 = simulate(low, capped, "5. Capped + implied < 10% only")

    print(f"\n{'Strategy':<48} {'n':>4} {'wins':>5} {'P&L':>8} {'Cap':>9} {'ROI cap':>9}")
    print("-" * 90)
    for s in (s1, s2, s3, s4, s5):
        print(f"{s['label']:<48} {s['n']:>4} {s['wins']:>5} £{s['pnl']:>+7.0f} £{s['capital']:>7.0f} {s['roi_capital']:>+8.2%}")

    print("\n=== Detailed: Strategy 2 (capped liability, no filter) by year ===")
    df["stake"] = df["draw_price_t60"].apply(capped)
    df["liability"] = df["stake"] * (df["draw_price_t60"] - 1.0)
    df["capital"] = df["stake"] + df["liability"]
    df["is_draw_side"] = df["outcome"].isin(["draw", "tie"])
    df["pnl"] = np.where(df["outcome"] == "no_result", 0.0,
                np.where(df["is_draw_side"], -df["liability"], df["stake"] * (1.0 - COMMISSION)))
    df["year"] = df["event_date"].str[:4]
    print(df.groupby("year").agg(
        n=("pnl", "size"),
        draws=("is_draw_side", "sum"),
        pnl=("pnl", "sum"),
        capital=("capital", "sum"),
    ).round(2).to_string())

    print("\n=== Strategy 2 by market-implied bucket ===")
    df["bucket"] = pd.qcut(df["draw_implied"], q=5, duplicates="drop")
    bt = df.groupby("bucket", observed=True).agg(
        n=("pnl", "size"),
        market_p=("draw_implied", "mean"),
        actual_rate=("is_draw_side", "mean"),
        pnl=("pnl", "sum"),
        capital=("capital", "sum"),
        avg_stake=("stake", "mean"),
    ).round(3)
    bt["roi_cap"] = (bt["pnl"] / bt["capital"]).round(3)
    print(bt.to_string())

    print("\n=== Drawdown test for naive vs capped (sequential P&L) ===")
    for label, stake_fn in [("Naive £10", lambda p: STAKE_FLAT), (f"Capped £{MAX_LIABILITY:.0f}", capped)]:
        df["stake_x"] = df["draw_price_t60"].apply(stake_fn)
        df["liab_x"] = df["stake_x"] * (df["draw_price_t60"] - 1.0)
        df["pnl_x"] = np.where(df["outcome"] == "no_result", 0.0,
                      np.where(df["is_draw_side"], -df["liab_x"], df["stake_x"] * 0.95))
        df_seq = df.sort_values("event_date").reset_index(drop=True)
        df_seq["cum"] = df_seq["pnl_x"].cumsum()
        df_seq["peak"] = df_seq["cum"].cummax()
        df_seq["dd"] = df_seq["cum"] - df_seq["peak"]
        max_dd = df_seq["dd"].min()
        print(f"  {label}: max drawdown £{max_dd:.2f}, final P&L £{df_seq['cum'].iloc[-1]:+.2f}")

    # Three concrete sizing examples for ENG-NZ
    print("\n=== Sizing examples for likely ENG-NZ draw prices ===")
    for price in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0]:
        stake = capped(price)
        liability = stake * (price - 1)
        implied = 1.0 / price
        print(f"  draw at {price:.1f}  (implied {implied:.1%})  -> stake £{stake:.2f}, liability £{liability:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
