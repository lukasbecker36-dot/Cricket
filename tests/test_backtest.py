import numpy as np
import pandas as pd

from src.config import BacktestConfig
from src.validation.backtest import apply_slippage, betfair_tick_size, prob_to_price, run_backtest


def test_tick_sizes_match_betfair_ladder():
    assert betfair_tick_size(1.5) == 0.01
    assert betfair_tick_size(2.5) == 0.02
    assert betfair_tick_size(5.5) == 0.1
    assert betfair_tick_size(15.0) == 0.5
    assert betfair_tick_size(200.0) == 10.0


def test_slippage_worsens_back_price_down():
    assert apply_slippage(3.0, "back", 1) < 3.0


def test_slippage_worsens_lay_price_up():
    assert apply_slippage(3.0, "lay", 1) > 3.0


def test_prob_to_price_inverse():
    assert abs(prob_to_price(0.5) - 2.0) < 1e-9


def test_backtest_with_no_edge_makes_no_trades():
    df = pd.DataFrame({
        "match_id": ["m"] * 5,
        "ball_index": list(range(5)),
        "p": [0.5] * 5,
        "y": [1, 0, 1, 0, 1],
    })
    cfg = BacktestConfig(edge_threshold=0.05)
    result = run_backtest(
        df, cfg, rng=np.random.default_rng(0),
        market_implied=pd.Series([0.5] * 5),
    )
    assert result.n_trades == 0
    assert result.total_pnl == 0.0


def test_commission_reduces_winning_pnl():
    df = pd.DataFrame({
        "match_id": ["m"],
        "ball_index": [0],
        "p": [0.9],
        "y": [1],  # back wins
    })
    cfg = BacktestConfig(edge_threshold=0.01, commission=0.05, slippage_ticks=0)
    result = run_backtest(
        df, cfg, stake=100.0, market_implied=pd.Series([0.5]),
    )
    # back at price 2.0 with stake 100 -> gross 100, commission 5, net 95
    assert result.n_trades == 1
    assert abs(result.trades["pnl"].iloc[0] - 95.0) < 1e-6
