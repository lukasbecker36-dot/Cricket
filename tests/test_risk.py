import pytest

from src.execution.risk import RiskLimits, RiskRejection, check_trade


def base_kwargs():
    return dict(
        model_p=0.6, stake=50.0, exposure_in_match=0.0, daily_loss=0.0,
        balls_remaining=60, staleness_seconds=2.0,
    )


def test_happy_path_passes():
    check_trade(**base_kwargs())


def test_probability_out_of_band_rejected():
    kw = base_kwargs() | {"model_p": 0.005}
    with pytest.raises(RiskRejection):
        check_trade(**kw)


def test_stale_data_rejected():
    kw = base_kwargs() | {"staleness_seconds": 60.0}
    with pytest.raises(RiskRejection):
        check_trade(**kw)


def test_too_few_balls_remaining_rejected():
    kw = base_kwargs() | {"balls_remaining": 3}
    with pytest.raises(RiskRejection):
        check_trade(**kw)


def test_stake_above_limit_rejected():
    kw = base_kwargs() | {"stake": 10_000}
    with pytest.raises(RiskRejection):
        check_trade(**kw)


def test_per_match_exposure_rejected():
    kw = base_kwargs() | {"exposure_in_match": 999.0, "stake": 50.0}
    limits = RiskLimits(max_exposure_per_match=1000.0)
    with pytest.raises(RiskRejection):
        check_trade(**kw, limits=limits)


def test_daily_loss_kills_switch():
    kw = base_kwargs() | {"daily_loss": 1_000.0}
    with pytest.raises(RiskRejection):
        check_trade(**kw)


def test_none_staleness_rejected():
    kw = base_kwargs() | {"staleness_seconds": None}
    with pytest.raises(RiskRejection):
        check_trade(**kw)
