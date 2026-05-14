from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.indicators import (
    IndicatorFactory,
    MarketMicrostructureIndicators,
    TimeEncoding,
    VolatilityIndicators,
    _rolling,
)


def test_rolling_defaults_min_periods_to_window_size() -> None:
    series = pd.Series([1.0, 2.0, 3.0])
    rolled = _rolling(series, window=2).mean()
    assert rolled.isna().tolist() == [True, False, False]
    np.testing.assert_allclose(rolled.dropna().to_numpy(), [1.5, 2.5])


def test_returns_adds_return_columns(ohlcv_df: pd.DataFrame) -> None:
    out = VolatilityIndicators.returns(ohlcv_df, close_col="Close", windows=[2])
    assert {"return", "return_2"}.issubset(out.columns)
    expected_second = pd.Series(ohlcv_df["Close"]).pct_change().iloc[1:3].mean()
    assert out["return_2"].iloc[2] == pytest.approx(expected_second)


def test_rolling_volatility_requires_bars_per_year_when_annualized(ohlcv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="bars_per_year"):
        VolatilityIndicators.rolling_volatility(ohlcv_df, close_col="Close", window=3, annualize=True)


def test_time_encoding_adds_cyclical_columns(ohlcv_df: pd.DataFrame) -> None:
    out = TimeEncoding.encode_intraday_cyclical(TimeEncoding.encode_weekday_cyclical(ohlcv_df))
    assert {"dow_sin", "dow_cos", "tod_sin", "tod_cos"}.issubset(out.columns)
    assert out[["dow_sin", "dow_cos", "tod_sin", "tod_cos"]].abs().max().max() <= 1.0


def test_lobster_l1_flow_derives_mid_spread_and_default_exec_columns(lobster_df: pd.DataFrame) -> None:
    df = lobster_df.drop(columns=["mid", "spread", "exec_shares", "exec_signed_shares"])
    out = MarketMicrostructureIndicators.lobster_l1_flow(df, windows=[2])
    assert {"mid", "spread", "depth_l1", "l1_imb", "flow_imb_1"}.issubset(out.columns)
    assert out["exec_shares"].eq(0.0).all()
    assert out["exec_signed_shares"].eq(0.0).all()


def test_apply_microstructure_adds_lobster_features(lobster_df: pd.DataFrame) -> None:
    out = IndicatorFactory.apply_microstructure(lobster_df.copy(), levels=2)
    assert {"depth_l1", "microprice", "l1_ofi", "ewm_logret_hl10"}.issubset(out.columns)


def test_apply_microstructure_is_safe_with_minimal_input() -> None:
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
    out = IndicatorFactory.apply_microstructure(df)
    assert len(out) == len(df)
    assert "Close" in out.columns


def test_apply_for_direction_adds_bollinger_percent_b(ohlcv_df: pd.DataFrame) -> None:
    out = IndicatorFactory.apply_for_direction(ohlcv_df)
    assert {"BB_MA20", "BB_UP20", "BB_LW20", "BB_percent_b"}.issubset(out.columns)
    assert out["BB_percent_b"].dropna().iloc[-1] > 0


def test_apply_for_volatility_includes_documented_bb_width20(ohlcv_df: pd.DataFrame) -> None:
    out = IndicatorFactory.apply_for_volatility(ohlcv_df)
    assert "BB_width20" in out.columns
