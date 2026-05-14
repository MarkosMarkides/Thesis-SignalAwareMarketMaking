import numpy as np
import pandas as pd
import pytest

from technical.labelling import DirectionLabelerFutureAvg, VolatilityLabeler, label_direction_future_avg, label_volatility


def test_volatility_matches_hand_computed_forward_realized_volatility():
    prices = [100.0, 101.0, 100.0, 102.0, 103.0]
    df = pd.DataFrame({"mid": prices, "kept": [10, 20, 30, 40, 50]})

    out = label_volatility(df, close_col="mid", horizon=2, out_col="y_vol")

    log_returns = np.log(pd.Series(prices)).diff()
    expected = [
        np.sqrt(log_returns.iloc[1:3].pow(2).sum()),
        np.sqrt(log_returns.iloc[2:4].pow(2).sum()),
        np.sqrt(log_returns.iloc[3:5].pow(2).sum()),
        0.0,
        0.0,
    ]

    assert out["y_vol"].to_numpy() == pytest.approx(expected)
    assert out["kept"].tolist() == [10, 20, 30, 40, 50]


def test_volatility_custom_columns_and_class_wrapper_work():
    df = pd.DataFrame({"close_price": [10.0, 11.0, 12.0, 13.0]})
    direct = label_volatility(df, close_col="close_price", horizon=1, out_col="vol")
    wrapped = VolatilityLabeler(close_col="close_price", horizon=1, out_col="vol").label(df)

    assert "vol" in direct.columns
    assert direct["vol"].tolist() == wrapped["vol"].tolist()
    assert direct["vol"].iloc[-1] == 0.0


def test_direction_fixed_threshold_and_tail_drop():
    df = pd.DataFrame({"mid": [100.0, 101.0, 99.0, 102.0, 98.0, 103.0]})

    out = label_direction_future_avg(df, price_col="mid", horizon=1, threshold=0.0, verbose=False)

    assert out["mid"].tolist() == [100.0, 101.0, 99.0, 102.0, 98.0]
    assert out["y_dir"].tolist() == [1, 0, 1, 0, 1]


def test_direction_balanced_labels_are_deterministic_on_known_series():
    df = pd.DataFrame({"mid": [100.0, 101.0, 99.0, 102.0, 98.0, 103.0]})

    out = label_direction_future_avg(df, price_col="mid", horizon=1, balanced=True, verbose=False)

    assert out["y_dir"].tolist() == [2, 0, 1, 0, 1]


def test_direction_vwap_can_differ_from_simple_mean():
    df = pd.DataFrame({"mid": [100.0, 90.0, 110.0, 100.0], "volume": [1.0, 1.0, 100.0, 1.0]})

    vwap = label_direction_future_avg(
        df, price_col="mid", vol_col="volume", horizon=2, avg_mode="vwap", verbose=False
    )
    mean = label_direction_future_avg(df, price_col="mid", horizon=2, avg_mode="mean", verbose=False)

    assert vwap["y_dir"].tolist() == [1, 1]
    assert mean["y_dir"].tolist() == [2, 1]


def test_vwap_falls_back_to_mean_when_volume_column_is_missing():
    df = pd.DataFrame({"mid": [100.0, 90.0, 110.0, 100.0]})

    fallback = label_direction_future_avg(
        df, price_col="mid", vol_col="missing_volume", horizon=2, avg_mode="vwap", verbose=False
    )
    mean = label_direction_future_avg(df, price_col="mid", horizon=2, avg_mode="mean", verbose=False)

    assert fallback["y_dir"].tolist() == mean["y_dir"].tolist()


def test_direction_wrapper_default_uses_mean_not_vwap():
    df = pd.DataFrame({"mid": [100.0, 90.0, 110.0, 100.0], "volume": [1.0, 1.0, 100.0, 1.0]})

    default = label_direction_future_avg(
        df, price_col="mid", vol_col="volume", horizon=2, verbose=False
    )
    explicit_mean = label_direction_future_avg(
        df, price_col="mid", vol_col="volume", horizon=2, avg_mode="mean", verbose=False
    )
    class_default = DirectionLabelerFutureAvg(
        price_col="mid", vol_col="volume", horizon=2, avg_mode="vwap", verbose=False
    ).label(df)

    assert default["y_dir"].tolist() == explicit_mean["y_dir"].tolist()
    assert class_default["y_dir_avg"].tolist() != default["y_dir"].tolist()
