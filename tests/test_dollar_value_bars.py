from __future__ import annotations

import pandas as pd

from feature_engineering.dollar_value_bars import AdaptiveDollarBarsGenerator


def test_generator_carry_forwards_previous_close_into_next_bar_open() -> None:
    df = pd.DataFrame(
        {
            "Open time": pd.date_range("2024-01-01", periods=3, freq="min"),
            "Open": [10.0, 12.0, 13.0],
            "High": [11.0, 13.0, 14.0],
            "Low": [9.0, 11.0, 12.0],
            "Close": [10.0, 12.0, 13.0],
            "Volume": [10.0, 10.0, 10.0],
            "Quote asset volume": [100.0, 120.0, 130.0],
            "Number of trades": [1.0, 2.0, 3.0],
            "Taker buy base asset volume": [4.0, 5.0, 6.0],
            "Taker buy quote asset volume": [40.0, 50.0, 60.0],
        }
    )
    out = AdaptiveDollarBarsGenerator(lookback=1, k=1.0).generate_bars(df)
    assert len(out) == 3
    assert out.iloc[1]["Open"] == out.iloc[0]["Close"]


def test_generator_aggregates_extra_sum_and_last_columns() -> None:
    df = pd.DataFrame(
        {
            "Open time": pd.date_range("2024-01-01", periods=3, freq="min"),
            "Open": [10.0, 10.0, 10.0],
            "High": [11.0, 12.0, 13.0],
            "Low": [9.0, 8.0, 7.0],
            "Close": [10.0, 10.0, 10.0],
            "Volume": [10.0, 10.0, 10.0],
            "Quote asset volume": [100.0, 100.0, 100.0],
            "Number of trades": [1.0, 2.0, 3.0],
            "Taker buy base asset volume": [4.0, 5.0, 6.0],
            "Taker buy quote asset volume": [40.0, 50.0, 60.0],
            "foo": [1.0, 2.0, 3.0],
            "bar": ["a", "b", "c"],
        }
    )
    out = AdaptiveDollarBarsGenerator(
        lookback=1,
        k=3.0,
        extra_sum_cols=["foo"],
        extra_last_cols=["bar"],
    ).generate_bars(df)
    assert len(out) == 1
    assert out.iloc[0]["foo"] == 6.0
    assert out.iloc[0]["bar"] == "c"
    assert out.iloc[0]["High"] == 13.0
    assert out.iloc[0]["Low"] == 7.0


def test_generator_closes_bar_when_cumulative_value_reaches_threshold() -> None:
    df = pd.DataFrame(
        {
            "Open time": pd.date_range("2024-01-01", periods=1, freq="min"),
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.0],
            "Close": [10.0],
            "Volume": [10.0],
            "Quote asset volume": [100.0],
            "Number of trades": [1.0],
            "Taker buy base asset volume": [4.0],
            "Taker buy quote asset volume": [40.0],
        }
    )
    out = AdaptiveDollarBarsGenerator(lookback=1, k=1.0).generate_bars(df)
    assert len(out) == 1


def test_generator_drops_unfinished_trailing_bar() -> None:
    df = pd.DataFrame(
        {
            "Open time": pd.date_range("2024-01-01", periods=2, freq="min"),
            "Open": [10.0, 10.0],
            "High": [11.0, 11.0],
            "Low": [9.0, 9.0],
            "Close": [10.0, 10.0],
            "Volume": [10.0, 10.0],
            "Quote asset volume": [100.0, 100.0],
            "Number of trades": [1.0, 1.0],
            "Taker buy base asset volume": [4.0, 4.0],
            "Taker buy quote asset volume": [40.0, 40.0],
        }
    )
    out = AdaptiveDollarBarsGenerator(lookback=1, k=3.0).generate_bars(df)
    assert out.empty
