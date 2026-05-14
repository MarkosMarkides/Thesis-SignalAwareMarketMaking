from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.backtesting import (
    BacktestConfig,
    _enforce_no_cross,
    _queue_init,
    _round_to_tick,
    plot_results,
    run_backtest,
)


def _constant_quote(row, state, cfg):
    return 100.0, 101.0


def test_round_to_tick_rounds_by_side_and_rejects_bad_side() -> None:
    assert _round_to_tick(100.019, 0.01, "bid") == 100.01
    assert _round_to_tick(100.011, 0.01, "ask") == 100.02
    with pytest.raises(ValueError, match="side must be"):
        _round_to_tick(100.0, 0.01, "middle")


def test_enforce_no_cross_separates_crossed_quotes() -> None:
    bid, ask = _enforce_no_cross(100.01, 100.01, 0.01)
    assert bid < ask
    assert (bid, ask) == (100.0, 100.02)


def test_queue_init_clips_join_fraction_and_handles_bad_depth() -> None:
    assert _queue_init(10.0, 1.5) == 10.0
    assert _queue_init(10.0, -1.0) == 0.0
    assert _queue_init(np.nan, 0.5) == 0.0


def test_run_backtest_autodetects_time_abs_s(one_second_book_df: pd.DataFrame, empty_event_df: pd.DataFrame) -> None:
    df_1s = one_second_book_df.drop(columns=["time_s"])
    results, metrics = run_backtest(df_1s, empty_event_df, _constant_quote, BacktestConfig(order_size=5))
    assert "time_abs_s" in results.columns
    assert metrics["fill_rate"] == 0.0
    assert metrics["volume"] == 0.0


def test_run_backtest_raises_without_any_supported_time_column(empty_event_df: pd.DataFrame) -> None:
    df_1s = pd.DataFrame([{"mid": 100.5, "bid1": 100.0, "ask1": 101.0, "bid1_sz": 10.0, "ask1_sz": 10.0}])
    with pytest.raises(KeyError, match="Time column"):
        run_backtest(df_1s, empty_event_df, _constant_quote, BacktestConfig(order_size=5))


def test_run_backtest_floors_event_time_when_event_frame_lacks_integer_time_column(one_second_book_df: pd.DataFrame) -> None:
    df_1s = one_second_book_df.iloc[[0]].drop(columns=["time_s"])
    df_evt = pd.DataFrame(
        [
            {"time": 0.1, "event_type": 4, "size": 10.0, "price": 99.0, "direction": -1},
            {"time": 0.2, "event_type": 4, "size": 10.0, "price": 100.0, "direction": -1},
        ]
    )
    results, metrics = run_backtest(df_1s, df_evt, _constant_quote, BacktestConfig(order_size=5, queue_join_frac=0.0))
    assert results.iloc[0]["buy_qty"] == 5.0
    assert metrics["buy_count"] == 1.0


def test_run_backtest_executes_buy_fill_on_sweep_through(one_second_book_df: pd.DataFrame) -> None:
    df_1s = one_second_book_df.iloc[[0]]
    df_evt = pd.DataFrame(
        [
            {"time": 0.1, "event_type": 4, "size": 10.0, "price": 99.0, "direction": -1},
            {"time": 0.2, "event_type": 4, "size": 10.0, "price": 100.0, "direction": -1},
        ]
    )
    results, metrics = run_backtest(df_1s, df_evt, _constant_quote, BacktestConfig(order_size=5, queue_join_frac=0.0))
    assert results.iloc[0]["buy_qty"] == 5.0
    assert results.iloc[0]["inventory"] == 5.0
    assert metrics["pnl_final"] == pytest.approx(2.5)


def test_run_backtest_executes_sell_fill_on_sweep_through(one_second_book_df: pd.DataFrame) -> None:
    df_1s = one_second_book_df.iloc[[0]].drop(columns=["time_s"])
    df_evt = pd.DataFrame(
        [
            {"time": 0.1, "event_type": 4, "size": 10.0, "price": 102.0, "direction": 1},
            {"time": 0.2, "event_type": 4, "size": 10.0, "price": 101.0, "direction": 1},
        ]
    )
    results, metrics = run_backtest(df_1s, df_evt, _constant_quote, BacktestConfig(order_size=5, queue_join_frac=0.0))
    assert results.iloc[0]["sell_qty"] == 5.0
    assert results.iloc[0]["inventory"] == -5.0
    assert metrics["sell_count"] == 1.0


def test_inventory_limit_disables_bid_side_at_max_inventory(one_second_book_df: pd.DataFrame, empty_event_df: pd.DataFrame) -> None:
    results, _ = run_backtest(
        one_second_book_df.iloc[[0]],
        empty_event_df,
        _constant_quote,
        BacktestConfig(order_size=5, max_inventory=5),
        initial_inventory=5.0,
    )
    assert np.isnan(results.iloc[0]["quote_bid"])
    assert np.isfinite(results.iloc[0]["quote_ask"])


def test_plot_results_handles_empty_dataframe(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    plot_results(pd.DataFrame())
