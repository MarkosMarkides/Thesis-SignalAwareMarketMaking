from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from technical.backtesting import BacktestConfig, run_backtest
from strategies.stoikov import InventoryMMConfig, stoikov_quote_fn
from strategies.stoikov_extension import StoikovExtensionConfig, stoikov_extension_quote_fn


def _book_df(times: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": times,
            "mid": np.full(len(times), 100.0),
            "bid1": np.full(len(times), 99.99),
            "bid1_sz": np.full(len(times), 1000.0),
            "ask1": np.full(len(times), 100.01),
            "ask1_sz": np.full(len(times), 1000.0),
            "bid2": np.full(len(times), 99.98),
            "bid2_sz": np.full(len(times), 1000.0),
            "ask2": np.full(len(times), 100.02),
            "ask2_sz": np.full(len(times), 1000.0),
            "y_vol": np.full(len(times), 2.0),
            "y_dir": np.zeros(len(times), dtype=int),
        }
    )


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "time_s", "event_type", "size", "price", "direction"])


def test_shutdown_window_forces_flatten_and_suppresses_quotes() -> None:
    calls: list[int] = []

    def quote_fn(row: pd.Series, state: dict, cfg: BacktestConfig) -> tuple[float, float]:
        calls.append(int(row[cfg.col_time_s]))
        return 99.98, 100.02

    cfg = BacktestConfig(
        order_size=10,
        max_inventory=100,
        session_seconds=100.0,
        flatten_before_close_seconds=30.0,
        liquidation_price="touch",
        col_time_s="time_s",
    )

    results, metrics = run_backtest(
        _book_df(list(range(100))),
        _empty_events(),
        quote_fn,
        cfg=cfg,
        initial_inventory=20.0,
    )

    assert calls == list(range(70))

    shutdown = results[results["time_s"] >= 70]
    assert shutdown["quote_bid"].isna().all()
    assert shutdown["quote_ask"].isna().all()
    assert shutdown["buy_qty"].eq(0.0).all()
    assert shutdown["sell_qty"].eq(0.0).all()

    first_shutdown = results.loc[results["time_s"].eq(70)].iloc[0]
    assert first_shutdown["forced_liq_qty"] == pytest.approx(20.0)
    assert first_shutdown["forced_liq_side"] == "sell"
    assert first_shutdown["forced_liq_px"] == pytest.approx(99.99)
    assert first_shutdown["inventory"] == pytest.approx(0.0)
    assert results["inventory"].iloc[-1] == pytest.approx(0.0)

    assert metrics["volume"] == pytest.approx(20.0)


def test_session_final_row_flatten_prevents_holding_beyond_truncated_data() -> None:
    cfg = BacktestConfig(
        order_size=10,
        max_inventory=100,
        session_seconds=100.0,
        flatten_before_close_seconds=30.0,
        force_flatten_at_session_end=True,
        liquidation_price="touch",
        col_time_s="time_s",
    )

    results, metrics = run_backtest(
        _book_df(list(range(50))),
        _empty_events(),
        lambda row, state, cfg: (99.98, 100.02),
        cfg=cfg,
        initial_inventory=-15.0,
    )

    final = results.iloc[-1]
    assert final["time_s"] == 49
    assert final["forced_liq_qty"] == pytest.approx(15.0)
    assert final["forced_liq_side"] == "buy"
    assert final["forced_liq_px"] == pytest.approx(100.01)
    assert final["inventory"] == pytest.approx(0.0)
    assert metrics["volume"] == pytest.approx(15.0)


def test_shutdown_rule_applies_to_baseline_and_extension_quote_functions() -> None:
    df_1s = _book_df(list(range(100)))
    cfg = BacktestConfig(
        order_size=10,
        max_inventory=100,
        session_seconds=100.0,
        flatten_before_close_seconds=30.0,
        col_time_s="time_s",
    )

    baseline_quote = stoikov_quote_fn(InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0))
    extension_quote = stoikov_extension_quote_fn(
        StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0)
    )

    for quote_fn in (baseline_quote, extension_quote):
        results, metrics = run_backtest(
            df_1s,
            _empty_events(),
            quote_fn,
            cfg=cfg,
            initial_inventory=10.0,
        )
        assert results.loc[results["time_s"] >= 70, "quote_bid"].isna().all()
        assert results.loc[results["time_s"] >= 70, "quote_ask"].isna().all()
        assert results["inventory"].iloc[-1] == pytest.approx(0.0)
        assert metrics["volume"] == pytest.approx(10.0)


def test_shutdown_can_liquidate_at_mid_price() -> None:
    cfg = BacktestConfig(
        order_size=10,
        max_inventory=100,
        session_seconds=10.0,
        flatten_before_close_seconds=5.0,
        liquidation_price="mid",
        col_time_s="time_s",
    )

    results, _ = run_backtest(
        _book_df(list(range(10))),
        _empty_events(),
        lambda row, state, cfg: (99.98, 100.02),
        cfg=cfg,
        initial_inventory=12.0,
    )

    first_shutdown = results.loc[results["forced_liq_qty"] > 0].iloc[0]
    assert first_shutdown["forced_liq_px"] == pytest.approx(100.0)
    assert first_shutdown["forced_liq_side"] == "sell"


def test_market_close_time_uses_absolute_close_clock() -> None:
    close_time = 16 * 60 * 60
    times = [9 * 60 * 60 + 30 * 60, close_time - 61, close_time - 60, close_time - 59]
    cfg = BacktestConfig(
        order_size=10,
        max_inventory=100,
        session_gap_seconds=999999.0,
        flatten_before_close_seconds=60.0,
        market_close_time_s=float(close_time),
        liquidation_price="touch",
        col_time_s="time_s",
    )

    calls: list[int] = []

    def quote_fn(row: pd.Series, state: dict, cfg: BacktestConfig) -> tuple[float, float]:
        calls.append(int(row[cfg.col_time_s]))
        return 99.98, 100.02

    results, _ = run_backtest(
        _book_df(times),
        _empty_events(),
        quote_fn,
        cfg=cfg,
        initial_inventory=5.0,
    )

    assert calls == times[:2]
    assert results.loc[2, "forced_liq_qty"] == pytest.approx(5.0)
    assert results.loc[2, "forced_liq_px"] == pytest.approx(99.99)


def test_no_shutdown_or_final_flatten_when_flatten_before_close_is_none() -> None:
    cfg = BacktestConfig(
        order_size=10,
        max_inventory=100,
        session_seconds=10.0,
        flatten_before_close_seconds=None,
        force_flatten_at_session_end=True,
        col_time_s="time_s",
    )

    results, metrics = run_backtest(
        _book_df(list(range(3))),
        _empty_events(),
        lambda row, state, cfg: (99.98, 100.02),
        cfg=cfg,
        initial_inventory=10.0,
    )

    assert results["quote_bid"].notna().all()
    assert results["inventory"].iloc[-1] == pytest.approx(10.0)
    assert metrics["volume"] == pytest.approx(0.0)


def test_liquidation_fees_are_included_in_total_and_liquidation_fee_metrics() -> None:
    cfg = BacktestConfig(
        order_size=10,
        max_inventory=100,
        session_seconds=10.0,
        flatten_before_close_seconds=5.0,
        liquidation_price="touch",
        fee_rate=0.01,
        col_time_s="time_s",
    )

    results, metrics = run_backtest(
        _book_df(list(range(10))),
        _empty_events(),
        lambda row, state, cfg: (99.98, 100.02),
        cfg=cfg,
        initial_inventory=10.0,
    )

    expected_fee = 10.0 * 99.99 * 0.01
    first_shutdown = results.loc[results["forced_liq_qty"] > 0].iloc[0]
    assert first_shutdown["fees_paid"] == pytest.approx(expected_fee)
    assert results["fees_paid"].iloc[-1] == pytest.approx(expected_fee)
    assert metrics["volume"] == pytest.approx(10.0)
