import numpy as np
import pytest

from conftest import fixed_quotes, make_book, make_events, no_quotes
from technical.backtesting import BacktestConfig, run_backtest


def _cfg(**kwargs):
    base = {
        "order_size": 100,
        "max_inventory": 1000,
        "fee_rate": 0.0,
        "queue_join_frac": 0.0,
        "flatten_before_close_seconds": None,
    }
    base.update(kwargs)
    return BacktestConfig(**base)


def test_bid_rounds_down_and_ask_rounds_up_to_tick():
    book = make_book([0])
    cfg = _cfg(tick_size=0.01)

    results, _ = run_backtest(book, make_events([]), fixed_quotes(100.006, 100.014), cfg=cfg)

    assert results.loc[0, "quote_bid"] == pytest.approx(100.00)
    assert results.loc[0, "quote_ask"] == pytest.approx(100.02)


def test_crossed_quotes_are_moved_to_uncrossed_prices():
    book = make_book([0])
    cfg = _cfg(tick_size=0.01)

    results, _ = run_backtest(book, make_events([]), fixed_quotes(100.02, 100.01), cfg=cfg)

    assert results.loc[0, "quote_bid"] == pytest.approx(100.01)
    assert results.loc[0, "quote_ask"] == pytest.approx(100.02)
    assert results.loc[0, "quote_bid"] < results.loc[0, "quote_ask"]


def test_improving_touch_starts_at_front_of_queue_and_bid_fills_on_sell_sweep():
    book = make_book([0], bid1=99.99, ask1=100.03)
    events = make_events([(0.0, 4, 10.0, 99.98, -1)])

    results, metrics = run_backtest(book, events, fixed_quotes(100.00, 100.05), cfg=_cfg())

    assert results.loc[0, "buy_qty"] == pytest.approx(100.0)
    assert results.loc[0, "buy_px"] == pytest.approx(100.0)
    assert results.loc[0, "inventory"] == pytest.approx(100.0)
    assert results.loc[0, "cash"] == pytest.approx(-10000.0)
    assert metrics["fill_rate"] == pytest.approx(1.0)
    assert metrics["volume"] == pytest.approx(100.0)


def test_ask_fills_only_on_buy_initiated_sweep_through_ask():
    book = make_book([0], bid1=99.97, ask1=100.01)
    events = make_events([(0.0, 4, 10.0, 100.03, 1)])

    results, metrics = run_backtest(book, events, fixed_quotes(99.95, 100.00), cfg=_cfg())

    assert results.loc[0, "sell_qty"] == pytest.approx(100.0)
    assert results.loc[0, "sell_px"] == pytest.approx(100.0)
    assert results.loc[0, "inventory"] == pytest.approx(-100.0)
    assert results.loc[0, "cash"] == pytest.approx(10000.0)
    assert metrics["fill_rate"] == pytest.approx(1.0)
    assert metrics["volume"] == pytest.approx(100.0)


def test_touching_the_quote_without_sweep_through_does_not_fill():
    book = make_book([0], bid1=99.99, ask1=100.01)
    events = make_events([(0.0, 4, 100.0, 99.99, -1), (0.0, 4, 100.0, 100.01, 1)])

    results, metrics = run_backtest(book, events, fixed_quotes(99.99, 100.01), cfg=_cfg())

    assert results.loc[0, "buy_qty"] == pytest.approx(0.0)
    assert results.loc[0, "sell_qty"] == pytest.approx(0.0)
    assert metrics["volume"] == pytest.approx(0.0)


def test_no_fill_while_queue_ahead_remains_positive():
    book = make_book([0], bid1=99.99, ask1=100.01, bid1_sz=100.0)
    events = make_events([(0.0, 4, 50.0, 99.98, -1)])

    results, _ = run_backtest(
        book,
        events,
        fixed_quotes(99.99, 100.05),
        cfg=_cfg(queue_join_frac=1.0),
    )

    assert results.loc[0, "buy_qty"] == pytest.approx(0.0)
    assert results.loc[0, "inventory"] == pytest.approx(0.0)


def test_l1_queue_can_be_depleted_across_seconds_before_fill():
    book = make_book([0, 1], bid1=99.99, ask1=100.01, bid1_sz=100.0)
    events = make_events([(0.0, 4, 50.0, 99.98, -1), (1.0, 4, 50.0, 99.98, -1)])

    results, _ = run_backtest(
        book,
        events,
        fixed_quotes(99.99, 100.05),
        cfg=_cfg(queue_join_frac=1.0),
    )

    assert results.loc[0, "buy_qty"] == pytest.approx(0.0)
    assert results.loc[1, "buy_qty"] == pytest.approx(100.0)


def test_l2_queue_depth_is_used_when_quote_joins_l2_price():
    book = make_book([0], bid1=99.99, ask1=100.01, bid2=99.98, bid2_sz=200.0)
    events = make_events([(0.0, 4, 100.0, 99.97, -1)])

    results, _ = run_backtest(
        book,
        events,
        fixed_quotes(99.98, 100.05),
        cfg=_cfg(queue_join_frac=1.0),
    )

    assert results.loc[0, "buy_qty"] == pytest.approx(0.0)


def test_quote_price_change_resets_queue_position():
    book = make_book([0, 1], bid1=99.99, ask1=100.01, bid1_sz=100.0)
    events = make_events([(0.0, 4, 50.0, 99.98, -1), (1.0, 4, 50.0, 99.97, -1)])

    def quote_fn(row, state, cfg):
        if int(row["time_s"]) == 0:
            return 99.99, 100.05
        return 99.98, 100.05

    results, _ = run_backtest(book, events, quote_fn, cfg=_cfg(queue_join_frac=1.0))

    assert results.loc[0, "buy_qty"] == pytest.approx(0.0)
    assert results.loc[1, "buy_qty"] == pytest.approx(0.0)


def test_inventory_caps_disable_quotes_that_would_exceed_limits():
    book = make_book([0])

    long_results, _ = run_backtest(
        book,
        make_events([]),
        fixed_quotes(99.99, 100.01),
        cfg=_cfg(order_size=100, max_inventory=100),
        initial_inventory=100.0,
    )
    short_results, _ = run_backtest(
        book,
        make_events([]),
        fixed_quotes(99.99, 100.01),
        cfg=_cfg(order_size=100, max_inventory=100),
        initial_inventory=-100.0,
    )

    assert np.isnan(long_results.loc[0, "quote_bid"])
    assert np.isnan(short_results.loc[0, "quote_ask"])


def test_cash_inventory_fees_pnl_and_metrics_on_buy_then_sell_fixture():
    book = make_book([0, 1], bid1=99.99, ask1=100.01)
    events = make_events([(0.0, 4, 100.0, 99.98, -1), (1.0, 4, 100.0, 100.03, 1)])
    cfg = _cfg(fee_rate=0.001)

    results, metrics = run_backtest(book, events, fixed_quotes(100.00, 100.01), cfg=cfg)

    assert results.loc[0, "inventory"] == pytest.approx(100.0)
    assert results.loc[0, "cash"] == pytest.approx(-10010.0)
    assert results.loc[1, "inventory"] == pytest.approx(0.0)
    assert results.loc[1, "cash"] == pytest.approx(-19.001)
    assert results.loc[1, "pnl"] == pytest.approx(-19.001)

    assert metrics["pnl_final"] == pytest.approx(-19.001)
    assert metrics["pnl_std"] == pytest.approx(4.5005)
    assert metrics["sharpe_1s"] == pytest.approx(-1.0)
    assert metrics["avg_abs_inventory"] == pytest.approx(50.0)
    assert metrics["fill_rate"] == pytest.approx(1.0)
    assert metrics["volume"] == pytest.approx(200.0)
    assert set(metrics) == {"pnl_final", "pnl_std", "sharpe_1s", "avg_abs_inventory", "fill_rate", "volume"}
    assert results.loc[1, "fees_paid"] == pytest.approx(20.001)


def test_no_quote_function_leaves_position_unchanged_when_shutdown_is_disabled():
    book = make_book([0, 1])

    results, metrics = run_backtest(
        book,
        make_events([]),
        no_quotes,
        cfg=_cfg(),
        initial_inventory=25.0,
        initial_cash=10.0,
    )

    assert results["inventory"].tolist() == [25.0, 25.0]
    assert results["cash"].tolist() == [10.0, 10.0]
    assert metrics["volume"] == pytest.approx(0.0)
