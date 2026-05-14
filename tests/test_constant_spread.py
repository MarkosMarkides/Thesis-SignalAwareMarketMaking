import pandas as pd
import pytest

from strategies.constant_spread import constant_spread_strategy


def test_constant_spread_quotes_around_midpoint():
    quote_fn = constant_spread_strategy(constant=0.03, tick_size=0.01)
    row = pd.Series({"bid1": 99.99, "ask1": 100.01})

    bid, ask = quote_fn(row, {"inv": 500}, object())

    assert bid == pytest.approx(99.97)
    assert ask == pytest.approx(100.03)


def test_widens_when_constant_spread_is_tighter_than_l1_spread():
    quote_fn = constant_spread_strategy(constant=0.005, tick_size=0.01)
    row = pd.Series({"bid1": 99.95, "ask1": 100.05})

    bid, ask = quote_fn(row, {}, object())

    assert bid == pytest.approx(99.955)
    assert ask == pytest.approx(100.045)


def test_widening_never_uses_less_than_one_tick_spread():
    quote_fn = constant_spread_strategy(constant=0.001, tick_size=0.01)
    row = pd.Series({"bid1": 99.995, "ask1": 100.005})

    bid, ask = quote_fn(row, {}, object())

    assert ask - bid == pytest.approx(0.01)


def test_state_and_backtest_config_do_not_affect_constant_quotes():
    quote_fn = constant_spread_strategy(constant=0.02, tick_size=0.01)
    row = pd.Series({"bid1": 99.99, "ask1": 100.01})

    quote_a = quote_fn(row, {"inv": -1000, "cash": 999}, object())
    quote_b = quote_fn(row, {"inv": 1000, "cash": -999}, object())

    assert quote_a == quote_b


def test_custom_book_columns_work():
    quote_fn = constant_spread_strategy(constant=0.10, tick_size=0.01, col_bid1="best_bid", col_ask1="best_ask")
    row = pd.Series({"best_bid": 10.0, "best_ask": 10.2})

    bid, ask = quote_fn(row, {}, object())

    assert bid == pytest.approx(10.0)
    assert ask == pytest.approx(10.2)
