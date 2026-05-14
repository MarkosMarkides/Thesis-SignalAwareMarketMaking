import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from strategies.stoikov import InventoryMMConfig, InventoryMarketMaker, stoikov_quote_fn


def _bt_cfg(tick_size=0.01):
    return SimpleNamespace(col_time_s="time_s", tick_size=tick_size)


def test_reservation_price_formula():
    cfg = InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    mm = InventoryMarketMaker(cfg)

    assert mm.reservation_price(s=100.0, q=3, t=0.25) == pytest.approx(100.0 - 3 * 0.1 * 4.0 * 0.75)


def test_total_spread_formula_and_gamma_zero_branch():
    cfg = InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    mm = InventoryMarketMaker(cfg)
    expected = 0.1 * 4.0 * 1.0 + (2.0 / 0.1) * math.log(1.0 + 0.1 / 1.5)

    assert mm.total_spread(t=0.0) == pytest.approx(expected)

    zero_gamma = InventoryMarketMaker(InventoryMMConfig(gamma=0.0, sigma=2.0, k=2.0, T=1.0))
    assert zero_gamma.total_spread(t=0.0) == pytest.approx(1.0)


def test_spread_shrinks_as_model_time_approaches_T():
    mm = InventoryMarketMaker(InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0))

    assert mm.total_spread(t=0.9) < mm.total_spread(t=0.1)


def test_inventory_moves_reservation_price_in_expected_direction():
    mm = InventoryMarketMaker(InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0))

    neutral = mm.reservation_price(100.0, q=0, t=0.0)
    long = mm.reservation_price(100.0, q=5, t=0.0)
    short = mm.reservation_price(100.0, q=-5, t=0.0)

    assert long < neutral
    assert short > neutral


def test_q_max_suppresses_bid_at_long_cap_and_ask_at_short_cap():
    mm = InventoryMarketMaker(InventoryMMConfig(q_max=10))

    long_quote = mm.quotes(100.0, q=10, t=0.0)
    short_quote = mm.quotes(100.0, q=-10, t=0.0)

    assert long_quote["p_bid"] == -np.inf
    assert np.isfinite(long_quote["p_ask"])
    assert np.isfinite(short_quote["p_bid"])
    assert short_quote["p_ask"] == np.inf


def test_quote_from_state_clamps_model_time_and_rounds_inventory():
    mm = InventoryMarketMaker(InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0))

    before_start = mm.quote_from_state(100.0, inventory=2.6, t=-10.0)
    at_start = mm.quotes(100.0, q=3, t=0.0)
    after_end = mm.quote_from_state(100.0, inventory=2.6, t=10.0)
    at_end = mm.quotes(100.0, q=3, t=1.0)

    assert before_start["p_bid"] == pytest.approx(at_start["p_bid"])
    assert before_start["p_ask"] == pytest.approx(at_start["p_ask"])
    assert after_end["p_bid"] == pytest.approx(at_end["p_bid"])
    assert after_end["p_ask"] == pytest.approx(at_end["p_ask"])


def test_quote_factory_detects_new_sessions_from_first_row_backwards_time_and_large_gap():
    cfg = InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    quote_fn = stoikov_quote_fn(cfg, session_seconds=100.0, session_gap_seconds=100.0)
    direct = InventoryMarketMaker(cfg)

    rows = [
        (0.0, 0.0),
        (50.0, 0.5),
        (10.0, 0.0),
        (200.0, 0.0),
    ]

    for ts, model_time in rows:
        bid, ask = quote_fn(pd.Series({"time_s": ts, "mid": 100.0}), {"inv": 1.0}, _bt_cfg())
        expected = direct.quotes(100.0, q=1, t=model_time)
        assert bid == pytest.approx(expected["p_bid"])
        assert ask == pytest.approx(expected["p_ask"])


def test_explicit_time_range_overrides_session_detection():
    cfg = InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0, t0=0.0, t1=100.0)
    quote_fn = stoikov_quote_fn(cfg, session_seconds=10.0, session_gap_seconds=1.0)
    direct = InventoryMarketMaker(cfg)

    bid, ask = quote_fn(pd.Series({"time_s": 50.0, "mid": 100.0}), {"inv": 2.0}, _bt_cfg())
    expected = direct.quotes(100.0, q=2, t=0.5)

    assert bid == pytest.approx(expected["p_bid"])
    assert ask == pytest.approx(expected["p_ask"])
