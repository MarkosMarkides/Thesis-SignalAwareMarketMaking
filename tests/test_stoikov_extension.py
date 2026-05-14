import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from strategies.stoikov import InventoryMMConfig, stoikov_quote_fn
from strategies.stoikov_extension import (
    PredictedVolatilityMarketMaker,
    StoikovExtensionConfig,
    stoikov_extension_quote_fn,
)


def _bt_cfg(tick_size=0.01):
    return SimpleNamespace(col_time_s="time_s", tick_size=tick_size)


def _quote_ext(cfg, row, inv=0.0, tick_size=0.01):
    quote_fn = stoikov_extension_quote_fn(cfg)
    return quote_fn(pd.Series(row), {"inv": inv}, _bt_cfg(tick_size=tick_size))


def _quotes_ext(cfg, *, mid=100.0, sigma_hat=2.0, dir_hat=0, inv=0.0, t=0.0):
    mm = PredictedVolatilityMarketMaker(cfg)
    return mm.quote_from_state(
        mid=float(mid),
        inventory=float(inv),
        t=float(t),
        sigma_hat=float(sigma_hat),
        skew=float(cfg.skew_ticks) * 0.01 * float(dir_hat),
    )


def test_extension_matches_baseline_when_direction_zero_and_sigma_matches_for_any_alpha():
    base_cfg = InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0, q_max=None)
    ext_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=3.0, q_max=None)

    row_base = pd.Series({"time_s": 0.0, "mid": 100.0})
    row_ext = pd.Series({"time_s": 0.0, "mid": 100.0, "y_vol": 2.0, "y_dir": 0})

    base_bid, base_ask = stoikov_quote_fn(base_cfg)(row_base, {"inv": 3.0}, _bt_cfg())
    ext_bid, ext_ask = stoikov_extension_quote_fn(ext_cfg)(row_ext, {"inv": 3.0}, _bt_cfg())
    ext_out = _quotes_ext(ext_cfg, mid=100.0, sigma_hat=2.0, dir_hat=0, inv=3.0)

    assert ext_bid == base_bid
    assert ext_ask == base_ask
    assert math.isclose(ext_out["rho_eff"], 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_rho_and_rho_eff_are_one_when_sigma_hat_matches_sigma0():
    cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=0.0, skew_ticks=0.0)
    out = _quotes_ext(cfg, sigma_hat=2.0, inv=3.0)

    assert math.isclose(out["rho"], 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(out["rho_eff"], 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_directional_skew_is_exact_tick_shift_and_not_volatility_scaled():
    for sigma_hat in (1.0, 5.0):
        neutral_cfg = StoikovExtensionConfig(skew_ticks=2.5, sigma0=2.0, alpha=0.0)
        up_cfg = StoikovExtensionConfig(skew_ticks=2.5, sigma0=2.0, alpha=4.0)
        row_neutral = {"time_s": 0.0, "mid": 100.0, "y_vol": sigma_hat, "y_dir": 0}
        row_up = {"time_s": 0.0, "mid": 100.0, "y_vol": sigma_hat, "y_dir": 1}

        neutral_bid, neutral_ask = _quote_ext(neutral_cfg, row_neutral, tick_size=0.01)
        up_bid, up_ask = _quote_ext(up_cfg, row_up, tick_size=0.01)

        expected_skew = 2.5 * 0.01
        assert math.isclose(up_bid - neutral_bid, expected_skew, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(up_ask - neutral_ask, expected_skew, rel_tol=0.0, abs_tol=1e-12)


def test_positive_and_negative_direction_shift_both_quotes_by_signed_ticks():
    cfg = StoikovExtensionConfig(skew_ticks=3.0, sigma0=2.0, alpha=0.0)
    row_neutral = {"time_s": 0.0, "mid": 100.0, "y_vol": 2.0, "y_dir": 0}
    row_up = {"time_s": 0.0, "mid": 100.0, "y_vol": 2.0, "y_dir": 1}
    row_down = {"time_s": 0.0, "mid": 100.0, "y_vol": 2.0, "y_dir": -1}

    neutral_bid, neutral_ask = _quote_ext(cfg, row_neutral, tick_size=0.01)
    up_bid, up_ask = _quote_ext(cfg, row_up, tick_size=0.01)
    down_bid, down_ask = _quote_ext(cfg, row_down, tick_size=0.01)

    assert math.isclose(up_bid - neutral_bid, 0.03, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(up_ask - neutral_ask, 0.03, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(down_bid - neutral_bid, -0.03, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(down_ask - neutral_ask, -0.03, rel_tol=0.0, abs_tol=1e-12)


def test_tick_size_prefers_backtest_config_then_config_fallback():
    cfg = StoikovExtensionConfig(skew_ticks=1.0, tick_size=0.05, sigma0=2.0)
    quote_fn = stoikov_extension_quote_fn(cfg)
    row_neutral = pd.Series({"time_s": 0.0, "mid": 100.0, "y_vol": 2.0, "y_dir": 0})
    row_up = pd.Series({"time_s": 0.0, "mid": 100.0, "y_vol": 2.0, "y_dir": 1})

    neutral_bid, neutral_ask = quote_fn(row_neutral, {"inv": 0.0}, _bt_cfg(tick_size=0.02))
    up_bid, up_ask = quote_fn(row_up, {"inv": 0.0}, _bt_cfg(tick_size=0.02))
    assert math.isclose(up_bid - neutral_bid, 0.02, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(up_ask - neutral_ask, 0.02, rel_tol=0.0, abs_tol=1e-12)

    cfg_without_bt_tick = SimpleNamespace(col_time_s="time_s")
    neutral_bid, neutral_ask = quote_fn(row_neutral, {"inv": 0.0}, cfg_without_bt_tick)
    up_bid, up_ask = quote_fn(row_up, {"inv": 0.0}, cfg_without_bt_tick)
    assert math.isclose(up_bid - neutral_bid, 0.05, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(up_ask - neutral_ask, 0.05, rel_tol=0.0, abs_tol=1e-12)


def test_spread_stays_anchored_to_sigma0_when_sigma_hat_or_alpha_changes():
    low_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=0.0, skew_ticks=0.0)
    high_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=4.0, skew_ticks=0.0)
    low_bid, low_ask = _quote_ext(low_cfg, {"time_s": 0.0, "mid": 100.0, "y_vol": 1.0, "y_dir": 0})
    high_bid, high_ask = _quote_ext(high_cfg, {"time_s": 0.0, "mid": 100.0, "y_vol": 3.0, "y_dir": 0})

    expected_spread = low_cfg.gamma * (low_cfg.sigma0 ** 2) * low_cfg.T + (2.0 / low_cfg.gamma) * math.log(1.0 + low_cfg.gamma / low_cfg.k)
    assert math.isclose(low_ask - low_bid, expected_spread, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(high_ask - high_bid, expected_spread, rel_tol=0.0, abs_tol=1e-12)


def test_alpha_one_matches_current_raw_rho_design():
    cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=1.0, skew_ticks=0.0)
    out = _quotes_ext(cfg, mid=100.0, sigma_hat=3.0, inv=2.0)

    reservation_from_quotes = out["r"]
    rho = (3.0 / cfg.sigma0) ** 2
    expected_reservation = 100.0 - 2 * cfg.gamma * (cfg.sigma0 ** 2) * cfg.T * rho
    assert math.isclose(reservation_from_quotes, expected_reservation, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(out["rho"], rho, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(out["rho_eff"], rho, rel_tol=0.0, abs_tol=1e-12)


def test_alpha_zero_matches_baseline_inventory_behavior_even_when_sigma_hat_differs():
    base_cfg = InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    ext_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=0.0, skew_ticks=0.0)

    base_bid, base_ask = stoikov_quote_fn(base_cfg)(
        pd.Series({"time_s": 0.0, "mid": 100.0}),
        {"inv": 5.0},
        _bt_cfg(),
    )
    ext_out = _quotes_ext(ext_cfg, mid=100.0, sigma_hat=3.0, inv=5.0)

    assert math.isclose(ext_out["rho"], (3.0 / ext_cfg.sigma0) ** 2, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ext_out["rho_eff"], 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ext_out["p_bid"], base_bid, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ext_out["p_ask"], base_ask, rel_tol=0.0, abs_tol=1e-12)


def test_alpha_above_one_strengthens_inventory_pressure_more_than_alpha_one_when_rho_above_one():
    base_cfg = InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    alpha_one_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=1.0, skew_ticks=0.0)
    alpha_three_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=3.0, skew_ticks=0.0)

    base_bid, base_ask = stoikov_quote_fn(base_cfg)(
        pd.Series({"time_s": 0.0, "mid": 100.0}),
        {"inv": 5.0},
        _bt_cfg(),
    )
    alpha_one_out = _quotes_ext(alpha_one_cfg, mid=100.0, sigma_hat=3.0, inv=5.0)
    alpha_three_out = _quotes_ext(alpha_three_cfg, mid=100.0, sigma_hat=3.0, inv=5.0)

    base_reservation = 0.5 * (base_bid + base_ask)
    assert alpha_one_out["r"] < base_reservation
    assert alpha_three_out["r"] < alpha_one_out["r"]
    assert abs(alpha_three_out["r"] - base_reservation) > abs(alpha_one_out["r"] - base_reservation)
    assert alpha_three_out["rho_eff"] > alpha_one_out["rho_eff"]


def test_alpha_above_one_weakens_inventory_pressure_more_than_alpha_one_when_rho_below_one():
    base_cfg = InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    alpha_one_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=1.0, skew_ticks=0.0)
    alpha_three_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=3.0, skew_ticks=0.0)

    base_bid, base_ask = stoikov_quote_fn(base_cfg)(
        pd.Series({"time_s": 0.0, "mid": 100.0}),
        {"inv": 5.0},
        _bt_cfg(),
    )
    alpha_one_out = _quotes_ext(alpha_one_cfg, mid=100.0, sigma_hat=1.0, inv=5.0)
    alpha_three_out = _quotes_ext(alpha_three_cfg, mid=100.0, sigma_hat=1.0, inv=5.0)

    base_reservation = 0.5 * (base_bid + base_ask)
    assert alpha_one_out["r"] > base_reservation
    assert alpha_three_out["r"] > alpha_one_out["r"]
    assert abs(alpha_three_out["r"] - base_reservation) > abs(alpha_one_out["r"] - base_reservation)
    assert alpha_three_out["rho_eff"] < alpha_one_out["rho_eff"]


def test_varying_alpha_does_not_move_quotes_when_inventory_and_direction_are_zero():
    low_alpha_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=0.0, skew_ticks=0.0)
    high_alpha_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=4.0, skew_ticks=0.0)
    low_bid, low_ask = _quote_ext(low_alpha_cfg, {"time_s": 0.0, "mid": 100.0, "y_vol": 3.0, "y_dir": 0}, inv=0.0)
    high_bid, high_ask = _quote_ext(high_alpha_cfg, {"time_s": 0.0, "mid": 100.0, "y_vol": 1.0, "y_dir": 0}, inv=0.0)

    assert math.isclose(low_bid, high_bid, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(low_ask, high_ask, rel_tol=0.0, abs_tol=1e-12)


def test_baseline_style_inventory_cap_logic():
    cfg = StoikovExtensionConfig(q_max=10, sigma0=2.0)

    capped_long_bid, capped_long_ask = _quote_ext(
        cfg,
        {"time_s": 0.0, "mid": 100.0, "y_vol": 2.0, "y_dir": 0},
        inv=10.0,
    )
    capped_short_bid, capped_short_ask = _quote_ext(
        cfg,
        {"time_s": 0.0, "mid": 100.0, "y_vol": 2.0, "y_dir": 0},
        inv=-10.0,
    )

    assert capped_long_bid == -np.inf
    assert np.isfinite(capped_long_ask)
    assert np.isfinite(capped_short_bid)
    assert capped_short_ask == np.inf


def test_gamma_zero_branch_matches_baseline_limit():
    cfg = StoikovExtensionConfig(gamma=0.0, k=2.0, T=1.0, sigma0=2.0, skew_ticks=0.0)
    bid, ask = _quote_ext(cfg, {"time_s": 0.0, "mid": 100.0, "y_vol": 10.0, "y_dir": 0})

    assert math.isclose(ask - bid, 2.0 / cfg.k, rel_tol=0.0, abs_tol=1e-12)


def test_missing_direction_defaults_to_neutral():
    cfg = StoikovExtensionConfig(sigma0=2.0)
    with_missing = _quote_ext(cfg, {"time_s": 0.0, "mid": 100.0, "y_vol": 2.0})
    with_neutral = _quote_ext(cfg, {"time_s": 0.0, "mid": 100.0, "y_vol": 2.0, "y_dir": 0})

    assert with_missing == with_neutral


def test_log_sigma_conversion_matches_baseline_when_converted_sigma_hits_sigma0():
    base_cfg = InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    ext_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, sigma_is_log=True)

    base_bid, base_ask = stoikov_quote_fn(base_cfg)(
        pd.Series({"time_s": 0.0, "mid": 100.0}),
        {"inv": 0.0},
        _bt_cfg(),
    )
    ext_bid, ext_ask = stoikov_extension_quote_fn(ext_cfg)(
        pd.Series({"time_s": 0.0, "mid": 100.0, "y_vol": 0.02, "y_dir": 0}),
        {"inv": 0.0},
        _bt_cfg(),
    )

    assert math.isclose(ext_bid, base_bid, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ext_ask, base_ask, rel_tol=0.0, abs_tol=1e-12)


def test_sigma_horizon_divides_predicted_sigma_by_square_root_horizon():
    cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=1.0, sigma_horizon=4, skew_ticks=0.0)
    same_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=1.0, sigma_horizon=1, skew_ticks=0.0)

    horizon_bid, horizon_ask = stoikov_extension_quote_fn(cfg)(
        pd.Series({"time_s": 0.0, "mid": 100.0, "y_vol": 2.0, "y_dir": 0}),
        {"inv": 3.0},
        _bt_cfg(),
    )
    same_bid, same_ask = stoikov_extension_quote_fn(same_cfg)(
        pd.Series({"time_s": 0.0, "mid": 100.0, "y_vol": 1.0, "y_dir": 0}),
        {"inv": 3.0},
        _bt_cfg(),
    )

    assert math.isclose(horizon_bid, same_bid, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(horizon_ask, same_ask, rel_tol=0.0, abs_tol=1e-12)


def test_session_aware_time_mapping_matches_baseline_with_neutral_predictions():
    base_cfg = InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    ext_cfg = StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=1.0, skew_ticks=0.0)
    base_quote = stoikov_quote_fn(base_cfg, session_seconds=100.0, session_gap_seconds=20.0)
    ext_quote = stoikov_extension_quote_fn(ext_cfg, session_seconds=100.0, session_gap_seconds=20.0)

    for ts in [0.0, 50.0, 10.0, 40.0]:
        base_bid, base_ask = base_quote(pd.Series({"time_s": ts, "mid": 100.0}), {"inv": 2.0}, _bt_cfg())
        ext_bid, ext_ask = ext_quote(
            pd.Series({"time_s": ts, "mid": 100.0, "y_vol": 2.0, "y_dir": 0}),
            {"inv": 2.0},
            _bt_cfg(),
        )

        assert math.isclose(ext_bid, base_bid, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(ext_ask, base_ask, rel_tol=0.0, abs_tol=1e-12)
