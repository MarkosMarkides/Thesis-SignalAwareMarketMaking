import numpy as np
import pandas as pd
import pytest

from conftest import fixed_quotes, make_book, make_events
from strategies.stoikov import InventoryMMConfig, InventoryMarketMaker
from strategies.stoikov_extension import PredictedVolatilityMarketMaker, StoikovExtensionConfig
from technical.backtesting import BacktestConfig, run_backtest
from technical.labelling import label_direction_future_avg, label_volatility


def test_labelling_snapshot_on_tiny_dataframes():
    vol_df = pd.DataFrame({"mid": [100.0, 101.0, 100.0, 102.0, 103.0]})
    dir_df = pd.DataFrame({"mid": [100.0, 101.0, 99.0, 102.0, 98.0, 103.0]})

    vol = label_volatility(vol_df, close_col="mid", horizon=2, out_col="y_vol")
    direction = label_direction_future_avg(dir_df, price_col="mid", horizon=1, verbose=False)

    assert vol["y_vol"].tolist() == pytest.approx(
        [0.014071892842649461, 0.02216197491016704, 0.022075484080670577, 0.0, 0.0]
    )
    assert direction["y_dir"].tolist() == [1, 0, 1, 0, 1]


def test_baseline_stoikov_quote_snapshot():
    mm = InventoryMarketMaker(InventoryMMConfig(gamma=0.1, sigma=2.0, k=1.5, T=1.0))

    out = mm.quote_from_state(100.0, inventory=2.0, t=0.25)

    assert out["p_bid"] == pytest.approx(98.60461478862429)
    assert out["p_ask"] == pytest.approx(100.19538521137572)
    assert out["r"] == pytest.approx(99.4)
    assert out["spread"] == pytest.approx(1.5907704227514234)


def test_stoikov_extension_quote_snapshot():
    mm = PredictedVolatilityMarketMaker(
        StoikovExtensionConfig(gamma=0.1, k=1.5, T=1.0, sigma0=2.0, alpha=1.5, skew_ticks=2.0)
    )

    out = mm.quote_from_state(100.0, inventory=2.0, t=0.25, sigma_hat=3.0, skew=0.02)

    assert out["p_bid"] == pytest.approx(97.49961478862429)
    assert out["p_ask"] == pytest.approx(99.09038521137572)
    assert out["r"] == pytest.approx(98.295)
    assert out["spread"] == pytest.approx(1.5907704227514234)
    assert out["rho"] == pytest.approx(2.25)
    assert out["rho_eff"] == pytest.approx(2.875)


def test_backtest_snapshot_on_tiny_event_fixture():
    book = make_book([0, 1], bid1=99.99, ask1=100.01, bid1_sz=0.0, ask1_sz=0.0)
    events = make_events([(0.0, 4, 100.0, 99.98, -1), (1.0, 4, 100.0, 100.03, 1)])
    cfg = BacktestConfig(
        order_size=100,
        max_inventory=1000,
        queue_join_frac=0.0,
        flatten_before_close_seconds=None,
    )

    results, metrics = run_backtest(book, events, fixed_quotes(100.00, 100.01), cfg=cfg)

    assert results["buy_qty"].tolist() == [100.0, 0.0]
    assert results["sell_qty"].tolist() == [0.0, 100.0]
    assert results["inventory"].tolist() == [100.0, 0.0]
    assert results["cash"].tolist() == [-10000.0, 1.0]
    assert results["pnl"].tolist() == [0.0, 1.0]
    assert metrics == {
        "pnl_final": 1.0,
        "pnl_std": 0.5,
        "sharpe_1s": 1.0,
        "avg_abs_inventory": 50.0,
        "fill_rate": 1.0,
        "volume": 200.0,
    }
