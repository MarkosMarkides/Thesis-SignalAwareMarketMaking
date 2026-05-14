from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.backtesting import BacktestConfig, run_backtest
from feature_engineering.labelling import label_direction_future_avg, label_volatility
from strategies.stoikov import InventoryMMConfig, stoikov_quote_fn
from strategies.stoikov_extension import StoikovExtensionConfig, stoikov_extension_quote_fn


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "event_type", "size", "price", "direction"])


def test_synthetic_stock_path_runs_with_baseline_stoikov() -> None:
    base = pd.DataFrame(
        {
            "time_s": [0, 1, 2, 3],
            "Close": [100.0, 100.5, 100.25, 100.75],
            "bid1": [99.95, 100.45, 100.20, 100.70],
            "ask1": [100.05, 100.55, 100.30, 100.80],
            "bid1_sz": [10.0, 10.0, 10.0, 10.0],
            "ask1_sz": [10.0, 10.0, 10.0, 10.0],
        }
    )
    base["mid"] = 0.5 * (base["bid1"] + base["ask1"])
    labeled = base.join(label_volatility(base[["Close"]], close_col="Close", horizon=2)[["y_vol"]])

    quote_fn = stoikov_quote_fn(
        InventoryMMConfig(gamma=0.1, sigma=2.0, k=2.0, T=1.0, col_mid="mid", col_time="time_s", t0=0.0, t1=3.0)
    )
    results, metrics = run_backtest(labeled, _empty_events(), quote_fn, BacktestConfig(order_size=5))
    assert len(results) == len(labeled)
    assert "pnl_final" in metrics


def test_synthetic_stock_path_runs_with_extension_and_generated_labels() -> None:
    base = pd.DataFrame(
        {
            "time_s": [0, 1, 2, 3, 4, 5],
            "Close": [100.0, 101.0, 99.0, 102.0, 100.0, 103.0],
            "bid1": [99.95, 100.95, 98.95, 101.95, 99.95, 102.95],
            "ask1": [100.05, 101.05, 99.05, 102.05, 100.05, 103.05],
            "bid1_sz": [10.0, 11.0, 12.0, 11.0, 10.0, 9.0],
            "ask1_sz": [10.0, 9.0, 8.0, 9.0, 10.0, 11.0],
        }
    )
    base["mid"] = 0.5 * (base["bid1"] + base["ask1"])

    df_vol = label_volatility(base[["Close"]], close_col="Close", horizon=1)[["y_vol"]]
    df_dir = label_direction_future_avg(base[["Close"]], price_col="Close", horizon=1, verbose=False)[["y_dir"]]
    stock_path_df = base.join(df_vol).join(df_dir, how="inner")

    quote_fn = stoikov_extension_quote_fn(
        StoikovExtensionConfig(
            gamma=0.1,
            k=2.0,
            T=1.0,
            c=0.5,
            col_mid="mid",
            col_time="time_s",
            col_sigma="y_vol",
            col_dir="y_dir",
            dir_encoding="class012",
        )
    )
    results, metrics = run_backtest(stock_path_df, _empty_events(), quote_fn, BacktestConfig(order_size=5))
    assert len(results) == len(stock_path_df)
    assert results["quote_bid"].notna().any()
    assert results["quote_ask"].notna().any()
    assert metrics["volume"] == 0.0
