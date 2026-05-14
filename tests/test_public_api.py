import inspect


def test_active_public_imports_are_available():
    from strategies.constant_spread import constant_spread_strategy
    from strategies.stoikov import InventoryMMConfig, InventoryMarketMaker, stoikov_quote_fn
    from strategies.stoikov_extension import (
        PredictedVolatilityMarketMaker,
        StoikovExtensionConfig,
        stoikov_extension_quote_fn,
    )
    from technical.backtesting import BacktestConfig, plot_results, run_backtest
    from technical.labelling import (
        DirectionLabelerFutureAvg,
        VolatilityLabeler,
        label_direction_future_avg,
        label_volatility,
    )

    assert callable(constant_spread_strategy)
    assert InventoryMarketMaker(InventoryMMConfig())
    assert callable(stoikov_quote_fn)
    assert PredictedVolatilityMarketMaker(StoikovExtensionConfig())
    assert callable(stoikov_extension_quote_fn)
    assert BacktestConfig()
    assert callable(run_backtest)
    assert callable(plot_results)
    assert VolatilityLabeler()
    assert DirectionLabelerFutureAvg()
    assert callable(label_volatility)
    assert callable(label_direction_future_avg)


def test_notebook_used_parameters_remain_in_public_signatures():
    from strategies.constant_spread import constant_spread_strategy
    from strategies.stoikov import InventoryMMConfig, stoikov_quote_fn
    from strategies.stoikov_extension import StoikovExtensionConfig, stoikov_extension_quote_fn
    from technical.backtesting import BacktestConfig, run_backtest
    from technical.labelling import label_direction_future_avg, label_volatility

    assert {"constant", "tick_size", "col_bid1", "col_ask1"} <= set(inspect.signature(constant_spread_strategy).parameters)
    assert {"gamma", "sigma", "k", "T", "q_max", "col_mid", "col_time", "t0", "t1"} <= set(
        inspect.signature(InventoryMMConfig).parameters
    )
    assert {"mm_cfg", "session_seconds", "session_gap_seconds"} <= set(inspect.signature(stoikov_quote_fn).parameters)
    assert {"sigma0", "alpha", "skew_ticks", "sigma_is_log", "sigma_horizon", "col_sigma", "col_dir"} <= set(
        inspect.signature(StoikovExtensionConfig).parameters
    )
    assert {"ext_cfg", "session_seconds", "session_gap_seconds"} <= set(
        inspect.signature(stoikov_extension_quote_fn).parameters
    )
    assert {"tick_size", "order_size", "max_inventory", "col_time_s", "flatten_before_close_seconds"} <= set(
        inspect.signature(BacktestConfig).parameters
    )
    assert {"df_1s", "df_evt", "quote_fn", "cfg", "initial_cash", "initial_inventory"} <= set(
        inspect.signature(run_backtest).parameters
    )
    assert {"df", "close_col", "horizon", "out_col"} <= set(inspect.signature(label_volatility).parameters)
    assert {"df", "price_col", "vol_col", "horizon", "avg_mode", "balanced"} <= set(
        inspect.signature(label_direction_future_avg).parameters
    )

