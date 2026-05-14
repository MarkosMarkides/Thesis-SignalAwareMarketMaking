# `technical/backtesting.py`

See also: [Documentation index](../README.md), [Architecture and caveats](../architecture-and-caveats.md)

Source file: [../../technical/backtesting.py](../../technical/backtesting.py)

## Purpose

This file turns quote functions into historical market-making simulations.

It is the shared backtester for the constant spread benchmark, the baseline Stoikov strategy, the signal-aware strategy, and the oracle variants.

## Public API

- `BacktestConfig`
- `run_backtest`
- `plot_results`

`plot_results` draws the PnL and inventory plots used in the final notebook.

## Quote Function Contract

`run_backtest(...)` accepts any quote function with this shape:

```text
quote_fn(row, state, cfg) -> (bid, ask)
```

The strategy returns raw bid and ask prices. The backtester is responsible for making those quotes executable in the historical replay.

## Main Backtest Steps

At each one-second book row, the backtester:

1. checks whether the session is in the flattening window
2. asks the strategy for bid and ask quotes
3. rounds bid quotes down and ask quotes up to the tick grid
4. prevents crossed or equal quotes
5. disables quote sides that would exceed inventory limits
6. initializes or updates queue position
7. checks execution events for that second
8. records fills at the strategy quote price
9. updates cash, inventory, fees, and PnL

The same logic is used for every final strategy.

## Tick Rounding And No-Cross Logic

Bids are rounded down to the nearest tick. Asks are rounded up to the nearest tick.

If the rounded bid is greater than or equal to the rounded ask, the backtester recenters the quotes and moves them apart. This prevents the strategy from submitting crossed quotes after rounding.

## Queue Approximation

The backtester uses a simple queue model.

If a quote improves the visible best price, its queue starts at zero. If it joins displayed liquidity, its queue starts behind a fraction of displayed depth. In the final notebook, `queue_join_frac=1`, so a quote that joins displayed liquidity starts behind the displayed size.

When the quote price changes, the queue position resets.

## Event-Based Fills

Only execution events are used for fills. In the current setup, execution event types are `{4, 5}`.

Sell executions can fill the strategy bid. Buy executions can fill the strategy ask. A fill requires queue depletion and sweep-through behavior in the execution messages. A trade merely touching the quote is not enough by itself.

This is still an approximation. It is more realistic than filling from one-second candles, but it is not a full reconstruction of exchange priority.

## Cash, Inventory, And PnL

When the strategy buys, inventory rises and cash falls. When it sells, inventory falls and cash rises.

PnL is marked each second as:

```text
cash + inventory * midpoint
```

The final run uses zero fees and rebates.

## Session-End Liquidation

The backtester handles end-of-session flattening for all strategies.

In the final setup, passive quoting stops during the final 30 minutes of the session. Remaining inventory is liquidated at the touch:

- long inventory is sold at the best bid
- short inventory is bought back at the best ask

This keeps all strategies flat by the end of the session.

## Metrics

The metrics dictionary contains:

- `pnl_final`
- `pnl_std`
- `sharpe_1s`
- `avg_abs_inventory`
- `fill_rate`
- `volume`

`sharpe_1s` is the mean one-second PnL change divided by the standard deviation of one-second PnL changes. In the paper it is called the one-second risk-adjusted PnL ratio.

`fill_rate` is the fraction of one-second rows with at least one passive fill. `volume` includes passive fills and forced liquidation volume.

## What It Does Not Do

The backtester does not:

- model latency
- model the strategy's own cancellations
- reconstruct full queue priority
- model market impact
- estimate strategy parameters
- train forecasting models

It is a controlled historical replay tool for comparing quote rules under the same assumptions.
