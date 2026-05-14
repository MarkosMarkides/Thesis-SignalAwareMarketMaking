# `strategies/constant_spread.py`

See also: [Documentation index](../README.md), [Architecture and caveats](../architecture-and-caveats.md)

Source file: [../../strategies/constant_spread.py](../../strategies/constant_spread.py)

## Purpose

This file implements the simplest benchmark in the final notebook.

The constant spread strategy is not the main thesis strategy. It is included as a plain reference point before adding inventory-aware Stoikov logic.

## Public API

- `constant_spread_strategy`

The factory returns a quote function with the same interface used by the backtester:

```text
quote_fn(row, state, cfg) -> (bid, ask)
```

## What It Does

The quote function:

1. reads the best bid and best ask
2. computes the midpoint
3. places a bid and ask around the midpoint using a fixed distance
4. checks the visible best bid-ask spread
5. widens its quote if its own spread would be too tight relative to the visible market

The final notebook uses `constant=0.02`, which means a two-cent distance on each side before the backtester applies tick rounding and execution constraints.

If that quote would be tighter than the visible best bid-ask spread, the function widens it to stay just inside the visible spread, subject to a one-tick minimum.

## What It Does Not Use

The constant spread strategy does not use:

- inventory
- volatility
- direction forecasts
- remaining model time
- event data

This is why it is useful as a benchmark. It shows what happens when the strategy supplies liquidity without active inventory control.

## What The Backtester Handles

This strategy does not round prices, enforce inventory limits, simulate fills, or flatten inventory. Those steps are handled by [../../technical/backtesting.py](../../technical/backtesting.py).

## Current Role

In [../../main.ipynb](../../main.ipynb), this benchmark appears before the baseline Stoikov strategy. It provides a simple comparison against the inventory-aware and signal-aware strategies.
