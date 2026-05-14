# `strategies/stoikov.py`

See also: [Documentation index](../README.md), [Architecture and caveats](../architecture-and-caveats.md)

Source file: [../../strategies/stoikov.py](../../strategies/stoikov.py)

## Purpose

This file implements the fixed-volatility Stoikov baseline used in the final notebook.

It is the main inventory-aware benchmark. The signal-aware strategy in [../../strategies/stoikov_extension.py](../../strategies/stoikov_extension.py) is compared against this baseline.

## Public API

- `InventoryMMConfig`
- `InventoryMarketMaker`
- `stoikov_quote_fn`

## Main Idea

The baseline places bid and ask quotes around an inventory-adjusted reservation price.

If inventory is positive, the reservation price moves below the midpoint to encourage selling. If inventory is negative, it moves above the midpoint to encourage buying.

The strategy also computes a spread from risk aversion, baseline volatility, model time, and the liquidity parameter.

## Reservation Price

The implemented reservation price is:

```text
r = s - q * gamma * sigma^2 * tau
```

where:

- `s` is the midpoint
- `q` is inventory
- `gamma` is risk aversion
- `sigma` is the fixed baseline volatility
- `tau` is remaining model time

## Spread

The implemented total spread is:

```text
gamma * sigma^2 * tau + (2 / gamma) * log(1 + gamma / k)
```

If `gamma` is zero, the implementation uses the low-risk-aversion branch:

```text
2 / k
```

## Inventory Cap

If inventory is at or above the long cap, the strategy disables the bid side. If inventory is at or below the short cap, it disables the ask side.

The backtester also enforces inventory limits, so this protection exists in both the quote logic and the replay layer.

## Time Mapping

The strategy receives timestamps in seconds, but the Stoikov formula uses model time.

The quote factory maps each session onto the interval from 0 to `T`. It resets that mapping when the row stream starts, when time moves backwards, or when a large time gap indicates a new session.

In the final notebook, `T=5.0`.

## Backtester Contract

`stoikov_quote_fn(...)` returns a quote function:

```text
quote_fn(row, state, cfg) -> (bid, ask)
```

The quote function returns raw float prices. It does not round prices, enforce the tick grid, simulate fills, or liquidate inventory. Those steps belong to [../../technical/backtesting.py](../../technical/backtesting.py).

## Current Final-Notebook Settings

The final notebook uses:

- `gamma=0.05`
- `k=30.0`
- `T=5.0`
- `q_max=100`
- baseline volatility `0.017678641777`

The baseline volatility is computed as the median training forward volatility after conversion into price units per square root second.

## Caveats

- The file only generates quotes.
- Volatility units are supplied by the caller.
- Session detection is practical and based on timestamp gaps.
- End-of-session liquidation is handled by the backtester, not by this strategy file.
