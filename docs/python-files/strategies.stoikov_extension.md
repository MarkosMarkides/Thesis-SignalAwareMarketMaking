# `strategies/stoikov_extension.py`

See also: [Documentation index](../README.md), [Architecture and caveats](../architecture-and-caveats.md)

Source file: [../../strategies/stoikov_extension.py](../../strategies/stoikov_extension.py)

## Purpose

This file implements the signal-aware strategy used in the final workflow.

It keeps the Stoikov quote structure but lets two forecasts enter the reservation price:

- predicted volatility
- signed predicted direction

It does not learn bid and ask prices directly.

## Public API

- `StoikovExtensionConfig`
- `PredictedVolatilityMarketMaker`
- `stoikov_extension_quote_fn`

## Main Idea

The baseline uses a fixed volatility reference. The signal-aware strategy compares predicted volatility with that baseline and adjusts the inventory penalty.

Direction is handled separately. A signed direction input shifts the reservation price by a fixed number of ticks.

The quoted spread remains the baseline Stoikov spread.

## Volatility Input

The quote function reads the volatility value from `col_sigma`.

In the final notebook, the volatility forecast is a horizon log-volatility prediction. The strategy converts it before use:

1. divide by `sqrt(sigma_horizon)`
2. multiply by the current midpoint when `sigma_is_log=True`

This gives a price-based volatility input in the same units as the baseline volatility.

The extension then forms:

```text
rho = (sigma_hat / sigma0)^2
rho_eff = 1 + alpha * (rho - 1)
```

The reservation price uses `rho_eff` in the inventory term.

## Direction Input

The quote function reads the signed direction value from `col_dir`.

The expected signed values are:

- `-1`: predicted down
- `0`: predicted neutral
- `+1`: predicted up

The direction skew is:

```text
skew_ticks * tick_size * signed_direction
```

In the final notebook, `skew_ticks=1.0` and `tick_size=0.01`, so the direction forecast shifts the reservation price by one tick up, one tick down, or not at all.

## Reservation Price

The implemented reservation price is:

```text
r = s - q * gamma * sigma0^2 * tau * rho_eff + skew
```

This is equivalent to using an effective inventory-risk term while keeping the skew separate.

## Spread

The spread is not forecast-driven.

It is computed from the baseline volatility reference:

```text
gamma * sigma0^2 * tau + (2 / gamma) * log(1 + gamma / k)
```

This is deliberate. Volatility forecasts change inventory pressure, and direction forecasts shift the quote center. They do not directly control the quoted spread.

## Oracle Use

The same quote function is used for the oracle variants.

The difference is the input:

- the forecast-based strategy uses predicted volatility and predicted direction
- the volatility oracle uses realized future volatility and zero direction
- the direction oracle uses baseline volatility and realized future direction
- the combined oracle uses realized future volatility and realized future direction

The oracle variants are not tradable because they use future labels.

## Backtester Contract

`stoikov_extension_quote_fn(...)` returns:

```text
quote_fn(row, state, cfg) -> (bid, ask)
```

The strategy returns raw float prices. The backtester handles tick rounding, no-cross enforcement, inventory limits, queue approximation, fills, cash and inventory accounting, and liquidation.

## Current Final-Notebook Settings

The final notebook uses:

- `gamma=0.05`
- `k=30.0`
- `T=5.0`
- `sigma0=0.017678641777`
- `alpha=3.0`
- `q_max=100`
- `skew_ticks=1.0`
- `sigma_horizon=20`
- `sigma_is_log=True` for forecast and volatility-oracle runs

## Caveats

- Direction classifier classes are not signed strategy directions until the notebook maps them.
- Volatility units are caller-sensitive.
- The strategy only changes the reservation price; the spread remains baseline anchored.
- The file generates quotes only. Execution behavior comes from [../../technical/backtesting.py](../../technical/backtesting.py).
