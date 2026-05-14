# `technical/labelling.py`

See also: [Documentation index](../README.md), [Architecture and caveats](../architecture-and-caveats.md)

Source file: [../../technical/labelling.py](../../technical/labelling.py)

## Purpose

This file defines the supervised targets used by the forecasting notebooks and by the oracle strategies.

The labels are forward-looking. They are valid targets for training and evaluation, but they are not features available to a tradable strategy.

## Public API

- `VolatilityLabeler`
- `label_volatility`
- `DirectionLabelerFutureAvg`
- `label_direction_future_avg`

## Volatility Label

`label_volatility` creates forward realized log-volatility.

The steps are:

1. convert the price series to log prices
2. compute one-step log returns
3. sum future squared log returns over the chosen horizon
4. take the square root
5. write the result to the output column

For the final workflow, the horizon is 20 seconds.

The label is:

```text
sqrt(sum of future squared log returns over H seconds)
```

The labeler fills incomplete tail rows with zero, but the active notebooks remove rows without a complete future window before modelling.

## Direction Label

`label_direction_future_avg` creates the direction target.

It compares the current price with the average future price over the chosen horizon. In the final notebooks, the future average is the simple mean of the next 20 midpoint observations.

The comparison is made as a log return:

```text
log(future average price / current price)
```

The final workflow uses `balanced=True`, so each session gets its own lower and upper tercile cutoffs. This creates balanced down, up, and neutral classes within each session.

## Class Encoding

The raw label encoding is:

- `0`: down
- `1`: up
- `2`: neutral

This is the classifier target encoding.

The strategy does not use these raw labels directly. Before the signal-aware strategy uses a direction value, the notebooks map the classes into signed directions:

- `0 -> -1`
- `1 -> +1`
- `2 -> 0`

This mapping happens in the notebooks, not inside `technical/labelling.py`.

## Session Handling

The label functions do not detect sessions by themselves. The notebooks group the data by `session_id` and call the label functions session by session.

This matters because future windows should not cross from one trading session into the next.

## Current Notebook Usage

This file is used by:

- [../../exploration/directional_ML_methods.ipynb](../../exploration/directional_ML_methods.ipynb)
- [../../exploration/volatility_ML_methods.ipynb](../../exploration/volatility_ML_methods.ipynb)
- [../../exploration/train_main_model_artifacts.ipynb](../../exploration/train_main_model_artifacts.ipynb)
- [../../main.ipynb](../../main.ipynb)

It defines:

- the direction target for the XGBoost classifier
- the volatility target for the XGBoost regressor
- the realized future labels used by the oracle strategies

Changing this file changes the meaning of the forecasts and the oracle comparison.
