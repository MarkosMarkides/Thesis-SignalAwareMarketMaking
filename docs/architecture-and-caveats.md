# Architecture And Caveats

This document explains how the current project fits together.

The repo is notebook-led. The notebooks run the AAPL experiment, and the Python files hold the reusable pieces.

## System Map

```text
data/AAPL_1s.parquet
data/AAPL_evt.parquet
  -> exploration/aapl_lobster_eda.ipynb
  -> exploration/directional_ML_methods.ipynb
  -> exploration/volatility_ML_methods.ipynb
  -> exploration/train_main_model_artifacts.ipynb
       -> artifacts/main_models/direction_xgb.joblib
       -> artifacts/main_models/vol_xgb.joblib
       -> artifacts/main_models/model_metadata.json
  -> main.ipynb
       -> Constant Spread
       -> Baseline Stoikov
       -> Signal-Aware Strategy
       -> Volatility Oracle
       -> Direction Oracle
       -> Combined Oracle
```

The Python layer supports the notebook workflow:

```text
technical/labelling.py
  -> direction and volatility labels

strategies/constant_spread.py
  -> constant spread quote function

strategies/stoikov.py
  -> fixed-volatility Stoikov quote function

strategies/stoikov_extension.py
  -> signal-aware quote function

technical/backtesting.py
  -> historical replay, fills, inventory, cash, PnL, and metrics
```

## Data Flow

The final data flow is:

```text
raw book and event parquet files
  -> session construction
  -> event aggregation by second
  -> one-second event lag
  -> feature construction
  -> forward direction and volatility labels
  -> time-ordered train/test split
  -> saved XGBoost artifacts
  -> imported test-period forecasts
  -> strategy input table
  -> event-based backtests
  -> final comparison metrics
```

The train/test split is chronological. Sessions 0 to 16 train the final artifacts, and sessions 17 to 21 are held out for final prediction and backtesting.

## Data And Sessions

The book file is a one-second AAPL order book grid. The event file contains order submissions, cancellations, deletions, visible executions, and hidden executions.

Sessions are inferred from `time_abs_s`. A new session begins when time moves backwards or when the gap between adjacent rows exceeds 3,600 seconds.

The final modelling sample uses:

- horizon: 20 seconds
- rolling warmup: 20 seconds
- event feature lag: 1 second
- held out test sessions: 5

Rows without a complete future label window are removed before modelling. In the current run, the final modelling sample contains 513,452 rows: 396,718 training rows from sessions 0 to 16 and 116,734 test rows from sessions 17 to 21.

## Event Alignment

Event messages are aggregated onto the one-second grid. The event features are then shifted forward by one second before being merged with the book data.

This is important. If messages inside second \(t\) were used as features for the quote at second \(t\), the model would be allowed to see information that may not have been available when the quote was placed. The one-second lag keeps the feature set aligned with the quote decision.

## Feature Set

The final feature set is deliberately explainable. It is not presented as an exhaustive search over all possible predictors.

The features cover:

- imbalance at the touch and across deeper levels
- displayed bid and ask depth
- spread and one-tick spread state
- microprice and microprice bias
- recent event and trade activity
- order flow imbalance
- short-horizon returns
- realized volatility
- simple technical indicators such as RSI, MACD, Bollinger position, and VWAP

The same selected feature list is stored in `artifacts/main_models/model_metadata.json`.

## Labels

Direction and volatility labels are forward-looking supervised targets.

The direction label compares the current midpoint with the average future midpoint over the next 20 seconds. In the final workflow, labels are balanced within each session into down, up, and neutral classes. The raw class encoding is:

- `0`: down
- `1`: up
- `2`: neutral

Before strategy use, these are mapped into signed directions:

- `0 -> -1`
- `1 -> +1`
- `2 -> 0`

The volatility label is forward realized log-volatility over the same 20 second horizon.

## Forecasting Artifacts

The final artifacts are trained in [../exploration/train_main_model_artifacts.ipynb](../exploration/train_main_model_artifacts.ipynb) and loaded in [../main.ipynb](../main.ipynb).

The direction artifact is an `XGBClassifier`. The volatility artifact is an `XGBRegressor`. Both use the same selected feature set and the same chronological training sessions.

The final notebook clips volatility predictions at zero. It then converts horizon log-volatility into price units before using it in the Stoikov reservation price.

## Strategy Mechanics

### Constant Spread

The constant spread benchmark quotes around the midpoint using a fixed half spread. It does not use inventory, volatility, direction, or time.

### Baseline Stoikov

The baseline uses the Avellaneda and Stoikov inventory-aware reservation price and spread structure. It uses a fixed baseline volatility estimated from the training period.

Inventory shifts the reservation price. Positive inventory moves the reservation price down, encouraging selling. Negative inventory moves it up, encouraging buying.

In the current run, the baseline volatility is the median training forward volatility converted into price units per square root second: `0.017678641777`.

### Signal-Aware Strategy

The signal-aware strategy modifies the reservation price, not the whole quoting rule.

Predicted volatility changes the inventory penalty through an effective inventory risk term. Predicted direction shifts the reservation price by a signed tick. The spread remains equal to the baseline Stoikov spread.

This is the central design choice. The forecasts enter the model through interpretable parts of the reservation price, while the quoted spread remains baseline anchored.

### Oracles

The oracle variants use realized future labels instead of forecasts:

- the volatility oracle uses realized future volatility and no direction skew
- the direction oracle uses realized future direction and baseline volatility
- the combined oracle uses both realized future volatility and realized future direction

These strategies are not tradable. They are used to diagnose whether the signal-aware rule would benefit from stronger signals.

## Backtesting

[../technical/backtesting.py](../technical/backtesting.py) is an event-based historical replay engine.

At each book row, it:

1. asks the strategy for raw bid and ask quotes
2. rounds bids down and asks up to the tick grid
3. prevents crossed quotes
4. disables quote sides that would break inventory limits
5. approximates queue position
6. checks execution messages for fills
7. updates cash and inventory
8. marks PnL to the midpoint
9. liquidates remaining inventory near the end of each session

The final run uses 10-share orders, a +/-100 share inventory limit, zero fees, and liquidation at the touch.

## Main Caveats

- The project is a research workflow, not a production trading system.
- [../main.ipynb](../main.ipynb) is the final experiment; exploration notebooks are supporting evidence.
- The final notebook loads model artifacts instead of retraining them.
- The oracle variants use future information and are not tradable.
- The backtester approximates queue priority and fills from historical executions; it is not a full matching engine.
- Event lagging is part of the causal design and should not be removed casually.
- Direction labels and signed strategy directions are different objects.
- The empirical result is sample-specific and should be presented as methodology evidence, not as a universal profitability claim.
