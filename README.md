# Adaptive Market Making With Forecast Signals

This repository contains the workflow for a methodology-focused market-making thesis using AAPL LOBSTER data.

The project is not mainly about building a novel machine learning model. The forecasting models are used as inputs to a structured market-making rule. The main question is whether short-horizon volatility and direction forecasts can improve an inventory-aware quoting strategy without replacing the economic logic of the Avellaneda and Stoikov framework.

The current writing file is [paper.md](paper.md). The final experiment is [main.ipynb](main.ipynb).

## Current Workflow

The active workflow is:

1. inspect the AAPL LOBSTER sample in [exploration/aapl_lobster_eda.ipynb](exploration/aapl_lobster_eda.ipynb)
2. compare direction forecasting methods in [exploration/directional_ML_methods.ipynb](exploration/directional_ML_methods.ipynb)
3. compare volatility forecasting methods in [exploration/volatility_ML_methods.ipynb](exploration/volatility_ML_methods.ipynb)
4. train the final XGBoost artifacts in [exploration/train_main_model_artifacts.ipynb](exploration/train_main_model_artifacts.ipynb)
5. load those artifacts and run the final backtests in [main.ipynb](main.ipynb)

*The final notebook loads trained artifacts. It does not train the final models again.

## Data

The active data files are:

- `data/AAPL_1s.parquet`
- `data/AAPL_evt.parquet`

The book file contains the one-second AAPL limit order book grid. The event file contains timestamped order book messages. The event file is used twice: first to build event-based predictors, and later to simulate fills in the historical backtest.

Sessions are inferred from `time_abs_s`. A new session starts when time moves backwards or when the gap between observations is larger than 3,600 seconds.

The final modelling sample is split by time. Earlier sessions are used for training, and the last five sessions are held out for final prediction and backtesting.

## Strategy Comparison

The final comparison in [main.ipynb](main.ipynb) uses:

- **Constant Spread**: midpoint quoting with a fixed distance from the midpoint.
- **Baseline Stoikov**: fixed-volatility inventory-aware quoting.
- **Signal-Aware Strategy**: the Stoikov reservation price adjusted by predicted volatility and predicted direction.
- **Volatility Oracle**: the signal-aware strategy using realized future volatility and no direction signal.
- **Direction Oracle**: the signal-aware strategy using realized future direction and baseline volatility.
- **Combined Oracle**: the signal-aware strategy using realized future volatility and realized future direction.

The oracle variants are diagnostic only. They use future labels and are not tradable.

## Forecasting Role

The final forecasting artifacts are:

- `artifacts/main_models/direction_xgb.joblib`
- `artifacts/main_models/vol_xgb.joblib`
- `artifacts/main_models/model_metadata.json`

Both final models are XGBoost models. The direction model predicts down, up, or neutral movement over a 20 second horizon. The volatility model predicts forward realized log-volatility over the same horizon.

Before the strategy uses direction predictions, classes are mapped into signed values:

- down becomes `-1`
- up becomes `+1`
- neutral becomes `0`

The volatility forecast is converted into the same units as the baseline volatility before it enters the Stoikov reservation price.

## Signal-Aware Strategy

The signal-aware strategy keeps the baseline spread anchored to the fixed-volatility Stoikov rule. Forecasts do not directly widen or narrow the spread.

The extension changes only the quote center:

- predicted volatility changes the inventory penalty through an effective inventory risk term
- predicted direction shifts the reservation price by a one-tick signed skew

This is the main design choice in the project. The model remains an interpretable quoting rule, not a learned trading policy.

## Backtesting

All strategies use the same event-based historical backtester in [technical/backtesting.py](technical/backtesting.py).

The backtester handles:

- tick rounding
- no-cross quote enforcement
- inventory limits
- queue approximation
- fills from execution events
- cash, inventory, and PnL accounting
- end-of-session liquidation
- zero transaction fees in the final run

The final metrics are:

- `pnl_final`
- `pnl_std`
- `sharpe_1s`
- `avg_abs_inventory`
- `fill_rate`
- `volume`

In the paper, `sharpe_1s` is described as the one-second risk-adjusted PnL ratio.

## Documentation

Detailed documentation lives in:

- [docs/README.md](docs/README.md)
- [docs/notebook-research.md](docs/notebook-research.md)
- [docs/architecture-and-caveats.md](docs/architecture-and-caveats.md)
- [docs/python-files/technical.labelling.md](docs/python-files/technical.labelling.md)
- [docs/python-files/technical.backtesting.md](docs/python-files/technical.backtesting.md)
- [docs/python-files/strategies.stoikov.md](docs/python-files/strategies.stoikov.md)
- [docs/python-files/strategies.stoikov_extension.md](docs/python-files/strategies.stoikov_extension.md)
- [docs/python-files/strategies.constant_spread.md](docs/python-files/strategies.constant_spread.md)

