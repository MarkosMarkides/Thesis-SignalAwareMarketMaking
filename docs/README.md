# Documentation Index

This folder documents the current thesis workflow.

The repository has two active layers:

- notebooks, where the AAPL experiment is assembled and run
- Python modules, where labels, quote rules, and backtesting logic live

The current paper draft is [../paper.md](../paper.md). The final experiment is [../main.ipynb](../main.ipynb).

## Pages

- [notebook-research.md](notebook-research.md): what each notebook does and which notebook is final.
- [architecture-and-caveats.md](architecture-and-caveats.md): data flow, model flow, strategy flow, and main caveats.
- [python-files/technical.labelling.md](python-files/technical.labelling.md): direction and volatility target construction.
- [python-files/technical.backtesting.md](python-files/technical.backtesting.md): event-based historical backtesting.
- [python-files/strategies.stoikov.md](python-files/strategies.stoikov.md): fixed-volatility Stoikov baseline.
- [python-files/strategies.stoikov_extension.md](python-files/strategies.stoikov_extension.md): signal-aware strategy.
- [python-files/strategies.constant_spread.md](python-files/strategies.constant_spread.md): constant spread benchmark.

## What The Repo Does

The project tests whether short-horizon forecasts can improve a structured market-making rule.

The active workflow:

1. loads AAPL book and event parquet files
2. infers trading sessions from `time_abs_s`
3. aggregates event messages onto the one-second book grid
4. lags event-derived features by one second
5. builds explainable book, event, return, volatility, and technical features
6. creates direction and volatility labels over a 20 second horizon
7. trains final XGBoost artifacts only on training sessions
8. loads those artifacts in [../main.ipynb](../main.ipynb)
9. runs constant spread, baseline Stoikov, signal-aware, and oracle backtests

The forecasting models support the strategy. They are not the main contribution.

## Active Notebooks

- [../exploration/aapl_lobster_eda.ipynb](../exploration/aapl_lobster_eda.ipynb): descriptive EDA.
- [../exploration/directional_ML_methods.ipynb](../exploration/directional_ML_methods.ipynb): direction model comparison.
- [../exploration/volatility_ML_methods.ipynb](../exploration/volatility_ML_methods.ipynb): volatility model comparison.
- [../exploration/train_main_model_artifacts.ipynb](../exploration/train_main_model_artifacts.ipynb): final artifact training.
- [../main.ipynb](../main.ipynb): final strategy workflow and paper-facing results.

## Active Python Files

- [../technical/labelling.py](../technical/labelling.py): supervised labels.
- [../technical/backtesting.py](../technical/backtesting.py): quote replay, fills, inventory, PnL, and metrics.
- [../strategies/stoikov.py](../strategies/stoikov.py): baseline inventory-aware quote rule.
- [../strategies/stoikov_extension.py](../strategies/stoikov_extension.py): signal-aware quote rule.
- [../strategies/constant_spread.py](../strategies/constant_spread.py): simple midpoint benchmark.

## What To Trust For What

Use [../main.ipynb](../main.ipynb) for the final strategy results.

Use [../exploration/train_main_model_artifacts.ipynb](../exploration/train_main_model_artifacts.ipynb) and `artifacts/main_models/model_metadata.json` for the final model artifact settings.

Use [../technical/labelling.py](../technical/labelling.py) for exact target definitions.

Use [../strategies/stoikov.py](../strategies/stoikov.py), [../strategies/stoikov_extension.py](../strategies/stoikov_extension.py), and [../technical/backtesting.py](../technical/backtesting.py) for exact strategy and backtest mechanics.

## Documentation Policy

The docs explain the current mechanics and file responsibilities. They avoid copying long notebook outputs. Numerical results should come from the executed notebook and from [../paper.md](../paper.md), where the paper-facing tables are written.
