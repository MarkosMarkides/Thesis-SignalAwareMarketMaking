# Notebook Research Layer

This document explains what each notebook does.

The short version: exploration notebooks help choose and understand the setup; [../main.ipynb](../main.ipynb) is the final notebook used for the paper results.

## `exploration/aapl_lobster_eda.ipynb`

This notebook is descriptive.

It inspects the AAPL book and event files and helps explain the data used in the thesis. It looks at the midpoint path, spread behavior, one-second returns, realized volatility, displayed depth, imbalance, and session structure.

It does not train the final models and does not run the final strategy backtest.

## `exploration/directional_ML_methods.ipynb`

This notebook compares direction forecasting methods.

The direction target comes from [../technical/labelling.py](../technical/labelling.py). It compares the current midpoint with the average future midpoint over the forecasting horizon and turns that future movement into down, up, and neutral classes.

This notebook is used for model comparison and context. It is not the final trading result.

## `exploration/volatility_ML_methods.ipynb`

This notebook compares volatility forecasting methods.

The volatility target comes from [../technical/labelling.py](../technical/labelling.py). It is forward realized log-volatility over the forecasting horizon.

This notebook is useful because it contains the feature engineering logic that was carried into the artifact-training notebook and the final notebook. It does not perform strategy evaluation.

## `exploration/train_main_model_artifacts.ipynb`

This notebook trains the final artifacts used by [../main.ipynb](../main.ipynb).

It saves:

- `artifacts/main_models/direction_xgb.joblib`
- `artifacts/main_models/vol_xgb.joblib`
- `artifacts/main_models/model_metadata.json`

The direction model is an XGBoost classifier. The volatility model is an XGBoost regressor. Both are trained only on the training sessions, not on the held out test sessions.

The saved metadata records the forecasting horizon, train/test sessions, selected features, and model parameters.

## `main.ipynb`

This is the final experiment notebook.

It does the following:

1. loads the AAPL book file and event file
2. identifies trading sessions
3. aggregates event messages by second
4. lags event features by one second
5. builds the selected feature set
6. constructs direction and volatility labels
7. splits the sample into training and test sessions
8. loads the saved XGBoost artifacts
9. produces out of sample direction and volatility forecasts
10. builds the strategy input table for the held out sessions
11. runs the constant spread benchmark
12. runs the baseline Stoikov strategy
13. runs the forecast-based signal-aware strategy
14. runs the three oracle variants
15. reports the final comparison metrics

The final notebook does not train the models. That separation is intentional: model fitting happens in the artifact-training notebook, while final evaluation happens in `main.ipynb`.

## Shared Conventions

The active notebooks share these conventions:

- AAPL book data are sampled on a one-second grid.
- Event messages are aggregated by second.
- Event features are lagged by one second before use.
- Labels use a 20 second forward horizon.
- Features and labels are built within sessions.
- The train/test split is time ordered.
- The final five sessions are held out for final prediction and backtesting.

## Strategy Names In The Final Notebook

The final comparison contains:

- `constant_spread`
- `baseline_stoikov`
- `ml_extension`
- `volatility_oracle`
- `direction_oracle`
- `combined_oracle`

In the paper, `ml_extension` is described as the forecast-based signal-aware strategy.

## Metrics

The final notebook reports:

- `pnl_final`
- `pnl_std`
- `sharpe_1s`
- `avg_abs_inventory`
- `fill_rate`
- `volume`

In the paper, `sharpe_1s` is written as the one-second risk-adjusted PnL ratio.

## Things To Keep In Mind

- The exploration notebooks are supporting work.
- The final result comes from [../main.ipynb](../main.ipynb).
- The oracle variants use future labels and are not tradable.
- Direction class labels must be mapped before being used as signed strategy directions.
- Numerical results should be read from executed notebook outputs and from [../paper.md](../paper.md), not from old screenshots or old drafts.
