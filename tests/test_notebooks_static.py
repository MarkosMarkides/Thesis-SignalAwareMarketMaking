from pathlib import Path

import pytest

from conftest import notebook_json, notebook_source


@pytest.mark.notebook
def test_all_expected_notebooks_are_valid_json(notebook_paths):
    for path in notebook_paths:
        nb = notebook_json(path)
        assert nb["cells"]
        assert nb["metadata"]


@pytest.mark.notebook
def test_only_main_notebook_lives_at_project_root(project_root):
    root_notebooks = sorted(path.name for path in project_root.glob("*.ipynb"))

    assert root_notebooks == ["main.ipynb"]


@pytest.mark.notebook
def test_notebooks_use_current_module_paths(notebook_paths):
    text = "\n".join(notebook_source(path) for path in notebook_paths)

    assert "technical.labelling" in text
    assert "technical.backtesting" in text
    assert "strategies.stoikov" in text
    assert "strategies.stoikov_extension" in text
    assert "feature_engineering" not in text
    assert "backtesting.backtesting" not in text


@pytest.mark.notebook
def test_main_loads_saved_artifacts_instead_of_training_final_models(project_root):
    text = notebook_source(project_root / "main.ipynb")

    assert 'joblib.load(ARTIFACT_DIR / "direction_xgb.joblib")' in text
    assert 'joblib.load(ARTIFACT_DIR / "vol_xgb.joblib")' in text
    assert "torch.load" not in text
    assert "RandomForestClassifier(" not in text
    assert "XGBClassifier(" not in text
    assert "XGBRegressor(" not in text
    assert "rf_model.fit(" not in text
    assert "lstm_model.fit(" not in text


@pytest.mark.notebook
def test_main_contains_baseline_extension_and_oracle_backtests(project_root):
    text = notebook_source(project_root / "main.ipynb")

    assert "Constant Spread Strategy Backtest" in text
    assert "Baseline Stoikov Backtest" in text
    assert "ML Stoikov Extension Backtest" in text
    assert "Oracle Stoikov Extension Backtest" in text
    assert text.index("Constant Spread Strategy Backtest") < text.index("Baseline Stoikov Backtest")
    assert text.index("Baseline Stoikov Backtest") < text.index("ML Stoikov Extension Backtest")
    assert text.index("ML Stoikov Extension Backtest") < text.index("Oracle Stoikov Extension Backtest")


@pytest.mark.notebook
def test_exploration_ml_notebooks_use_project_root_data_path(project_root):
    ml_notebooks = [
        project_root / "exploration" / "directional_ML_methods.ipynb",
        project_root / "exploration" / "volatility_ML_methods.ipynb",
        project_root / "exploration" / "train_main_model_artifacts.ipynb",
    ]

    for path in ml_notebooks:
        text = notebook_source(path)
        assert "PROJECT_ROOT" in text
        assert 'DATA_DIR = PROJECT_ROOT / "data"' in text


@pytest.mark.notebook
def test_exploration_folder_contains_all_non_main_notebooks(project_root):
    exploration = sorted(path.name for path in (project_root / "exploration").glob("*.ipynb"))

    assert exploration == [
        "aapl_lobster_eda.ipynb",
        "directional_ML_methods.ipynb",
        "train_main_model_artifacts.ipynb",
        "volatility_ML_methods.ipynb",
    ]
