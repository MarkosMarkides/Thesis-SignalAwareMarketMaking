from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "feature_engineering.dollar_value_bars",
        "feature_engineering.indicators",
        "feature_engineering.labelling",
        "strategies.constant_spread",
        "strategies.stoikov",
        "strategies.stoikov_extension",
        "backtesting.backtesting",
    ],
)
def test_repo_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


def test_main_import_fails_with_missing_data_collection() -> None:
    sys.modules.pop("main", None)
    with pytest.raises(ModuleNotFoundError, match="data.data_collection"):
        importlib.import_module("main")
