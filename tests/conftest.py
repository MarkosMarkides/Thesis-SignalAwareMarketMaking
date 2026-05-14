import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_PATHS = [
    PROJECT_ROOT / "main.ipynb",
    PROJECT_ROOT / "exploration" / "aapl_lobster_eda.ipynb",
    PROJECT_ROOT / "exploration" / "directional_ML_methods.ipynb",
    PROJECT_ROOT / "exploration" / "volatility_ML_methods.ipynb",
    PROJECT_ROOT / "exploration" / "train_main_model_artifacts.ipynb",
]


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def artifact_dir(project_root):
    return project_root / "artifacts" / "main_models"


@pytest.fixture
def notebook_paths():
    return NOTEBOOK_PATHS


def notebook_json(path):
    return json.loads(Path(path).read_text())


def notebook_source(path):
    nb = notebook_json(path)
    parts = []
    for cell in nb["cells"]:
        src = cell.get("source", "")
        parts.append("".join(src) if isinstance(src, list) else src)
    return "\n".join(parts)


@pytest.fixture
def empty_events():
    return pd.DataFrame(columns=["time", "event_type", "size", "price", "direction"])


@pytest.fixture
def small_book():
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "mid": [100.0, 100.0, 100.0, 100.0],
            "bid1": [99.99, 99.99, 99.99, 99.99],
            "ask1": [100.01, 100.01, 100.01, 100.01],
            "bid1_sz": [100.0, 100.0, 100.0, 100.0],
            "ask1_sz": [100.0, 100.0, 100.0, 100.0],
            "bid2": [99.98, 99.98, 99.98, 99.98],
            "ask2": [100.02, 100.02, 100.02, 100.02],
            "bid2_sz": [200.0, 200.0, 200.0, 200.0],
            "ask2_sz": [200.0, 200.0, 200.0, 200.0],
        }
    )


def make_book(
    times,
    mid=100.0,
    bid1=99.99,
    ask1=100.01,
    bid1_sz=100.0,
    ask1_sz=100.0,
    bid2=99.98,
    ask2=100.02,
    bid2_sz=200.0,
    ask2_sz=200.0,
):
    n = len(times)
    return pd.DataFrame(
        {
            "time_s": list(times),
            "mid": np.full(n, mid, dtype=float),
            "bid1": np.full(n, bid1, dtype=float),
            "ask1": np.full(n, ask1, dtype=float),
            "bid1_sz": np.full(n, bid1_sz, dtype=float),
            "ask1_sz": np.full(n, ask1_sz, dtype=float),
            "bid2": np.full(n, bid2, dtype=float),
            "ask2": np.full(n, ask2, dtype=float),
            "bid2_sz": np.full(n, bid2_sz, dtype=float),
            "ask2_sz": np.full(n, ask2_sz, dtype=float),
        }
    )


def make_events(rows):
    return pd.DataFrame(rows, columns=["time", "event_type", "size", "price", "direction"])


def no_quotes(row, state, cfg):
    return np.nan, np.nan


def fixed_quotes(bid, ask):
    def quote_fn(row, state, cfg):
        return bid, ask

    return quote_fn


def assert_dict_close(actual, expected):
    for key, value in expected.items():
        assert key in actual
        if isinstance(value, float):
            assert actual[key] == pytest.approx(value)
        else:
            assert actual[key] == value

