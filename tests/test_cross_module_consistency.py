from __future__ import annotations

import math

import pandas as pd

from feature_engineering.labelling import label_direction_future_avg
from strategies.stoikov import InventoryMMConfig, stoikov_quote_fn
from strategies.stoikov_extension import StoikovExtensionConfig, _decode_dir, stoikov_extension_quote_fn


def test_direction_labeler_uses_current_three_class_domain() -> None:
    df = pd.DataFrame({"Close": [100.0, 99.0, 101.0, 98.0, 102.0, 97.0, 103.0]})
    out = label_direction_future_avg(df, price_col="Close", horizon=1, balanced=True, verbose=False)
    assert set(out["y_dir"].unique()) == {0, 1, 2}


def test_extension_decodes_current_label_classes_by_default() -> None:
    assert _decode_dir(0, dir_encoding="class012") == -1
    assert _decode_dir(1, dir_encoding="class012") == 1
    assert _decode_dir(2, dir_encoding="class012") == 0


def test_down_labels_preserve_negative_direction_when_passed_to_extension() -> None:
    df = pd.DataFrame({"Close": [100.0, 90.0, 80.0]})
    out = label_direction_future_avg(df, price_col="Close", horizon=1, threshold=0.0, verbose=False)
    mapped = out["y_dir"].map(lambda v: _decode_dir(v, dir_encoding="class012"))
    assert (mapped == -1).all()


def test_baseline_and_extension_quote_factories_share_callable_shape() -> None:
    row = pd.Series({"time_s": 0.0, "mid": 100.0, "sig": 2.0, "dir": 1})
    state = {"inv": 0.0}
    cfg_bt = type("Cfg", (), {"col_time_s": "time_s"})()

    base_quote = stoikov_quote_fn(InventoryMMConfig(col_mid="mid", col_time="time_s"))
    ext_quote = stoikov_extension_quote_fn(
        StoikovExtensionConfig(col_mid="mid", col_time="time_s", col_sigma="sig", col_dir="dir")
    )

    assert len(base_quote(row, state, cfg_bt)) == 2
    assert len(ext_quote(row, state, cfg_bt)) == 2


def test_baseline_and_extension_default_time_decay_match_at_session_end() -> None:
    row_end = pd.Series({"time_s": 10.0, "mid": 100.0, "sig": 2.0, "dir": 0})
    state = {"inv": 0.0}
    cfg_bt = type("Cfg", (), {"col_time_s": "time_s"})()

    base_quote = stoikov_quote_fn(
        InventoryMMConfig(gamma=0.1, sigma=2.0, k=2.0, T=1.0, col_mid="mid", col_time="time_s", t0=0.0, t1=10.0)
    )
    ext_quote = stoikov_extension_quote_fn(
        StoikovExtensionConfig(
            gamma=0.1,
            k=2.0,
            T=1.0,
            col_mid="mid",
            col_time="time_s",
            col_sigma="sig",
            col_dir="dir",
            t0=0.0,
            t1=10.0,
        )
    )

    base_bid, base_ask = base_quote(row_end, state, cfg_bt)
    ext_bid, ext_ask = ext_quote(row_end, state, cfg_bt)
    assert math.isclose(base_ask - base_bid, ext_ask - ext_bid, rel_tol=0.0, abs_tol=1e-12)
