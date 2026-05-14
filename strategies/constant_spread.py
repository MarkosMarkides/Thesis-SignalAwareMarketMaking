from typing import Callable, Dict, Tuple

import pandas as pd


def constant_spread_strategy(
    constant: float,
    tick_size: float = 0.01,
    col_bid1: str = "bid1",
    col_ask1: str = "ask1",
) -> Callable[[pd.Series, Dict[str, float], object], Tuple[float, float]]:
    constant = float(constant)
    tick_size = float(tick_size)

    def quote_fn(row: pd.Series, state: Dict[str, float], cfg: object) -> Tuple[float, float]:
        bid1 = float(row[col_bid1])
        ask1 = float(row[col_ask1])

        mid = 0.5 * (bid1 + ask1)
        bid = mid - constant
        ask = mid + constant

        l1_spread = ask1 - bid1
        my_spread = ask - bid

        if my_spread < l1_spread:
            target_spread = max(l1_spread - tick_size, tick_size)
            bid = mid - 0.5 * target_spread
            ask = mid + 0.5 * target_spread

        return bid, ask

    return quote_fn
