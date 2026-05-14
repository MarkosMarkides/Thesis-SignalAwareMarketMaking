from dataclasses import dataclass
import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StoikovExtensionConfig:
    gamma: float = 0.1
    k: float = 1.5
    T: float = 1.0
    sigma0: float = float("nan")
    alpha: float = 1.0
    q_max: Optional[int] = None
    skew_ticks: float = 1.0
    tick_size: Optional[float] = None
    sigma_is_log: bool = False
    sigma_horizon: int = 1
    col_mid: str = "mid"
    col_time: str = "time_s"
    col_sigma: str = "y_vol"
    col_dir: str = "y_dir"
    t0: Optional[float] = None
    t1: Optional[float] = None


class PredictedVolatilityMarketMaker:
    def __init__(self, cfg: StoikovExtensionConfig):
        self.cfg = cfg
        self._t0: Optional[float] = None
        self._t1: Optional[float] = None

    def reservation_price(self, s: float, q: int, t: float, rho_eff: float, skew: float) -> float:
        c = self.cfg
        tau = max(c.T - t, 0.0)
        return s - q * c.gamma * (c.sigma0 ** 2) * tau * rho_eff + skew

    def total_spread(self, t: float) -> float:
        c = self.cfg
        tau = max(c.T - t, 0.0)

        if c.gamma == 0:
            return 2.0 / c.k

        return c.gamma * (c.sigma0 ** 2) * tau + (2.0 / c.gamma) * math.log(1.0 + c.gamma / c.k)

    def quotes(self, s: float, q: int, t: float, sigma_hat: float, skew: float) -> Dict[str, float]:
        c = self.cfg

        rho = (sigma_hat / c.sigma0) ** 2
        rho_eff = 1.0 + c.alpha * (rho - 1.0)

        r = self.reservation_price(s, q, t, rho_eff, skew)
        spread = self.total_spread(t)

        p_bid = r - 0.5 * spread
        p_ask = r + 0.5 * spread

        delta_b = max(1e-12, s - p_bid)
        delta_a = max(1e-12, p_ask - s)

        if c.q_max is not None:
            if q >= c.q_max:
                p_bid = -np.inf
                delta_b = float("inf")
            if q <= -c.q_max:
                p_ask = np.inf
                delta_a = float("inf")

        return {
            "p_bid": p_bid,
            "p_ask": p_ask,
            "r": r,
            "spread": spread,
            "delta_b": delta_b,
            "delta_a": delta_a,
            "rho": rho,
            "rho_eff": rho_eff,
        }

    def set_time_range(self, t0: float, t1: float) -> None:
        self._t0 = float(t0)
        self._t1 = float(t1)

    def time_from_row(self, row: pd.Series, cfg_time_col: str = "time_s") -> float:
        if cfg_time_col not in row:
            return 0.0

        ts = float(row[cfg_time_col])
        t0 = self.cfg.t0 if self.cfg.t0 is not None else self._t0
        t1 = self.cfg.t1 if self.cfg.t1 is not None else self._t1

        if t0 is None or t1 is None or t1 <= t0:
            return 0.0

        frac = (ts - t0) / (t1 - t0)
        frac = max(0.0, min(1.0, frac))
        return frac * self.cfg.T

    def quote_from_state(
        self,
        mid: float,
        inventory: float,
        t: float,
        sigma_hat: float,
        skew: float,
    ) -> Dict[str, float]:
        t = max(0.0, min(self.cfg.T, t))
        q = int(round(float(inventory)))
        return self.quotes(mid, q, t, sigma_hat, skew)


def stoikov_extension_quote_fn(
    ext_cfg: StoikovExtensionConfig,
    session_seconds: float = 23400.0,
    session_gap_seconds: float = 3600.0,
) -> Callable[[pd.Series, Dict[str, float], object], Tuple[float, float]]:
    mm = PredictedVolatilityMarketMaker(ext_cfg)
    prev_ts: Optional[float] = None

    def quote_fn(row: pd.Series, state: Dict[str, float], cfg_bt: object) -> Tuple[float, float]:
        nonlocal prev_ts

        mid = float(row[ext_cfg.col_mid])
        sigma_hat = abs(float(row[ext_cfg.col_sigma]))

        if ext_cfg.sigma_horizon > 1:
            sigma_hat = sigma_hat / math.sqrt(float(ext_cfg.sigma_horizon))

        if ext_cfg.sigma_is_log:
            sigma_hat = sigma_hat * abs(mid)

        dir_hat = int(round(float(row.get(ext_cfg.col_dir, 0))))

        tick_size = getattr(cfg_bt, "tick_size", None)
        if tick_size is None:
            tick_size = ext_cfg.tick_size
        if tick_size is None:
            tick_size = 0.01

        skew = float(ext_cfg.skew_ticks) * float(tick_size) * float(dir_hat)
        q = int(round(float(state.get("inv", 0.0))))
        time_col = getattr(cfg_bt, "col_time_s", ext_cfg.col_time) or ext_cfg.col_time

        if ext_cfg.t0 is None or ext_cfg.t1 is None:
            ts = float(row[time_col])
            new_session = prev_ts is None or ts < prev_ts or (ts - prev_ts) > session_gap_seconds
            if new_session:
                mm.set_time_range(ts, ts + session_seconds)
            prev_ts = ts

        t = mm.time_from_row(row, time_col)
        out = mm.quote_from_state(mid, q, t, sigma_hat, skew)
        return float(out["p_bid"]), float(out["p_ask"])

    return quote_fn
