from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    tick_size: float = 0.01
    order_size: int = 100
    max_inventory: int = 1000
    fee_rate: float = 0.0
    queue_join_frac: float = 0.5

    col_time_s: str = "time_s"
    col_mid: str = "mid"
    col_bid1: str = "bid1"
    col_ask1: str = "ask1"
    col_bid1_sz: str = "bid1_sz"
    col_ask1_sz: str = "ask1_sz"
    col_bid2: str = "bid2"
    col_ask2: str = "ask2"
    col_bid2_sz: str = "bid2_sz"
    col_ask2_sz: str = "ask2_sz"

    col_evt_time: str = "time"
    col_evt_type: str = "event_type"
    col_evt_size: str = "size"
    col_evt_price: str = "price"
    col_evt_dir: str = "direction"
    exec_types: Set[int] = field(default_factory=lambda: {4, 5})

    session_seconds: float = 23400.0
    session_gap_seconds: float = 3600.0
    flatten_before_close_seconds: Optional[float] = 30.0 * 60.0
    force_flatten_at_session_end: bool = True
    liquidation_price: str = "touch"
    market_close_time_s: Optional[float] = None

    px_eps: float = 1e-8


def _round_to_tick(px: float, tick: float, side: str) -> float:
    if not np.isfinite(px):
        return np.nan
    if side == "bid":
        return float(np.floor(px / tick) * tick)
    return float(np.ceil(px / tick) * tick)


def _enforce_no_cross(bid: float, ask: float, tick: float) -> Tuple[float, float]:
    if not (np.isfinite(bid) and np.isfinite(ask)):
        return bid, ask
    if bid < ask:
        return bid, ask

    mid = 0.5 * (bid + ask)
    bid = _round_to_tick(mid - 0.5 * tick, tick, "bid")
    ask = _round_to_tick(mid + 0.5 * tick, tick, "ask")

    if bid >= ask:
        bid -= tick
        ask += tick

    return bid, ask


def _queue_start(depth: float, queue_join_frac: float) -> float:
    queue_join_frac = float(np.clip(queue_join_frac, 0.0, 1.0))
    return float(np.round(max(float(depth), 0.0) * queue_join_frac))


def _session_info(times: np.ndarray, cfg: BacktestConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    session_id = np.zeros(len(times), dtype=int)
    session_start = np.zeros(len(times), dtype=float)
    session_last_index = np.zeros(len(times), dtype=int)

    current_session = 0
    start_time = float(times[0]) if len(times) else 0.0

    for i, t in enumerate(times):
        if i > 0 and (t < times[i - 1] or (t - times[i - 1]) > cfg.session_gap_seconds):
            current_session += 1
            start_time = float(t)

        session_id[i] = current_session
        session_start[i] = start_time

    for sid in np.unique(session_id):
        rows = np.where(session_id == sid)[0]
        session_last_index[rows] = rows[-1]

    if cfg.market_close_time_s is None:
        session_close = session_start + cfg.session_seconds
    else:
        day_seconds = 24.0 * 60.0 * 60.0
        session_close = np.floor(times / day_seconds) * day_seconds + float(cfg.market_close_time_s)

    return session_id, session_close, session_last_index


def _liquidation_price(inv: float, mid: float, bid1: float, ask1: float, cfg: BacktestConfig) -> float:
    if str(cfg.liquidation_price).lower() == "mid":
        return float(mid)
    if inv > 0:
        return float(bid1)
    return float(ask1)


def _result_row(
    time_col: str,
    t: int,
    session_id: int,
    mid: float,
    bid1: float,
    ask1: float,
    quote_bid: float,
    quote_ask: float,
    buy_qty: float,
    buy_px: float,
    sell_qty: float,
    sell_px: float,
    forced_liq_qty: float,
    forced_liq_px: float,
    forced_liq_side: Optional[str],
    in_flatten_window: bool,
    state: Dict[str, float],
    fees_paid: float,
) -> Dict[str, object]:
    return {
        time_col: t,
        "session_id": int(session_id),
        "mid": mid,
        "bid1": bid1,
        "ask1": ask1,
        "quote_bid": quote_bid,
        "quote_ask": quote_ask,
        "buy_qty": buy_qty,
        "buy_px": buy_px if buy_qty > 0 else np.nan,
        "sell_qty": sell_qty,
        "sell_px": sell_px if sell_qty > 0 else np.nan,
        "forced_liq_qty": forced_liq_qty,
        "forced_liq_px": forced_liq_px,
        "forced_liq_side": forced_liq_side,
        "in_flatten_window": in_flatten_window,
        "inventory": state["inv"],
        "cash": state["cash"],
        "pnl": state["cash"] + state["inv"] * mid,
        "fees_paid": fees_paid,
    }


def run_backtest(
    df_1s: pd.DataFrame,
    df_evt: pd.DataFrame,
    quote_fn: Callable[[pd.Series, Dict, BacktestConfig], Tuple[float, float]],
    cfg: Optional[BacktestConfig] = None,
    initial_cash: float = 0.0,
    initial_inventory: float = 0.0,
) -> Tuple[pd.DataFrame, Dict]:
    if cfg is None:
        cfg = BacktestConfig()

    time_col = cfg.col_time_s
    df_1s = df_1s.sort_values(time_col).reset_index(drop=True)
    times = df_1s[time_col].astype(float).to_numpy()
    session_id, session_close, session_last_index = _session_info(times, cfg)

    df_evt = df_evt.sort_values(cfg.col_evt_time).reset_index(drop=True)
    if time_col not in df_evt.columns:
        df_evt[time_col] = np.floor(df_evt[cfg.col_evt_time]).astype(int)

    exec_events = df_evt[df_evt[cfg.col_evt_type].isin(cfg.exec_types)]
    event_groups = exec_events.groupby(time_col)

    state = {"inv": float(initial_inventory), "cash": float(initial_cash)}
    records = []
    fees_paid = 0.0

    queue_bid = None
    queue_ask = None
    last_bid = np.nan
    last_ask = np.nan

    flatten_on = cfg.flatten_before_close_seconds is not None
    if flatten_on:
        flatten_start = session_close - float(cfg.flatten_before_close_seconds)
    else:
        flatten_start = np.full(len(df_1s), np.inf)

    for row_i, row in df_1s.iterrows():
        t = int(row[time_col])
        mid = float(row[cfg.col_mid])
        bid1 = float(row[cfg.col_bid1])
        ask1 = float(row[cfg.col_ask1])

        in_flatten_window = bool(flatten_on and float(row[time_col]) >= flatten_start[row_i] - cfg.px_eps)
        is_final_row = bool(flatten_on and cfg.force_flatten_at_session_end and row_i == session_last_index[row_i])

        if in_flatten_window or is_final_row:
            queue_bid = None
            queue_ask = None
            last_bid = np.nan
            last_ask = np.nan

            forced_qty = 0.0
            forced_px = np.nan
            forced_side = None

            inv = float(state["inv"])
            if abs(inv) > 1e-12:
                forced_qty = abs(inv)
                forced_px = _liquidation_price(inv, mid, bid1, ask1, cfg)

                if inv > 0:
                    forced_side = "sell"
                    state["cash"] += forced_qty * forced_px
                    state["inv"] -= forced_qty
                else:
                    forced_side = "buy"
                    state["cash"] -= forced_qty * forced_px
                    state["inv"] += forced_qty

                fee = forced_qty * forced_px * cfg.fee_rate
                state["cash"] -= fee
                fees_paid += fee

            records.append(
                _result_row(
                    time_col, t, session_id[row_i], mid, bid1, ask1,
                    np.nan, np.nan,
                    0.0, 0.0, 0.0, 0.0,
                    forced_qty, forced_px, forced_side,
                    in_flatten_window, state, fees_paid,
                )
            )
            continue

        raw_bid, raw_ask = quote_fn(row, state, cfg)
        quote_bid = _round_to_tick(raw_bid, cfg.tick_size, "bid")
        quote_ask = _round_to_tick(raw_ask, cfg.tick_size, "ask")
        quote_bid, quote_ask = _enforce_no_cross(quote_bid, quote_ask, cfg.tick_size)

        inv = state["inv"]
        if inv + cfg.order_size > cfg.max_inventory:
            quote_bid = np.nan
        if inv - cfg.order_size < -cfg.max_inventory:
            quote_ask = np.nan

        if np.isfinite(quote_bid):
            if np.isnan(last_bid) or abs(quote_bid - last_bid) > cfg.px_eps:
                if quote_bid > bid1 + cfg.px_eps:
                    queue_bid = 0.0
                else:
                    depth = float(row[cfg.col_bid1_sz])
                    if cfg.col_bid2 in row and abs(quote_bid - float(row[cfg.col_bid2])) <= cfg.px_eps:
                        depth = float(row.get(cfg.col_bid2_sz, depth))
                    queue_bid = _queue_start(depth, cfg.queue_join_frac)
        else:
            queue_bid = None

        if np.isfinite(quote_ask):
            if np.isnan(last_ask) or abs(quote_ask - last_ask) > cfg.px_eps:
                if quote_ask < ask1 - cfg.px_eps:
                    queue_ask = 0.0
                else:
                    depth = float(row[cfg.col_ask1_sz])
                    if cfg.col_ask2 in row and abs(quote_ask - float(row[cfg.col_ask2])) <= cfg.px_eps:
                        depth = float(row.get(cfg.col_ask2_sz, depth))
                    queue_ask = _queue_start(depth, cfg.queue_join_frac)
        else:
            queue_ask = None

        last_bid = quote_bid
        last_ask = quote_ask

        buy_qty = 0.0
        sell_qty = 0.0
        buy_px = 0.0
        sell_px = 0.0
        buy_left = float(cfg.order_size)
        sell_left = float(cfg.order_size)
        bid_filled = False
        ask_filled = False

        if t in event_groups.groups:
            events = event_groups.get_group(t)

            sell_exec_px = events.loc[events[cfg.col_evt_dir].astype(int) == -1, cfg.col_evt_price]
            buy_exec_px = events.loc[events[cfg.col_evt_dir].astype(int) == 1, cfg.col_evt_price]

            min_sell_px = np.inf if len(sell_exec_px) == 0 else float(np.nanmin(sell_exec_px.to_numpy(dtype=float)))
            max_buy_px = -np.inf if len(buy_exec_px) == 0 else float(np.nanmax(buy_exec_px.to_numpy(dtype=float)))

            for _, ev in events.iterrows():
                if (np.isnan(quote_bid) or buy_left <= 0) and (np.isnan(quote_ask) or sell_left <= 0):
                    break

                ev_price = float(ev[cfg.col_evt_price])
                ev_size = float(ev[cfg.col_evt_size])
                ev_dir = int(ev[cfg.col_evt_dir])

                if ev_dir == -1 and np.isfinite(quote_bid):
                    if ev_price <= quote_bid + cfg.px_eps:
                        queue_bid = max(0.0, float(queue_bid) - ev_size)
                    if buy_left > 0 and not bid_filled and min_sell_px < quote_bid - cfg.px_eps and queue_bid <= 0:
                        buy_qty += buy_left
                        buy_px = quote_bid
                        buy_left = 0.0
                        bid_filled = True

                if ev_dir == 1 and np.isfinite(quote_ask):
                    if ev_price >= quote_ask - cfg.px_eps:
                        queue_ask = max(0.0, float(queue_ask) - ev_size)
                    if sell_left > 0 and not ask_filled and max_buy_px > quote_ask + cfg.px_eps and queue_ask <= 0:
                        sell_qty += sell_left
                        sell_px = quote_ask
                        sell_left = 0.0
                        ask_filled = True

        fee = 0.0
        if buy_qty > 0:
            state["inv"] += buy_qty
            state["cash"] -= buy_qty * buy_px
            fee += buy_qty * buy_px * cfg.fee_rate

        if sell_qty > 0:
            state["inv"] -= sell_qty
            state["cash"] += sell_qty * sell_px
            fee += sell_qty * sell_px * cfg.fee_rate

        state["cash"] -= fee
        fees_paid += fee

        records.append(
            _result_row(
                time_col, t, session_id[row_i], mid, bid1, ask1,
                quote_bid, quote_ask,
                buy_qty, buy_px, sell_qty, sell_px,
                0.0, np.nan, None,
                False, state, fees_paid,
            )
        )

    results = pd.DataFrame(records)
    metrics = {}

    if not results.empty:
        pnl = results["pnl"]
        pnl_change = pnl.diff().fillna(0.0)
        passive_volume = float(results["buy_qty"].sum() + results["sell_qty"].sum())
        liquidation_volume = float(results["forced_liq_qty"].sum())
        has_fill = (results["buy_qty"] > 0) | (results["sell_qty"] > 0)

        pnl_mean = float(pnl_change.mean())
        pnl_std = float(pnl_change.std(ddof=0))

        metrics["pnl_final"] = float(pnl.iloc[-1])
        metrics["pnl_std"] = pnl_std
        metrics["sharpe_1s"] = pnl_mean / pnl_std if pnl_std > 1e-12 else 0.0
        metrics["avg_abs_inventory"] = float(results["inventory"].abs().mean())
        metrics["fill_rate"] = float(has_fill.mean())
        metrics["volume"] = passive_volume + liquidation_volume

    return results, metrics


def plot_results(
    results_df: pd.DataFrame,
    title: str = "Backtest Results",
    trading_day_seconds: float = 23400.0,
) -> None:
    import matplotlib.pyplot as plt

    if results_df is None or results_df.empty:
        print("No results to plot.")
        return

    x = np.arange(len(results_df), dtype=float) / float(trading_day_seconds)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(x, results_df["pnl"], label="Cumulative PnL", color="green")
    ax1.set_title(f"{title} - Cumulative PnL")
    ax1.set_ylabel("PnL ($)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(x, results_df["inventory"], label="Inventory", color="blue")
    ax2.set_title("Inventory Position")
    ax2.set_xlabel("Trading days since start")
    ax2.set_ylabel("Inventory (shares)")
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()
