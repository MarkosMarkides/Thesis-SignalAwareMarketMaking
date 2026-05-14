import numpy as np
import pandas as pd


class VolatilityLabeler:
    def __init__(self, close_col: str = "Close", horizon: int = 12, out_col: str = "y_vol") -> None:
        self.close_col = close_col
        self.horizon = int(horizon)
        self.out_col = out_col

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        log_price = np.log(out[self.close_col].astype(float))
        log_return = log_price.diff()

        future_variance = (
            (log_return ** 2)
            .rolling(self.horizon, min_periods=self.horizon)
            .sum()
            .shift(-self.horizon)
        )

        out[self.out_col] = np.sqrt(future_variance).fillna(0.0)
        return out


def label_volatility(
    df: pd.DataFrame,
    close_col: str = "Close",
    horizon: int = 12,
    out_col: str = "y_vol",
) -> pd.DataFrame:
    return VolatilityLabeler(close_col=close_col, horizon=horizon, out_col=out_col).label(df)


class DirectionLabelerFutureAvg:
    def __init__(
        self,
        price_col: str = "Close",
        vol_col: str | None = None,
        horizon: int = 12,
        out_col: str = "y_dir_avg",
        avg_mode: str = "vwap",
        threshold: float = 0.0,
        balanced: bool = False,
        verbose: bool = True,
    ) -> None:
        self.price_col = price_col
        self.vol_col = vol_col
        self.horizon = int(horizon)
        self.out_col = out_col
        self.avg_mode = avg_mode
        self.threshold = threshold
        self.balanced = balanced
        self.verbose = bool(verbose)

    def _forward_mean(self, df: pd.DataFrame) -> pd.Series:
        price = df[self.price_col].astype(float)
        return price.shift(-1).rolling(self.horizon, min_periods=self.horizon).mean().shift(-(self.horizon - 1))

    def _forward_vwap(self, df: pd.DataFrame) -> pd.Series:
        if self.vol_col is None or self.vol_col not in df.columns:
            return self._forward_mean(df)

        price = df[self.price_col].astype(float).shift(-1)
        volume = df[self.vol_col].astype(float).shift(-1)

        numerator = (price * volume).rolling(self.horizon, min_periods=self.horizon).sum().shift(-(self.horizon - 1))
        denominator = volume.rolling(self.horizon, min_periods=self.horizon).sum().shift(-(self.horizon - 1))

        return (numerator / denominator.replace(0.0, np.nan)).fillna(self._forward_mean(df))

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if self.avg_mode == "vwap":
            future_price = self._forward_vwap(out)
        else:
            future_price = self._forward_mean(out)

        current_price = out[self.price_col].astype(float)
        future_return = np.log(future_price / current_price.replace(0.0, np.nan))

        if self.balanced:
            down_cutoff = np.nanquantile(future_return, 1 / 3)
            up_cutoff = np.nanquantile(future_return, 2 / 3)
            down = future_return < down_cutoff
            up = future_return > up_cutoff
        else:
            down = future_return < -self.threshold
            up = future_return > self.threshold

        labels = np.select([down, up], [0, 1], default=2)
        valid = future_return.notna()

        out = out.loc[valid].copy()
        out[self.out_col] = labels[valid].astype(int)

        if self.verbose:
            counts = out[self.out_col].value_counts()
            total = float(counts.sum())
            pct_0 = 100.0 * float(counts.get(0, 0)) / total if total else 0.0
            pct_1 = 100.0 * float(counts.get(1, 0)) / total if total else 0.0
            pct_2 = 100.0 * float(counts.get(2, 0)) / total if total else 0.0
            print(f"[{self.out_col}] label %: 0={pct_0:.2f}%  1={pct_1:.2f}%  2={pct_2:.2f}%  (n={int(total)})")

        return out


def label_direction_future_avg(
    df: pd.DataFrame,
    price_col: str = "Close",
    vol_col: str | None = None,
    horizon: int = 12,
    out_col: str = "y_dir",
    avg_mode: str = "mean",
    threshold: float = 0.0,
    balanced: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    return DirectionLabelerFutureAvg(
        price_col=price_col,
        vol_col=vol_col,
        horizon=horizon,
        out_col=out_col,
        avg_mode=avg_mode,
        threshold=threshold,
        balanced=balanced,
        verbose=verbose,
    ).label(df)
