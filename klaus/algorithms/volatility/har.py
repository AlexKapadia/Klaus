"""Heterogeneous Autoregressive (HAR) volatility model."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class HARModel(BaseAlgorithm):
    """HAR model for realised volatility forecasting.

    RV_t = alpha + beta_d * RV_d + beta_w * RV_w + beta_m * RV_m

    Uses daily, weekly, monthly realised volatility components to
    forecast next-period vol. Generates signals based on whether
    forecast vol is above or below current vol.
    """

    name = "har"
    supported_instruments = ["XCUUSD", "XAUUSD", "XTIUSD"]
    preferred_regimes = [Regime.TRENDING, Regime.MEAN_REVERTING]
    min_bars_required = 50

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.daily_lag = self.params.get("daily_lag", 1)
        self.weekly_lag = self.params.get("weekly_lag", 5)
        self.monthly_lag = self.params.get("monthly_lag", 22)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.monthly_lag + 10:
            return None

        returns = data["close"].pct_change().dropna()

        if len(returns) < self.monthly_lag + 5:
            return None

        # Compute realised volatility components
        rv_daily = returns.rolling(self.daily_lag).std()
        rv_weekly = returns.rolling(self.weekly_lag).std()
        rv_monthly = returns.rolling(self.monthly_lag).std()

        # Build regression features
        features = pd.DataFrame({
            "rv_d": rv_daily.shift(1),
            "rv_w": rv_weekly.shift(1),
            "rv_m": rv_monthly.shift(1),
            "rv_target": rv_daily,
        }).dropna()

        if len(features) < 30:
            return None

        # Simple OLS for HAR
        X = features[["rv_d", "rv_w", "rv_m"]].values
        y = features["rv_target"].values

        # Add intercept
        X_aug = np.column_stack([np.ones(len(X)), X])

        try:
            beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        # Forecast next-period vol
        latest_features = np.array([
            1.0,
            rv_daily.iloc[-1],
            rv_weekly.iloc[-1],
            rv_monthly.iloc[-1],
        ])

        forecast_vol = float(latest_features @ beta)
        current_vol = rv_daily.iloc[-1]

        if current_vol <= 0 or np.isnan(forecast_vol):
            return None

        vol_change = (forecast_vol - current_vol) / current_vol

        direction = None
        # Vol decreasing → trending/calmer → favour long
        if vol_change < -0.1:
            direction = Direction.LONG
        # Vol increasing → more volatile → favour short / caution
        elif vol_change > 0.1:
            direction = Direction.SHORT
        else:
            return None

        strength = np.clip(abs(vol_change), 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={"forecast_vol": forecast_vol, "current_vol": float(current_vol), "vol_change": float(vol_change)},
        )
