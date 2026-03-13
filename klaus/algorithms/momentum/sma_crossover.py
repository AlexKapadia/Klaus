"""SMA Crossover with volatility scaling."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class SMACrossover(BaseAlgorithm):
    """Simple Moving Average crossover strategy.

    - LONG when fast SMA crosses above slow SMA
    - SHORT when fast SMA crosses below slow SMA
    - Signal strength scaled by normalised distance between SMAs
    - Optional volatility scaling: reduce strength in high-vol environments
    """

    name = "sma_crossover"
    supported_instruments = ["*"]
    preferred_regimes = [Regime.TRENDING]
    min_bars_required = 60

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.fast_period = self.params.get("fast_period", 20)
        self.slow_period = self.params.get("slow_period", 50)
        self.vol_scaling = self.params.get("vol_scaling", True)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.slow_period + 2:
            return None

        close = data["close"]
        fast_sma = close.rolling(self.fast_period).mean()
        slow_sma = close.rolling(self.slow_period).mean()

        if fast_sma.iloc[-1] is np.nan or slow_sma.iloc[-1] is np.nan:
            return None

        # Detect crossover
        prev_diff = fast_sma.iloc[-2] - slow_sma.iloc[-2]
        curr_diff = fast_sma.iloc[-1] - slow_sma.iloc[-1]

        if np.isnan(prev_diff) or np.isnan(curr_diff):
            return None

        # No crossover
        if not ((prev_diff <= 0 and curr_diff > 0) or (prev_diff >= 0 and curr_diff < 0)):
            return None

        direction = Direction.LONG if curr_diff > 0 else Direction.SHORT

        # Strength = normalised distance
        spread_pct = abs(curr_diff) / close.iloc[-1]
        strength = min(spread_pct * 100, 1.0)  # cap at 1.0

        # Volatility scaling
        if self.vol_scaling and "rolling_volatility" in data.columns:
            vol = data["rolling_volatility"].iloc[-1]
            if not np.isnan(vol):
                median_vol = data["rolling_volatility"].median()
                if median_vol > 0:
                    vol_ratio = vol / median_vol
                    strength = strength / max(vol_ratio, 1.0)

        strength = np.clip(strength, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={"fast_sma": float(fast_sma.iloc[-1]), "slow_sma": float(slow_sma.iloc[-1])},
        )
