"""Bollinger Bands mean-reversion strategy."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class BollingerBands(BaseAlgorithm):
    """Mean-reversion using Bollinger Bands.

    - LONG when price touches/crosses below lower band
    - SHORT when price touches/crosses above upper band
    - Optional: exit at middle band
    - Strength proportional to band penetration depth
    """

    name = "bollinger"
    supported_instruments = ["*"]
    preferred_regimes = [Regime.MEAN_REVERTING]
    min_bars_required = 30

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.period = self.params.get("period", 20)
        self.num_std = self.params.get("num_std", 2.0)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.period + 1:
            return None

        close = data["close"]
        sma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()

        upper = sma + self.num_std * std
        lower = sma - self.num_std * std

        current_price = close.iloc[-1]
        prev_price = close.iloc[-2]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        prev_upper = upper.iloc[-2]
        prev_lower = lower.iloc[-2]

        if any(np.isnan(v) for v in [current_upper, current_lower, prev_upper, prev_lower]):
            return None

        band_width = current_upper - current_lower
        if band_width <= 0:
            return None

        direction = None
        penetration = 0.0

        # Price crossed below lower band → LONG (mean reversion up)
        if current_price <= current_lower and prev_price > prev_lower:
            direction = Direction.LONG
            penetration = (current_lower - current_price) / band_width

        # Price crossed above upper band → SHORT (mean reversion down)
        elif current_price >= current_upper and prev_price < prev_upper:
            direction = Direction.SHORT
            penetration = (current_price - current_upper) / band_width

        if direction is None:
            return None

        strength = np.clip(0.5 + penetration * 2, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "bb_upper": float(current_upper),
                "bb_lower": float(current_lower),
                "bb_middle": float(sma.iloc[-1]),
            },
        )
