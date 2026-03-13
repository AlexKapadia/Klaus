"""Bollinger Band mean reversion for forex."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from stepsister.core.registry import register_fx_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_fx_algorithm
class FXBollinger(BaseAlgorithm):
    """Bollinger Band mean-reversion on forex pairs.

    Entry when price moves beyond the Bollinger Band:
    - Below lower band → LONG (expect reversion up)
    - Above upper band → SHORT (expect reversion down)

    Exit logic delegated to trailing stop / TP.
    """

    name = "fx_bollinger"
    supported_instruments = ["*"]
    preferred_regimes = [Regime.MEAN_REVERTING]
    min_bars_required = 30

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.period = self.params.get("period", 20)
        self.num_std = self.params.get("num_std", 2.0)
        self.exit_at_mean = self.params.get("exit_at_mean", True)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.period + 2:
            return None

        close = data["close"]
        sma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()

        upper = sma + self.num_std * std
        lower = sma - self.num_std * std

        current_price = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_sma = sma.iloc[-1]

        if any(np.isnan(v) for v in [current_price, current_upper, current_lower, current_sma]):
            return None

        band_width = current_upper - current_lower
        if band_width <= 0:
            return None

        direction = None
        if current_price < current_lower:
            direction = Direction.LONG
        elif current_price > current_upper:
            direction = Direction.SHORT
        else:
            return None

        # Strength from how far beyond the band
        if direction == Direction.LONG:
            excess = (current_lower - current_price) / band_width
        else:
            excess = (current_price - current_upper) / band_width

        strength = np.clip(excess * 2.0 + 0.3, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "bb_upper": float(current_upper),
                "bb_lower": float(current_lower),
                "bb_middle": float(current_sma),
                "bb_excess": float(excess),
            },
        )
