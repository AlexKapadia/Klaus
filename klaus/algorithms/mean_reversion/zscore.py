"""Z-score mean reversion strategy."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class ZScoreReversion(BaseAlgorithm):
    """Mean-reversion based on rolling z-score of price.

    - LONG when z-score < -entry_threshold (price unusually low)
    - SHORT when z-score > +entry_threshold (price unusually high)
    - Signal strength proportional to |z-score| beyond threshold
    """

    name = "zscore_reversion"
    supported_instruments = ["*"]
    preferred_regimes = [Regime.MEAN_REVERTING]
    min_bars_required = 70

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = self.params.get("lookback", 60)
        self.entry_threshold = self.params.get("entry_threshold", 2.0)
        self.exit_threshold = self.params.get("exit_threshold", 0.5)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.lookback + 1:
            return None

        close = data["close"]
        rolling_mean = close.rolling(self.lookback).mean()
        rolling_std = close.rolling(self.lookback).std()

        current_z = (close.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

        if np.isnan(current_z):
            return None

        direction = None
        if current_z < -self.entry_threshold:
            direction = Direction.LONG
        elif current_z > self.entry_threshold:
            direction = Direction.SHORT
        else:
            return None

        # Strength increases with deviation beyond threshold
        excess = abs(current_z) - self.entry_threshold
        strength = np.clip(excess / 2.0, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={"zscore": float(current_z)},
        )
