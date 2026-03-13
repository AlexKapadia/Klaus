"""Z-score mean reversion for forex."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from stepsister.core.registry import register_fx_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_fx_algorithm
class FXZScore(BaseAlgorithm):
    """Z-score mean-reversion on forex pairs.

    Computes a rolling z-score of the closing price and trades
    reversion when the z-score exceeds the entry threshold.
    """

    name = "fx_zscore"
    supported_instruments = ["*"]
    preferred_regimes = [Regime.MEAN_REVERTING]
    min_bars_required = 70

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = self.params.get("lookback", 60)
        self.entry_threshold = self.params.get("entry_threshold", 2.0)
        self.exit_threshold = self.params.get("exit_threshold", 0.5)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.lookback + 5:
            return None

        close = data["close"]
        rolling_mean = close.rolling(self.lookback).mean()
        rolling_std = close.rolling(self.lookback).std()

        current_z = (close.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

        if np.isnan(current_z):
            return None

        if abs(current_z) < self.entry_threshold:
            return None

        # Trade against the deviation
        direction = Direction.SHORT if current_z > self.entry_threshold else Direction.LONG

        # Strength from z-score excess
        z_excess = (abs(current_z) - self.entry_threshold) / 2.0
        strength = np.clip(z_excess + 0.2, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "zscore": float(current_z),
                "rolling_mean": float(rolling_mean.iloc[-1]),
            },
        )
