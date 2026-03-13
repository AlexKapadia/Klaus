"""Soybean crush spread strategy."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class SoybeanCrush(BaseAlgorithm):
    """Trades the soybean crush spread.

    Crush spread = soybean meal + soybean oil - soybeans.
    When spread deviates from mean, trade the reversion.
    Simplified: uses soybean price z-score as a proxy when
    full crush spread components aren't available.
    """

    name = "soybean_crush"
    supported_instruments = ["SOYBEAN"]
    preferred_regimes = [Regime.MEAN_REVERTING]
    min_bars_required = 70

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = self.params.get("lookback", 60)
        self.entry_z = self.params.get("entry_z", 2.0)
        self.exit_z = self.params.get("exit_z", 0.5)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.lookback + 1:
            return None

        close = data["close"]
        rolling_mean = close.rolling(self.lookback).mean()
        rolling_std = close.rolling(self.lookback).std()

        z = (close.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

        if np.isnan(z):
            return None

        direction = None
        if z < -self.entry_z:
            direction = Direction.LONG   # cheap relative to crush spread
        elif z > self.entry_z:
            direction = Direction.SHORT  # expensive relative to crush spread
        else:
            return None

        strength = np.clip((abs(z) - self.entry_z) / 2.0, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={"crush_z": float(z)},
        )
