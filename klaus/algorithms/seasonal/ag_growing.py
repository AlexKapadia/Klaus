"""Agricultural growing season strategy."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class AgGrowing(BaseAlgorithm):
    """Seasonal strategy for agricultural commodities.

    - Planting season (Apr-Jun): uncertainty → typically bullish
    - Growing season (Jul-Aug): weather-dependent, neutral
    - Harvest season (Sep-Nov): supply increase → bearish pressure
    - Winter (Dec-Mar): storage/demand dynamics

    Combines seasonal bias with volatility and trend filters.
    """

    name = "ag_growing"
    supported_instruments = ["CORN", "SOYBEAN", "WHEAT"]
    preferred_regimes = [Regime.MEAN_REVERTING, Regime.TRENDING]
    min_bars_required = 50

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.planting_start = self.params.get("planting_start_month", 4)
        self.planting_end = self.params.get("planting_end_month", 6)
        self.harvest_start = self.params.get("harvest_start_month", 9)
        self.harvest_end = self.params.get("harvest_end_month", 11)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.min_bars_required:
            return None

        now = datetime.utcnow()
        month = now.month

        seasonal_bias = self._get_seasonal_bias(month)
        if seasonal_bias == 0:
            return None

        # Trend confirmation
        close = data["close"]
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean() if len(data) >= 50 else sma_20

        if np.isnan(sma_20.iloc[-1]):
            return None

        trend_up = close.iloc[-1] > sma_20.iloc[-1]
        trend_down = close.iloc[-1] < sma_20.iloc[-1]

        direction = None
        strength = 0.3

        if seasonal_bias > 0 and trend_up:
            direction = Direction.LONG
            strength = 0.5
        elif seasonal_bias < 0 and trend_down:
            direction = Direction.SHORT
            strength = 0.5
        elif seasonal_bias > 0:
            direction = Direction.LONG
            strength = 0.25
        elif seasonal_bias < 0:
            direction = Direction.SHORT
            strength = 0.25

        if direction is None:
            return None

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={"month": month, "seasonal_bias": seasonal_bias, "season": self._season_name(month)},
        )

    def _get_seasonal_bias(self, month: int) -> int:
        if self.planting_start <= month <= self.planting_end:
            return 1   # planting uncertainty → bullish
        if self.harvest_start <= month <= self.harvest_end:
            return -1  # harvest supply → bearish
        return 0

    @staticmethod
    def _season_name(month: int) -> str:
        if 4 <= month <= 6:
            return "planting"
        if 7 <= month <= 8:
            return "growing"
        if 9 <= month <= 11:
            return "harvest"
        return "winter"
