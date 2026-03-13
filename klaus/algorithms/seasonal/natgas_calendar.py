"""Natural gas seasonal calendar strategy."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class NatGasCalendar(BaseAlgorithm):
    """Seasonal calendar strategy for natural gas.

    Natural gas follows injection/withdrawal cycles:
    - Injection season (Apr-Oct): storage builds → bearish pressure
    - Withdrawal season (Nov-Mar): storage draws → bullish pressure

    Combines seasonal bias with trend confirmation (SMA).
    """

    name = "natgas_calendar"
    supported_instruments = ["XNGUSD"]
    preferred_regimes = [Regime.TRENDING, Regime.MEAN_REVERTING]
    min_bars_required = 50

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.injection_start = self.params.get("injection_start_month", 4)
        self.injection_end = self.params.get("injection_end_month", 10)
        self.withdrawal_start = self.params.get("withdrawal_start_month", 11)
        self.withdrawal_end = self.params.get("withdrawal_end_month", 3)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.min_bars_required:
            return None

        now = datetime.utcnow()
        month = now.month

        # Determine seasonal bias
        seasonal_bias = self._get_seasonal_bias(month)
        if seasonal_bias == 0:
            return None

        # Confirm with simple trend filter (20-SMA)
        close = data["close"]
        sma_20 = close.rolling(20).mean()

        if np.isnan(sma_20.iloc[-1]):
            return None

        trend_bullish = close.iloc[-1] > sma_20.iloc[-1]
        trend_bearish = close.iloc[-1] < sma_20.iloc[-1]

        direction = None
        strength = 0.4  # base seasonal strength

        if seasonal_bias > 0 and trend_bullish:
            # Withdrawal season + bullish trend
            direction = Direction.LONG
            strength = 0.6
        elif seasonal_bias < 0 and trend_bearish:
            # Injection season + bearish trend
            direction = Direction.SHORT
            strength = 0.6
        elif seasonal_bias > 0:
            # Withdrawal season but no trend confirmation — weaker signal
            direction = Direction.LONG
            strength = 0.3
        elif seasonal_bias < 0:
            direction = Direction.SHORT
            strength = 0.3

        if direction is None:
            return None

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={"month": month, "seasonal_bias": seasonal_bias},
        )

    def _get_seasonal_bias(self, month: int) -> int:
        """Return +1 (bullish/withdrawal), -1 (bearish/injection), or 0."""
        # Withdrawal: Nov-Mar
        if month >= self.withdrawal_start or month <= self.withdrawal_end:
            return 1
        # Injection: Apr-Oct
        if self.injection_start <= month <= self.injection_end:
            return -1
        return 0
