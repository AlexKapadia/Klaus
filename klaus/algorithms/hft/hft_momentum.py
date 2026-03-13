"""Ultra-simple HFT momentum — fires on any clear directional move.

No fancy conditions. If the last N bars moved in one direction with
sufficient magnitude, trade that direction. Designed to fire frequently
and rely on the risk manager for position sizing and stops.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class HFTMomentum(BaseAlgorithm):
    """Fires on any short-term directional move above a threshold.

    Looks at returns over the last N bars. If cumulative return
    exceeds a tiny threshold, fires a signal in that direction.
    """

    name = "hft_momentum"
    supported_instruments = ["XAUUSD", "XTIUSD", "XBRUSD", "XNGUSD", "XAGUSD"]
    preferred_regimes = [Regime.TRENDING, Regime.MEAN_REVERTING, Regime.VOLATILE]
    min_bars_required = 15

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = self.params.get("lookback", 5)
        self.threshold = self.params.get("threshold", 0.0003)
        self.strong_threshold = self.params.get("strong_threshold", 0.005)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.lookback + 5:
            return None

        close = data["close"]
        ret = (close.iloc[-1] - close.iloc[-self.lookback - 1]) / close.iloc[-self.lookback - 1]

        if np.isnan(ret):
            return None

        if abs(ret) < self.threshold:
            return None

        direction = Direction.LONG if ret > 0 else Direction.SHORT

        # Strength: logarithmic scaling — resists saturation on volatile instruments
        raw = abs(ret) / self.strong_threshold
        strength = np.clip(np.log1p(raw) / np.log1p(10.0), 0.05, 1.0)

        # Micro ATR for risk
        high_low = data["high"] - data["low"]
        micro_atr = high_low.tail(10).mean()

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "momentum_return": float(ret),
                "micro_atr": float(micro_atr) if not np.isnan(micro_atr) else 0.0,
                "hft_sl_mult": 0.25,
                "hft_tp_mult": 0.30,
            },
        )
