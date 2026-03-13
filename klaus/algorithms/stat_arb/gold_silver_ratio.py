"""Gold-silver ratio mean-reversion strategy."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class GoldSilverRatio(BaseAlgorithm):
    """Trades mean-reversion of the gold/silver price ratio.

    Historically ~60-80. When ratio is unusually high, silver is cheap
    relative to gold → long silver (or short gold). Vice versa.
    """

    name = "gold_silver_ratio"
    supported_instruments = ["XAUUSD", "XAGUSD"]
    preferred_regimes = [Regime.MEAN_REVERTING]
    min_bars_required = 130

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = self.params.get("lookback", 120)
        self.entry_z = self.params.get("entry_z", 2.0)
        self.exit_z = self.params.get("exit_z", 0.5)
        self._pair_data: Optional[pd.DataFrame] = None

    def set_pair_data(self, pair_df: pd.DataFrame) -> None:
        """Provide the other metal's data."""
        self._pair_data = pair_df

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if self._pair_data is None or self._pair_data.empty:
            return None

        if len(data) < self.lookback + 1 or len(self._pair_data) < self.lookback + 1:
            return None

        # Determine which is gold, which is silver
        gold_symbol = self.params.get("gold_symbol", "XAUUSD")
        silver_symbol = self.params.get("silver_symbol", "XAGUSD")

        if symbol == gold_symbol:
            gold_close = data["close"]
            silver_close = self._pair_data["close"]
        else:
            gold_close = self._pair_data["close"]
            silver_close = data["close"]

        # Align
        combined = pd.DataFrame({"gold": gold_close, "silver": silver_close}).dropna()
        if len(combined) < self.lookback:
            return None

        ratio = combined["gold"] / combined["silver"].replace(0, np.nan)
        ratio = ratio.dropna()

        rolling_mean = ratio.rolling(self.lookback).mean()
        rolling_std = ratio.rolling(self.lookback).std()

        z = (ratio.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

        if np.isnan(z):
            return None

        direction = None
        # High ratio → gold expensive / silver cheap
        if z > self.entry_z:
            # If we're trading silver, go long; if gold, go short
            direction = Direction.LONG if symbol == silver_symbol else Direction.SHORT
        elif z < -self.entry_z:
            # Low ratio → gold cheap / silver expensive
            direction = Direction.SHORT if symbol == silver_symbol else Direction.LONG
        else:
            return None

        strength = np.clip((abs(z) - self.entry_z) / 2.0, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={"gs_ratio": float(ratio.iloc[-1]), "gs_ratio_z": float(z)},
        )
