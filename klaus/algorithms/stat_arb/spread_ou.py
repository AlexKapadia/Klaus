"""Ornstein-Uhlenbeck spread trading (e.g., WTI vs Brent)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class SpreadOU(BaseAlgorithm):
    """Pairs/spread trading using the OU model.

    Trades the spread between two correlated instruments (e.g., WTI-Brent).
    Enter when spread z-score exceeds entry threshold; exit at exit threshold.
    Validates mean-reversion via half-life test.
    """

    name = "spread_ou"
    supported_instruments = ["XTIUSD", "XBRUSD"]
    preferred_regimes = [Regime.MEAN_REVERTING]
    min_bars_required = 70

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = self.params.get("lookback", 60)
        self.entry_z = self.params.get("entry_z", 2.0)
        self.exit_z = self.params.get("exit_z", 0.5)
        self.half_life_max = self.params.get("half_life_max", 30)
        self._pair_data: Optional[pd.DataFrame] = None

    def set_pair_data(self, pair_df: pd.DataFrame) -> None:
        """Provide the other leg's OHLCV data for spread computation."""
        self._pair_data = pair_df

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if self._pair_data is None or self._pair_data.empty:
            return None

        if len(data) < self.lookback + 1 or len(self._pair_data) < self.lookback + 1:
            return None

        # Align on common index
        combined = pd.DataFrame({
            "a": data["close"],
            "b": self._pair_data["close"],
        }).dropna()

        if len(combined) < self.lookback:
            return None

        spread = combined["a"] - combined["b"]

        # Check half-life of mean reversion
        half_life = self._compute_half_life(spread.values)
        if half_life is None or half_life > self.half_life_max or half_life < 1:
            return None

        # Z-score of spread
        rolling_mean = spread.rolling(self.lookback).mean()
        rolling_std = spread.rolling(self.lookback).std()
        z = (spread.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

        if np.isnan(z):
            return None

        direction = None
        if z < -self.entry_z:
            direction = Direction.LONG   # spread too low, buy A sell B
        elif z > self.entry_z:
            direction = Direction.SHORT  # spread too high, sell A buy B
        else:
            return None

        strength = np.clip((abs(z) - self.entry_z) / 2.0, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={"spread_z": float(z), "half_life": float(half_life)},
        )

    @staticmethod
    def _compute_half_life(spread: np.ndarray) -> Optional[float]:
        """Estimate the half-life of mean reversion via AR(1) regression."""
        if len(spread) < 10:
            return None

        lag = spread[:-1]
        diff = np.diff(spread)

        # OLS: diff = alpha + beta * lag
        lag_mean = lag.mean()
        diff_mean = diff.mean()

        numerator = np.sum((lag - lag_mean) * (diff - diff_mean))
        denominator = np.sum((lag - lag_mean) ** 2)

        if denominator == 0:
            return None

        beta = numerator / denominator

        if beta >= 0:
            return None  # Not mean-reverting

        half_life = -np.log(2) / beta
        return float(half_life)
