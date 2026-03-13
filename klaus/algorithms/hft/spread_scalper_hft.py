"""High-frequency WTI-Brent spread scalper on M1 bars.

Extension of the standard spread_ou to minute-bar frequency. Exploits
the tighter mean-reversion dynamics of the crude oil spread at sub-hourly
timescales. Based on Fanelli et al. (2023) regime-switching OU model
adapted to higher frequency.

Uses a fast-adapting z-score with short lookback and tight entry/exit
thresholds for rapid turnover scalping.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class SpreadScalperHFT(BaseAlgorithm):
    """HF spread scalping between correlated commodity pairs on M1 bars.

    Pairs: WTI/Brent, Gold/Silver, or any two provided via set_pair_data().
    Much faster mean-reversion assumptions than the H1 spread_ou.
    """

    name = "spread_scalper_hft"
    supported_instruments = ["XTIUSD", "XBRUSD", "XAUUSD", "XAGUSD"]
    preferred_regimes = [Regime.MEAN_REVERTING]
    min_bars_required = 40

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = self.params.get("lookback", 30)
        self.entry_z = self.params.get("entry_z", 1.5)
        self.exit_z = self.params.get("exit_z", 0.3)
        self.half_life_max = self.params.get("half_life_max", 15)
        self.min_spread_vol = self.params.get("min_spread_vol", 0.0001)
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

        # Log spread for better stationarity
        spread = np.log(combined["a"]) - np.log(combined["b"])

        # Quick half-life check
        half_life = self._compute_half_life(spread.values[-self.lookback:])
        if half_life is None or half_life > self.half_life_max or half_life < 0.5:
            return None

        # Rolling z-score with short lookback
        rolling_mean = spread.rolling(self.lookback).mean()
        rolling_std = spread.rolling(self.lookback).std()

        current_std = rolling_std.iloc[-1]
        if np.isnan(current_std) or current_std < self.min_spread_vol:
            return None

        z = (spread.iloc[-1] - rolling_mean.iloc[-1]) / current_std

        if np.isnan(z):
            return None

        # Entry logic
        direction = None
        if z < -self.entry_z:
            direction = Direction.LONG  # spread too low, buy A sell B
        elif z > self.entry_z:
            direction = Direction.SHORT  # spread too high, sell A buy B
        else:
            return None

        # Strength from z-score excess and half-life confidence
        z_excess = (abs(z) - self.entry_z) / 2.0
        hl_confidence = 1.0 - (half_life / self.half_life_max)
        strength = np.clip(z_excess * 0.6 + hl_confidence * 0.4, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "spread_z": float(z),
                "half_life": float(half_life),
                "spread_std": float(current_std),
                "hft_sl_mult": 0.25,
                "hft_tp_mult": 0.30,
            },
        )

    @staticmethod
    def _compute_half_life(spread: np.ndarray) -> Optional[float]:
        """Estimate half-life of mean reversion via AR(1) regression."""
        if len(spread) < 10:
            return None
        lag = spread[:-1]
        diff = np.diff(spread)
        lag_mean = lag.mean()
        diff_mean = diff.mean()
        numerator = np.sum((lag - lag_mean) * (diff - diff_mean))
        denominator = np.sum((lag - lag_mean) ** 2)
        if denominator == 0:
            return None
        beta = numerator / denominator
        if beta >= 0:
            return None
        return float(-np.log(2) / beta)
