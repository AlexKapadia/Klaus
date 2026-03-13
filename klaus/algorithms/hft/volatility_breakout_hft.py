"""ATR volatility breakout on minute bars.

Identifies consolidation periods (low ATR relative to recent history)
and trades breakouts when price escapes the compression range.
Inspired by the squeeze/expansion dynamics documented in energy
futures microstructure literature.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class VolatilityBreakoutHFT(BaseAlgorithm):
    """Volatility compression → breakout strategy on M1/M5 bars.

    Detects when ATR drops below a percentile of its recent history
    (compression), then enters when price makes a directional move
    exceeding a multiple of the compressed ATR.
    """

    name = "vol_breakout_hft"
    supported_instruments = ["XAUUSD", "XTIUSD", "XBRUSD", "XNGUSD", "XAGUSD"]
    preferred_regimes = [Regime.VOLATILE, Regime.TRENDING]
    min_bars_required = 60

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.atr_period = self.params.get("atr_period", 10)
        self.lookback = self.params.get("lookback", 50)
        self.compression_pctile = self.params.get("compression_percentile", 25)
        self.breakout_mult = self.params.get("breakout_multiplier", 2.0)
        self.volume_confirm = self.params.get("volume_confirm", True)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.lookback + self.atr_period:
            return None

        # Compute ATR on minute bars
        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift(1)).abs()
        low_close = (data["low"] - data["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(self.atr_period).mean()

        current_atr = atr.iloc[-1]
        if np.isnan(current_atr) or current_atr <= 0:
            return None

        # Check for compression: current ATR in bottom percentile
        atr_history = atr.tail(self.lookback).dropna()
        if len(atr_history) < 20:
            return None

        pctile = np.percentile(atr_history, self.compression_pctile)

        # We need the ATR to have BEEN compressed recently (last 5 bars)
        recent_atrs = atr.tail(5).values
        was_compressed = np.any(recent_atrs <= pctile)

        if not was_compressed:
            return None

        # Now check for breakout: current bar range > breakout_mult * compressed ATR
        compressed_atr = np.min(recent_atrs[~np.isnan(recent_atrs)])
        current_range = data["high"].iloc[-1] - data["low"].iloc[-1]

        if current_range < self.breakout_mult * compressed_atr:
            return None

        # Direction from the breakout bar
        bar_body = data["close"].iloc[-1] - data["open"].iloc[-1]
        if abs(bar_body) < compressed_atr * 0.1:
            return None  # Indecisive bar

        direction = Direction.LONG if bar_body > 0 else Direction.SHORT

        # Volume confirmation
        if self.volume_confirm and "volume" in data.columns:
            vol = data["volume"]
            avg_vol = vol.tail(20).mean()
            if avg_vol > 0 and vol.iloc[-1] < avg_vol:
                return None

        # Strength from breakout magnitude
        breakout_ratio = current_range / (compressed_atr * self.breakout_mult)
        strength = np.clip(breakout_ratio * 0.5, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "compressed_atr": float(compressed_atr),
                "current_range": float(current_range),
                "breakout_ratio": float(breakout_ratio),
                "hft_sl_mult": 0.25,
                "hft_tp_mult": 0.30,
            },
        )
