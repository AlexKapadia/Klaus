"""Ultra-fast EMA crossover scalper for forex M1 bars.

Uses ultra-fast EMA crossovers (3/8) with RSI extremes filter to catch
micro-momentum bursts on 1-minute forex data. Volume confirmation
disabled — tick volume is unreliable in spot FX.

Designed for tight SL/TP and maximum turnover on all major and cross
pairs. Relaxed RSI thresholds (70/30) to maximise signal frequency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.types import Direction, Regime, Signal
from stepsister.core.registry import register_fx_algorithm


@register_fx_algorithm
class FxTickScalper(BaseAlgorithm):
    """Sub-minute EMA crossover scalper for spot forex.

    Entry: fast EMA (3) crosses slow EMA (8) with RSI confirmation.
    Volume confirmation disabled for FX (tick volume unreliable).
    Relaxed RSI thresholds for maximum signal frequency.
    """

    name = "fx_tick_scalper"
    supported_instruments = ["*"]
    preferred_regimes = [Regime.TRENDING]
    min_bars_required = 30

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.fast_ema = self.params.get("fast_ema", 3)
        self.slow_ema = self.params.get("slow_ema", 8)
        self.rsi_period = self.params.get("rsi_period", 7)
        self.rsi_ob = self.params.get("rsi_overbought", 70)
        self.rsi_os = self.params.get("rsi_oversold", 30)
        self.confirmation_bars = self.params.get("confirmation_bars", 2)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.slow_ema + self.rsi_period + 5:
            return None

        close = data["close"]

        # Fast/slow EMA
        ema_fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_ema, adjust=False).mean()

        # Detect crossover in last 2 bars
        prev_diff = ema_fast.iloc[-2] - ema_slow.iloc[-2]
        curr_diff = ema_fast.iloc[-1] - ema_slow.iloc[-1]

        if np.isnan(prev_diff) or np.isnan(curr_diff):
            return None

        # No crossover
        if not ((prev_diff <= 0 and curr_diff > 0) or (prev_diff >= 0 and curr_diff < 0)):
            return None

        bullish_cross = curr_diff > 0

        # RSI filter — don't buy into overbought, don't sell into oversold
        rsi = self._compute_rsi(close, self.rsi_period)
        if np.isnan(rsi):
            return None

        if bullish_cross and rsi > self.rsi_ob:
            return None  # overbought, skip
        if not bullish_cross and rsi < self.rsi_os:
            return None  # oversold, skip

        # Volume check disabled for forex — tick volume unreliable

        direction = Direction.LONG if bullish_cross else Direction.SHORT

        # Strength from EMA spread magnitude + RSI distance from neutral
        spread_pct = abs(curr_diff) / close.iloc[-1]
        rsi_strength = abs(rsi - 50) / 50
        strength = np.clip(spread_pct * 200 + rsi_strength * 0.3, 0.01, 1.0)

        # Compute micro ATR for tight SL/TP
        high_low = data["high"] - data["low"]
        micro_atr = high_low.tail(10).mean()

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "ema_spread": float(curr_diff),
                "rsi": float(rsi),
                "micro_atr": float(micro_atr) if not np.isnan(micro_atr) else 0.0,
                "hft_sl_mult": 0.25,
                "hft_tp_mult": 0.30,
            },
        )

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int) -> float:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return float(val) if not np.isnan(val) else np.nan
