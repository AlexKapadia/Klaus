"""Dynamic carry trade with volatility scaling.

Based on Dupuy & Marsh (2021, JBF) — Sharpe 1.08.
High-yielding currencies (AUD, NZD) are bought against low-yielders
(JPY, CHF). Position sized inversely with volatility.

Also informed by Chernov, Dahlquist & Lochstoer (2024, NBER) —
hedging geographically-based risks increases carry Sharpe to 1.29.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from stepsister.core.registry import register_fx_algorithm
from klaus.core.types import Direction, Regime, Signal


# Static carry bias: positive = go LONG on this pair (buy base, sell quote)
# Based on historical interest rate differentials
_CARRY_BIAS: dict[str, float] = {
    "AUDUSD": 1.0,     # AUD high-yield → long
    "NZDUSD": 1.0,     # NZD high-yield → long
    "USDJPY": 1.0,     # JPY low-yield → long USD/JPY (buy USD, sell JPY)
    "EURJPY": 1.0,     # JPY low-yield → long EUR/JPY
    "GBPJPY": 1.0,     # JPY low-yield → long GBP/JPY
    "USDCHF": 1.0,     # CHF low-yield → long USD/CHF
    "EURUSD": 0.0,     # Neutral — similar yields
    "GBPUSD": 0.0,     # Neutral
    "USDCAD": 0.0,     # Neutral
    "EURGBP": 0.0,     # Neutral
}


@register_fx_algorithm
class CarryTrade(BaseAlgorithm):
    """Dynamic carry trade on forex pairs.

    Goes long high-yielders vs low-yielders. Scales position
    inversely with volatility. Stronger signal when carry
    direction aligns with short-term momentum.
    """

    name = "carry_trade"
    supported_instruments = ["AUDUSD", "NZDUSD", "USDJPY", "EURJPY", "GBPJPY", "USDCHF"]
    preferred_regimes = [Regime.MEAN_REVERTING, Regime.TRENDING]
    min_bars_required = 30

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.forward_lookback = self.params.get("forward_lookback", 20)
        self.vol_scaling = self.params.get("vol_scaling", True)
        self.vol_target = self.params.get("vol_target", 0.10)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.forward_lookback + 5:
            return None

        carry_bias = _CARRY_BIAS.get(symbol, 0.0)
        if carry_bias == 0.0:
            return None

        close = data["close"]

        # Carry direction
        direction = Direction.LONG if carry_bias > 0 else Direction.SHORT

        # Check if momentum aligns with carry direction
        ret = (close.iloc[-1] - close.iloc[-self.forward_lookback - 1]) / close.iloc[-self.forward_lookback - 1]
        if np.isnan(ret):
            return None

        momentum_aligns = (ret > 0 and direction == Direction.LONG) or (ret < 0 and direction == Direction.SHORT)

        # Base strength from carry
        base_strength = 0.4

        # Bonus if momentum aligns
        if momentum_aligns:
            base_strength += 0.3

        # Volatility scaling: reduce strength when vol is high
        if self.vol_scaling:
            returns = close.pct_change().dropna()
            vol = returns.tail(self.forward_lookback).std() * np.sqrt(252)
            if not np.isnan(vol) and vol > 0:
                vol_ratio = self.vol_target / vol
                base_strength *= min(vol_ratio, 1.5)

        strength = np.clip(base_strength, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "carry_bias": float(carry_bias),
                "momentum_return": float(ret),
                "momentum_aligns": momentum_aligns,
            },
        )
