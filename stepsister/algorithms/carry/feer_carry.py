"""FEER-conditional carry trade.

Based on Jorda & Taylor (2012, JIE) — adding a fundamental
equilibrium exchange rate (FEER) filter to carry. Avoids carry
when the target currency is overvalued versus PPP. Raised
Sharpe from 0.30 to 0.67 while flipping skewness from -0.69 to +0.37.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from stepsister.core.registry import register_fx_algorithm
from klaus.core.types import Direction, Regime, Signal


_CARRY_DIRECTION: dict[str, Direction] = {
    "AUDUSD": Direction.LONG,
    "NZDUSD": Direction.LONG,
    "USDJPY": Direction.LONG,
    "EURJPY": Direction.LONG,
    "GBPJPY": Direction.LONG,
    "USDCHF": Direction.LONG,
}


@register_fx_algorithm
class FEERCarry(BaseAlgorithm):
    """FEER-conditional carry: only trade carry when currency is not overvalued.

    Uses the long-run rolling mean of the exchange rate as a PPP proxy.
    If the pair is overvalued (z-score > threshold), carry is suppressed.
    If undervalued or neutral, carry is active with enhanced confidence.
    """

    name = "feer_carry"
    supported_instruments = ["*"]
    preferred_regimes = [Regime.MEAN_REVERTING]
    min_bars_required = 130

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.ppp_lookback = self.params.get("ppp_lookback", 120)
        self.overval_threshold = self.params.get("overvaluation_threshold", 0.15)
        self.forward_lookback = self.params.get("forward_lookback", 20)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.ppp_lookback + 10:
            return None

        # Only trade pairs with known carry direction
        carry_dir = _CARRY_DIRECTION.get(symbol)
        if carry_dir is None:
            return None

        close = data["close"]

        # PPP proxy: long-term rolling mean
        ppp_mean = close.rolling(self.ppp_lookback).mean()
        ppp_std = close.rolling(self.ppp_lookback).std()

        current_mean = ppp_mean.iloc[-1]
        current_std = ppp_std.iloc[-1]

        if np.isnan(current_mean) or np.isnan(current_std) or current_std <= 0:
            return None

        # Valuation z-score
        z = (close.iloc[-1] - current_mean) / current_std

        if np.isnan(z):
            return None

        # Check overvaluation relative to carry direction
        # If we want to go LONG and the pair is already overvalued → skip
        if carry_dir == Direction.LONG and z > self.overval_threshold:
            return None
        # If we want to go SHORT and the pair is already undervalued → skip
        if carry_dir == Direction.SHORT and z < -self.overval_threshold:
            return None

        # Enhanced strength when undervalued in carry direction
        # (buying undervalued + carry = double tailwind)
        if carry_dir == Direction.LONG:
            underval_bonus = max(-z * 0.3, 0)
        else:
            underval_bonus = max(z * 0.3, 0)

        base_strength = 0.4 + underval_bonus
        strength = np.clip(base_strength, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=carry_dir,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "ppp_z": float(z),
                "ppp_mean": float(current_mean),
                "underval_bonus": float(underval_bonus),
            },
        )
