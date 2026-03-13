"""Cross-sectional FX momentum strategy.

Based on Menkhoff, Sarno, Schmeling & Schrimpf (2012, JFE) and
Zhang (2022, JFE). Ranks currencies by past returns and trades
the direction of strong momentum, scaled by volatility.

Sharpe ratio ~0.5-0.94 in the literature.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from stepsister.core.registry import register_fx_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_fx_algorithm
class FXMomentum(BaseAlgorithm):
    """Cross-sectional momentum on forex pairs.

    If the pair has strong positive returns over the ranking period,
    go LONG. If strong negative, go SHORT. Strength scaled by
    momentum magnitude relative to volatility.
    """

    name = "fx_momentum"
    supported_instruments = ["*"]
    preferred_regimes = [Regime.TRENDING]
    min_bars_required = 30

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.ranking_period = self.params.get("ranking_period", 20)
        self.holding_period = self.params.get("holding_period", 5)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.ranking_period + 5:
            return None

        close = data["close"]

        # Momentum: cumulative return over ranking period
        ret = (close.iloc[-1] - close.iloc[-self.ranking_period - 1]) / close.iloc[-self.ranking_period - 1]

        if np.isnan(ret):
            return None

        # Volatility-scaled momentum
        returns = close.pct_change().dropna()
        vol = returns.tail(self.ranking_period).std()

        if vol is None or np.isnan(vol) or vol <= 0:
            return None

        # Momentum z-score: return / volatility
        mom_z = ret / (vol * np.sqrt(self.ranking_period))

        if np.isnan(mom_z):
            return None

        # Need meaningful momentum — at least 0.5 sigma
        if abs(mom_z) < 0.5:
            return None

        direction = Direction.LONG if mom_z > 0 else Direction.SHORT

        # Strength from z-score magnitude
        strength = np.clip(abs(mom_z) / 3.0, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "momentum_return": float(ret),
                "momentum_z": float(mom_z),
                "volatility": float(vol),
            },
        )
