"""Geopolitical Risk (GPR) index signal for gold and oil."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal


@register_algorithm
class GPRIndex(BaseAlgorithm):
    """Geopolitical risk proxy using market-derived signals.

    Since the Caldara-Iacoviello GPR index is monthly and not live,
    this uses a market-based proxy:
    - Gold/VIX-proxy behaviour: sharp rallies in gold + high vol = geopolitical stress
    - Measured as z-score of (gold returns / volatility ratio)

    In high GPR regimes: long gold, long oil (supply disruption risk).
    """

    name = "gpr_index"
    supported_instruments = ["XAUUSD", "XTIUSD"]
    preferred_regimes = [Regime.TRENDING, Regime.VOLATILE]
    min_bars_required = 40

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = self.params.get("lookback", 30)
        self.threshold = self.params.get("threshold", 1.5)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.lookback + 5:
            return None

        close = data["close"]
        returns = close.pct_change()

        # Proxy GPR: sudden rallies + elevated vol
        # Use recent 5-day return relative to lookback vol
        recent_return = returns.tail(5).sum()
        lookback_vol = returns.tail(self.lookback).std()

        if lookback_vol <= 0 or np.isnan(lookback_vol):
            return None

        gpr_proxy = recent_return / lookback_vol

        if np.isnan(gpr_proxy):
            return None

        # High GPR → safe-haven buying (gold) or supply-fear (oil)
        if gpr_proxy > self.threshold:
            direction = Direction.LONG
        elif gpr_proxy < -self.threshold:
            # Negative GPR proxy → risk-on → short safe havens
            if symbol == "XAUUSD":
                direction = Direction.SHORT
            else:
                return None  # Oil doesn't short on risk-on from GPR
        else:
            return None

        strength = np.clip(abs(gpr_proxy) / 3.0, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={"gpr_proxy": float(gpr_proxy)},
        )
