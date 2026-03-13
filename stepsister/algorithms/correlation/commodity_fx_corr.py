"""Commodity-currency correlation trading.

Exploits established correlations between commodity prices and
currency pairs:
  - AUD/USD correlates positively with gold (r ~0.60-0.80)
  - USD/CAD correlates negatively with WTI crude (r ~-0.70 to -0.85)
  - NZD/USD correlates with dairy/agricultural commodities

Strategy: when the commodity moves >2 sigma from its 20-day mean but
the corresponding FX pair hasn't adjusted, enter the FX trade
anticipating convergence.

Based on the peer-reviewed survey Section V on commodity-currency
correlation trading.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from stepsister.core.registry import register_fx_algorithm
from klaus.core.types import Direction, Regime, Signal


# Commodity-FX relationships: (commodity_symbol, expected_beta)
# beta > 0: commodity up → FX pair up
# beta < 0: commodity up → FX pair down
_COMMODITY_FX_MAP: dict[str, tuple[str, float]] = {
    "AUDUSD": ("XAUUSD", 0.70),    # Gold up → AUD/USD up
    "USDCAD": ("XTIUSD", -0.75),   # Oil up → USD/CAD down
    "NZDUSD": ("XAUUSD", 0.50),    # Gold up → NZD/USD up (weaker)
}


@register_fx_algorithm
class CommodityFXCorrelation(BaseAlgorithm):
    """Commodity-currency correlation signal for forex.

    Monitors commodity price moves and trades the correlated FX pair
    when divergence suggests the FX pair hasn't yet adjusted.
    Requires commodity data via set_commodity_data().
    """

    name = "commodity_fx_corr"
    supported_instruments = list(_COMMODITY_FX_MAP.keys())
    preferred_regimes = [Regime.TRENDING]
    min_bars_required = 60

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.corr_lookback = self.params.get("correlation_lookback", 120)
        self.divergence_threshold = self.params.get("divergence_threshold", 2.0)
        self._commodity_data: dict[str, pd.DataFrame] = {}

    def set_commodity_data(self, commodity_symbol: str, data: pd.DataFrame) -> None:
        """Provide commodity OHLCV data for cross-asset correlation."""
        self._commodity_data[commodity_symbol] = data

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if symbol not in _COMMODITY_FX_MAP:
            return None

        commodity_symbol, expected_beta = _COMMODITY_FX_MAP[symbol]

        if commodity_symbol not in self._commodity_data:
            return None

        commodity_df = self._commodity_data[commodity_symbol]
        if commodity_df.empty or len(commodity_df) < 30:
            return None

        if len(data) < 30:
            return None

        # Commodity z-score from its own 20-day mean
        comm_close = commodity_df["close"]
        comm_mean = comm_close.rolling(20).mean()
        comm_std = comm_close.rolling(20).std()

        comm_z = (comm_close.iloc[-1] - comm_mean.iloc[-1]) / comm_std.iloc[-1]

        if np.isnan(comm_z):
            return None

        # Only trade when commodity has a significant deviation
        if abs(comm_z) < self.divergence_threshold:
            return None

        # FX pair's recent move
        fx_close = data["close"]
        fx_ret = (fx_close.iloc[-1] - fx_close.iloc[-6]) / fx_close.iloc[-6]

        if np.isnan(fx_ret):
            return None

        # Expected FX move based on commodity move and beta
        expected_fx_direction = np.sign(comm_z * expected_beta)

        # Check if FX has already moved in the expected direction
        fx_moved = (fx_ret > 0.001 and expected_fx_direction > 0) or \
                   (fx_ret < -0.001 and expected_fx_direction < 0)

        if fx_moved:
            return None  # Already adjusted — no edge

        # Trade FX in the expected convergence direction
        direction = Direction.LONG if expected_fx_direction > 0 else Direction.SHORT

        # Strength from commodity deviation magnitude
        strength = np.clip(abs(comm_z) / 4.0, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "commodity": commodity_symbol,
                "commodity_z": float(comm_z),
                "fx_return_5d": float(fx_ret),
                "expected_beta": float(expected_beta),
            },
        )
