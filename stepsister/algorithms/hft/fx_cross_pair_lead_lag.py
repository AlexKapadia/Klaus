"""Cross-pair lead-lag exploitation for forex on M1 bars.

Exploits the empirically documented lead-lag relationships between
correlated FX pairs at high frequency:
- EUR/USD leads GBP/USD (common USD denominator)
- EUR/USD inverse-leads EUR/GBP (EUR numerator)
- USD/JPY leads EUR/JPY and GBP/JPY (JPY denominator)
- AUD/USD and NZD/USD co-move with slight lead-lag

When the leader makes a significant move and the lagger hasn't
followed, trades the lagger in the expected convergence direction.

Based on Chaboud et al. (2014, Journal of Finance) on high-frequency
cross-currency arbitrage dynamics and Evans & Lyons (2002) on FX
microstructure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.types import Direction, Regime, Signal
from stepsister.core.registry import register_fx_algorithm


# Known FX lead-lag pairs: lagger -> (leader, expected_beta)
# beta > 0 means same direction; beta < 0 means inverse
_FX_LEAD_LAG_PAIRS = {
    "GBPUSD": ("EURUSD", 0.8),    # EUR/USD leads GBP/USD
    "EURGBP": ("EURUSD", -0.5),   # EUR/USD inverse-leads EUR/GBP
    "EURJPY": ("USDJPY", 0.7),    # USD/JPY leads EUR/JPY
    "GBPJPY": ("USDJPY", 0.7),    # USD/JPY leads GBP/JPY
    "AUDUSD": ("NZDUSD", 0.9),    # NZD/USD co-moves, slight lead
    "NZDUSD": ("AUDUSD", 0.9),    # AUD/USD co-moves, slight lead
}


@register_fx_algorithm
class FxCrossPairLeadLag(BaseAlgorithm):
    """Cross-pair lead-lag signal on M1 forex data.

    Monitors the leader's recent returns and trades the lagger
    when divergence exceeds a threshold, expecting convergence.
    """

    name = "fx_cross_pair_lead_lag"
    supported_instruments = list(_FX_LEAD_LAG_PAIRS.keys())
    preferred_regimes = [Regime.TRENDING, Regime.MEAN_REVERTING]
    min_bars_required = 30

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.return_window = self.params.get("return_window", 5)
        self.lag_window = self.params.get("lag_window", 1)
        self.divergence_threshold = self.params.get("divergence_threshold", 0.0003)
        self.corr_lookback = self.params.get("correlation_lookback", 60)
        self.min_correlation = self.params.get("min_correlation", 0.3)
        self._leader_data: dict[str, pd.DataFrame] = {}

    def set_leader_data(self, leader_symbol: str, data: pd.DataFrame) -> None:
        """Provide the leader instrument's M1 data."""
        self._leader_data[leader_symbol] = data

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if symbol not in _FX_LEAD_LAG_PAIRS:
            return None

        leader_symbol, expected_beta = _FX_LEAD_LAG_PAIRS[symbol]

        if leader_symbol not in self._leader_data:
            return None

        leader_df = self._leader_data[leader_symbol]

        if len(data) < self.corr_lookback or len(leader_df) < self.corr_lookback:
            return None

        # Align data on common timestamps
        combined = pd.DataFrame({
            "lagger": data["close"],
            "leader": leader_df["close"],
        }).dropna()

        if len(combined) < self.corr_lookback:
            return None

        # Verify correlation is still valid
        lagger_returns = combined["lagger"].pct_change()
        leader_returns = combined["leader"].pct_change()

        corr = lagger_returns.tail(self.corr_lookback).corr(
            leader_returns.tail(self.corr_lookback)
        )

        if np.isnan(corr) or abs(corr) < self.min_correlation:
            return None

        # Leader's recent move (with lag offset)
        leader_ret = leader_returns.iloc[-self.lag_window - self.return_window:-self.lag_window].sum()
        # Lagger's concurrent move
        lagger_ret = lagger_returns.iloc[-self.return_window:].sum()

        if np.isnan(leader_ret) or np.isnan(lagger_ret):
            return None

        # Expected lagger move based on leader and beta
        expected_move = leader_ret * expected_beta * np.sign(corr)
        divergence = expected_move - lagger_ret

        if abs(divergence) < self.divergence_threshold:
            return None

        # Trade the lagger in the direction of expected convergence
        direction = Direction.LONG if divergence > 0 else Direction.SHORT

        # Strength from divergence magnitude
        strength = np.clip(abs(divergence) / (self.divergence_threshold * 3), 0.01, 1.0)

        # Compute micro ATR for tight SL/TP
        high_low = data["high"] - data["low"]
        micro_atr = high_low.tail(10).mean()

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "leader": leader_symbol,
                "leader_return": float(leader_ret),
                "lagger_return": float(lagger_ret),
                "divergence": float(divergence),
                "correlation": float(corr),
                "micro_atr": float(micro_atr) if not np.isnan(micro_atr) else 0.0,
                "hft_sl_mult": 0.25,
                "hft_tp_mult": 0.30,
            },
        )
