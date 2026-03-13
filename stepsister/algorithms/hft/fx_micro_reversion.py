"""VWAP deviation mean-reversion for forex M1 bars.

Exploits short-term price overreactions on M1 forex data by detecting
when price deviates significantly from its volume-weighted micro-mean
(VWAP). Trades reversion when the deviation exceeds a threshold.

Based on market microstructure theory: short-term price deviations from
fundamental value caused by order flow imbalance revert quickly. FX
markets exhibit strong intraday mean-reversion at the minute level due
to dealer inventory management (Lyons, 2001).

Lower entry threshold (1.2) and acceleration check disabled by default
for maximum signal frequency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.types import Direction, Regime, Signal
from stepsister.core.registry import register_fx_algorithm


@register_fx_algorithm
class FxMicroReversion(BaseAlgorithm):
    """VWAP mean-reversion on minute-bar forex pairs.

    Computes a rolling micro-VWAP and trades when price deviates
    beyond a z-score threshold, anticipating snap-back. Lower entry
    threshold and disabled acceleration filter for high signal frequency.
    """

    name = "fx_micro_reversion"
    supported_instruments = ["*"]
    preferred_regimes = [Regime.MEAN_REVERTING]
    min_bars_required = 40

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.vwap_period = self.params.get("vwap_period", 30)
        self.zscore_entry = self.params.get("zscore_entry", 1.2)
        self.zscore_exit = self.params.get("zscore_exit", 0.3)
        self.min_spread_mult = self.params.get("min_spread_multiplier", 3)
        self.use_acceleration = self.params.get("use_acceleration", False)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.vwap_period + 5:
            return None

        close = data["close"]
        volume = data["volume"] if "volume" in data.columns else pd.Series(1, index=data.index)

        # Compute rolling VWAP
        tp = (data["high"] + data["low"] + data["close"]) / 3
        cum_tp_vol = (tp * volume).rolling(self.vwap_period).sum()
        cum_vol = volume.rolling(self.vwap_period).sum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

        # Deviation from VWAP
        deviation = close - vwap
        dev_std = deviation.rolling(self.vwap_period).std()

        current_dev = deviation.iloc[-1]
        current_std = dev_std.iloc[-1]

        if np.isnan(current_dev) or np.isnan(current_std) or current_std <= 0:
            return None

        z = current_dev / current_std

        if np.isnan(z):
            return None

        # Must exceed entry threshold
        if abs(z) < self.zscore_entry:
            return None

        # Price acceleration check (disabled by default for max frequency)
        if self.use_acceleration:
            returns = close.pct_change()
            accel = returns.diff()
            recent_accel = accel.iloc[-1]

            if np.isnan(recent_accel):
                return None

            # If price is still accelerating away from mean, wait
            if z > 0 and recent_accel > 0:
                return None
            if z < 0 and recent_accel < 0:
                return None

        # Trade against the deviation (mean reversion)
        direction = Direction.SHORT if z > self.zscore_entry else Direction.LONG

        # Strength from z-score magnitude
        z_excess = (abs(z) - self.zscore_entry) / 2.0
        strength = np.clip(z_excess, 0.01, 1.0)

        # Compute distance to VWAP for tight TP
        vwap_distance = abs(close.iloc[-1] - vwap.iloc[-1])

        # Compute micro ATR for tight SL/TP
        high_low = data["high"] - data["low"]
        micro_atr = high_low.tail(10).mean()

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "vwap_z": float(z),
                "vwap": float(vwap.iloc[-1]),
                "vwap_distance": float(vwap_distance),
                "micro_atr": float(micro_atr) if not np.isnan(micro_atr) else 0.0,
                "hft_sl_mult": 0.25,
                "hft_tp_mult": 0.30,
            },
        )
