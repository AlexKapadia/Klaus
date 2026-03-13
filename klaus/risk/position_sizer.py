"""Half-Kelly + ATR-based position sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from klaus.config.settings import get_settings


class PositionSizer:
    """Computes trade volume using half-Kelly criterion scaled by ATR.

    Volume = kelly_fraction * (win_rate - (1-win_rate)/payoff_ratio) * equity / (ATR * contract_size)
    Capped at max_risk_per_trade_pct of equity.
    """

    def __init__(self):
        self._settings = get_settings().risk
        # Default win rate / payoff ratio (updated as trades accumulate)
        self._win_rate = 0.50
        self._avg_win = 1.5
        self._avg_loss = 1.0

    def calculate_volume(
        self,
        equity: float,
        atr: float,
        signal_strength: float,
        symbol_info: dict,
    ) -> float:
        """Calculate position size in lots.

        Args:
            equity: Current account equity.
            atr: Current ATR(14) value for the instrument.
            signal_strength: Signal strength 0-1 from the algorithm.
            symbol_info: Dict with volume_min, volume_max, volume_step, trade_contract_size.
        """
        if equity <= 0 or atr <= 0:
            return 0.0

        # Kelly criterion
        payoff_ratio = self._avg_win / max(self._avg_loss, 0.001)
        kelly = self._win_rate - (1 - self._win_rate) / payoff_ratio
        kelly = max(kelly, 0.0)  # Never negative

        # Half-Kelly
        kelly_fraction = kelly * self._settings.kelly_fraction

        # Risk budget: max_risk_per_trade_pct of equity
        max_risk = equity * (self._settings.max_risk_per_trade_pct / 100)

        # Risk per unit = ATR * SL multiplier * contract size
        contract_size = symbol_info.get("trade_contract_size", 100)
        sl_distance = atr * self._settings.default_sl_atr_mult
        risk_per_lot = sl_distance * contract_size

        if risk_per_lot <= 0:
            return 0.0

        # Base volume from risk budget
        volume = max_risk / risk_per_lot

        # Scale by Kelly fraction and signal strength
        volume *= kelly_fraction * signal_strength

        # Respect symbol limits
        vol_min = symbol_info.get("volume_min", 0.01)
        vol_max = symbol_info.get("volume_max", 100.0)
        vol_step = symbol_info.get("volume_step", 0.01)

        # Round to volume step
        if vol_step > 0:
            volume = round(volume / vol_step) * vol_step

        volume = np.clip(volume, vol_min, vol_max)

        return float(volume)

    def update_statistics(self, win_rate: float, avg_win: float, avg_loss: float) -> None:
        """Update win rate and payoff statistics from trade history."""
        self._win_rate = win_rate
        self._avg_win = avg_win
        self._avg_loss = avg_loss
        logger.debug(f"PositionSizer stats updated: WR={win_rate:.2f} W/L={avg_win:.2f}/{avg_loss:.2f}")
