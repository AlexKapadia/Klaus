"""GJR-GARCH volatility model for trading volatile regimes."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.registry import register_algorithm
from klaus.core.types import Direction, Regime, Signal

try:
    from arch import arch_model
    _HAS_ARCH = True
except ImportError:
    _HAS_ARCH = False


@register_algorithm
class GJRGarch(BaseAlgorithm):
    """GJR-GARCH(1,1,1) volatility forecasting.

    In volatile regimes, forecast whether vol is expanding or contracting.
    - If forecast vol > current realised vol * threshold → expect vol expansion → short
    - If forecast vol < current realised vol / threshold → vol contraction → long
    """

    name = "gjr_garch"
    supported_instruments = ["XTIUSD", "XBRUSD", "XAUUSD"]
    preferred_regimes = [Regime.VOLATILE]
    min_bars_required = 200

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.p = self.params.get("p", 1)
        self.o = self.params.get("o", 1)
        self.q = self.params.get("q", 1)
        self.horizon = self.params.get("vol_forecast_horizon", 5)
        self.threshold = self.params.get("high_vol_threshold", 1.5)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if not _HAS_ARCH:
            logger.debug("arch package not available, skipping GJR-GARCH")
            return None

        if len(data) < self.min_bars_required:
            return None

        returns = data["close"].pct_change().dropna() * 100  # scale for arch

        if len(returns) < 100:
            return None

        try:
            model = arch_model(
                returns,
                vol="Garch",
                p=self.p,
                o=self.o,
                q=self.q,
                dist="Normal",
            )
            result = model.fit(disp="off", show_warning=False)
            forecast = result.forecast(horizon=self.horizon)

            forecast_var = forecast.variance.iloc[-1].mean()
            forecast_vol = np.sqrt(forecast_var)

            current_vol = returns.tail(20).std()

            if current_vol <= 0:
                return None

            vol_ratio = forecast_vol / current_vol

        except Exception as e:
            logger.debug(f"GJR-GARCH fit failed for {symbol}: {e}")
            return None

        direction = None
        if vol_ratio > self.threshold:
            # Volatility expanding → risk-off / short
            direction = Direction.SHORT
        elif vol_ratio < 1.0 / self.threshold:
            # Volatility contracting → risk-on / long
            direction = Direction.LONG
        else:
            return None

        strength = np.clip(abs(vol_ratio - 1.0) / 2.0, 0.01, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={"vol_ratio": float(vol_ratio), "forecast_vol": float(forecast_vol)},
        )
