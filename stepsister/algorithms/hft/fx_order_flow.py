"""VPIN-based order flow imbalance signal for forex HFT.

Based on Easley, Lopez de Prado & O'Hara (2012, Review of Financial Studies)
adapted for spot FX microstructure following Evans & Lyons (2002, Journal of
Political Economy). Uses bulk volume classification (BVC) with normal CDF to
estimate buy/sell pressure from tick-volume on M1 bars, then computes VPIN
over rolling volume buckets.

High VPIN -> informed trading detected -> anticipate volatility spike.
Directional signal derived from net order flow direction.

Lower VPIN threshold (0.4) than commodity version — FX market microstructure
exhibits different information asymmetry characteristics due to the
dealer-dominated OTC structure.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.types import Direction, Regime, Signal
from stepsister.core.registry import register_fx_algorithm


@register_fx_algorithm
class FxOrderFlow(BaseAlgorithm):
    """VPIN order flow imbalance for high-frequency forex trading.

    Classifies each M1 bar's volume as buy or sell using the BVC method
    (Easley et al., 2012). Accumulates into volume buckets and computes
    VPIN. Trades the direction of net imbalance when VPIN spikes.

    Lower VPIN threshold for FX due to different microstructure.
    """

    name = "fx_order_flow"
    supported_instruments = ["*"]
    preferred_regimes = [Regime.TRENDING, Regime.VOLATILE]
    min_bars_required = 100

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.n_buckets = self.params.get("n_buckets", 50)
        self.bucket_size_bars = self.params.get("bucket_size_bars", 10)
        self.vpin_threshold = self.params.get("vpin_threshold", 0.4)
        self.ofi_lookback = self.params.get("ofi_lookback", 20)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(data) < self.n_buckets * self.bucket_size_bars:
            return None

        close = data["close"].values
        volume = data["volume"].values

        # Bulk Volume Classification (BVC)
        # Approximate buy/sell volume using price change and normal CDF
        returns = np.diff(close) / close[:-1]
        # Avoid division by zero in low-vol periods
        rolling_std = pd.Series(returns).rolling(20).std().values
        rolling_std = np.where(rolling_std <= 0, 1e-8, rolling_std)

        z_scores = returns / rolling_std
        buy_pct = norm.cdf(z_scores)

        vol = volume[1:]  # align with returns
        buy_vol = vol * buy_pct
        sell_vol = vol * (1 - buy_pct)

        # Compute VPIN over last n_buckets of bucket_size_bars each
        total_needed = self.n_buckets * self.bucket_size_bars
        if len(buy_vol) < total_needed:
            return None

        recent_buy = buy_vol[-total_needed:]
        recent_sell = sell_vol[-total_needed:]

        # Reshape into buckets
        buy_buckets = recent_buy.reshape(self.n_buckets, self.bucket_size_bars).sum(axis=1)
        sell_buckets = recent_sell.reshape(self.n_buckets, self.bucket_size_bars).sum(axis=1)
        total_buckets = buy_buckets + sell_buckets

        # VPIN = average absolute imbalance / average total volume
        abs_imbalance = np.abs(buy_buckets - sell_buckets)
        total_vol = total_buckets.sum()
        if total_vol <= 0:
            return None

        vpin = abs_imbalance.sum() / total_vol

        if np.isnan(vpin):
            return None

        # Only trade when VPIN exceeds threshold (informed trading detected)
        if vpin < self.vpin_threshold:
            return None

        # Direction from recent net order flow
        recent_ofi = buy_vol[-self.ofi_lookback:].sum() - sell_vol[-self.ofi_lookback:].sum()
        recent_total = vol[-self.ofi_lookback:].sum()

        if recent_total <= 0:
            return None

        ofi_ratio = recent_ofi / recent_total

        if abs(ofi_ratio) < 0.02:
            return None

        direction = Direction.LONG if ofi_ratio > 0 else Direction.SHORT

        # Strength from VPIN intensity and OFI magnitude
        vpin_excess = (vpin - self.vpin_threshold) / (1.0 - self.vpin_threshold)
        ofi_strength = min(abs(ofi_ratio) * 2, 1.0)
        strength = np.clip(vpin_excess * 0.5 + ofi_strength * 0.5, 0.01, 1.0)

        # Compute micro ATR for tight SL/TP
        high_low = data["high"] - data["low"]
        micro_atr = high_low.tail(10).mean()

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=float(strength),
            algo_name=self.name,
            metadata={
                "vpin": float(vpin),
                "ofi_ratio": float(ofi_ratio),
                "buy_vol_recent": float(buy_vol[-self.ofi_lookback:].sum()),
                "sell_vol_recent": float(sell_vol[-self.ofi_lookback:].sum()),
                "micro_atr": float(micro_atr) if not np.isnan(micro_atr) else 0.0,
                "hft_sl_mult": 0.25,
                "hft_tp_mult": 0.30,
            },
        )
