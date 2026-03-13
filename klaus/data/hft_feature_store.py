"""Microstructure feature computation for HFT algorithms.

Extends the standard FeatureStore with features specific to sub-minute
trading: VPIN proxy, order flow imbalance, tick intensity, micro-VWAP,
realised volatility at multiple scales, and price acceleration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from klaus.data.feature_store import FeatureStore


class HFTFeatureStore:
    """Computes microstructure and HFT-specific indicators on M1 DataFrames."""

    @staticmethod
    def add_all_hft_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add the full HFT feature set to an M1 OHLCV DataFrame.

        Includes all standard features (via FeatureStore) plus
        microstructure features.
        """
        df = df.copy()

        # Standard features first (fast periods for M1 data)
        df = FeatureStore.add_returns(df)
        df = FeatureStore.add_sma(df, [5, 10, 20])
        df = FeatureStore.add_ema(df, [3, 8, 21])
        df = FeatureStore.add_rsi(df, period=7)
        df = FeatureStore.add_atr(df, period=10)
        df = FeatureStore.add_bollinger(df, period=20, num_std=2.0)
        df = FeatureStore.add_rolling_volatility(df, period=10)
        df = FeatureStore.add_macd(df, fast=8, slow=21, signal=5)

        # HFT-specific features
        df = HFTFeatureStore.add_multi_return(df)
        df = HFTFeatureStore.add_micro_vwap(df)
        df = HFTFeatureStore.add_vwap_deviation(df)
        df = HFTFeatureStore.add_order_flow_proxy(df)
        df = HFTFeatureStore.add_volume_features(df)
        df = HFTFeatureStore.add_price_acceleration(df)
        df = HFTFeatureStore.add_realised_volatility(df)
        df = HFTFeatureStore.add_range_features(df)
        df = HFTFeatureStore.add_vpin_proxy(df)

        return df

    @staticmethod
    def add_multi_return(df: pd.DataFrame) -> pd.DataFrame:
        """Multi-horizon returns for short-term signal extraction."""
        df["returns_1"] = df["close"].pct_change(1)
        df["returns_3"] = df["close"].pct_change(3)
        df["returns_5"] = df["close"].pct_change(5)
        df["returns_10"] = df["close"].pct_change(10)
        return df

    @staticmethod
    def add_micro_vwap(df: pd.DataFrame, period: int = 30) -> pd.DataFrame:
        """Rolling VWAP computed on typical price * volume."""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

        cum_tp_vol = (tp * volume).rolling(period).sum()
        cum_vol = volume.rolling(period).sum()
        df["micro_vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
        return df

    @staticmethod
    def add_vwap_deviation(df: pd.DataFrame) -> pd.DataFrame:
        """Z-score of price deviation from micro-VWAP."""
        if "micro_vwap" not in df.columns:
            df = HFTFeatureStore.add_micro_vwap(df)

        deviation = df["close"] - df["micro_vwap"]
        dev_std = deviation.rolling(30).std().replace(0, np.nan)
        df["vwap_z"] = deviation / dev_std
        return df

    @staticmethod
    def add_order_flow_proxy(df: pd.DataFrame) -> pd.DataFrame:
        """Order flow imbalance proxy from bar structure.

        Uses the position of close within the bar range as a proxy
        for buy/sell pressure. Close near high = buy pressure.
        """
        bar_range = (df["high"] - df["low"]).replace(0, np.nan)
        # [-1, 1] where 1 = close at high (buying), -1 = close at low (selling)
        df["ofi_proxy"] = 2 * (df["close"] - df["low"]) / bar_range - 1

        # Cumulative OFI over recent bars
        df["ofi_cumulative_5"] = df["ofi_proxy"].rolling(5).sum()
        df["ofi_cumulative_10"] = df["ofi_proxy"].rolling(10).sum()
        return df

    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features for microstructure signals."""
        if "volume" not in df.columns:
            df["volume_ratio"] = 1.0
            df["volume_z"] = 0.0
            df["tick_intensity"] = 1.0
            return df

        vol = df["volume"]

        # Volume ratio vs rolling average
        avg_vol = vol.rolling(20).mean().replace(0, np.nan)
        df["volume_ratio"] = vol / avg_vol

        # Volume z-score
        vol_std = vol.rolling(20).std().replace(0, np.nan)
        df["volume_z"] = (vol - avg_vol) / vol_std

        # Tick intensity (volume acceleration)
        df["tick_intensity"] = vol.rolling(5).mean() / vol.rolling(20).mean().replace(0, np.nan)

        return df

    @staticmethod
    def add_price_acceleration(df: pd.DataFrame) -> pd.DataFrame:
        """Second derivative of price (momentum of momentum)."""
        if "returns" not in df.columns:
            df["returns"] = df["close"].pct_change()

        df["price_acceleration"] = df["returns"].diff()
        df["acceleration_3"] = df["returns"].diff(3)
        return df

    @staticmethod
    def add_realised_volatility(df: pd.DataFrame) -> pd.DataFrame:
        """Multi-scale realised volatility using squared returns."""
        if "returns" not in df.columns:
            df["returns"] = df["close"].pct_change()

        sq_returns = df["returns"] ** 2

        # 5-bar realised vol (ultra-short)
        df["rv_5"] = np.sqrt(sq_returns.rolling(5).sum())
        # 10-bar realised vol
        df["rv_10"] = np.sqrt(sq_returns.rolling(10).sum())
        # 30-bar realised vol
        df["rv_30"] = np.sqrt(sq_returns.rolling(30).sum())

        # Volatility ratio (short/long) for regime sensing
        df["vol_ratio_5_30"] = df["rv_5"] / df["rv_30"].replace(0, np.nan)

        return df

    @staticmethod
    def add_range_features(df: pd.DataFrame) -> pd.DataFrame:
        """Bar range features for volatility and breakout detection."""
        bar_range = df["high"] - df["low"]
        avg_range = bar_range.rolling(10).mean().replace(0, np.nan)

        df["range_ratio"] = bar_range / avg_range

        # Bar body size relative to full range
        body = (df["close"] - df["open"]).abs()
        df["body_ratio"] = body / bar_range.replace(0, np.nan)

        # Upper/lower wick ratios
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / bar_range.replace(0, np.nan)
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / bar_range.replace(0, np.nan)

        return df

    @staticmethod
    def add_vpin_proxy(df: pd.DataFrame, bucket_bars: int = 10, n_buckets: int = 20) -> pd.DataFrame:
        """Simplified VPIN proxy using BVC on M1 bars.

        Bulk Volume Classification: uses the normal CDF of
        standardised returns to estimate buy/sell volume split.
        """
        if "volume" not in df.columns:
            df["vpin_proxy"] = 0.5
            return df

        if "returns" not in df.columns:
            df["returns"] = df["close"].pct_change()

        returns = df["returns"].values
        volume = df["volume"].values

        # Rolling std for BVC
        rolling_std = pd.Series(returns).rolling(20).std().values
        rolling_std = np.where((rolling_std <= 0) | np.isnan(rolling_std), 1e-8, rolling_std)

        z_scores = np.where(np.isnan(returns), 0, returns / rolling_std)
        buy_pct = norm.cdf(z_scores)

        buy_vol = volume * buy_pct
        sell_vol = volume * (1 - buy_pct)

        # Rolling VPIN over bucket windows
        total_bars = bucket_bars * n_buckets
        vpin_values = np.full(len(df), np.nan)

        for i in range(total_bars, len(df)):
            bv = buy_vol[i - total_bars:i]
            sv = sell_vol[i - total_bars:i]
            total = bv + sv
            total_sum = total.sum()
            if total_sum > 0:
                abs_imbalance = np.abs(bv - sv).sum()
                vpin_values[i] = abs_imbalance / total_sum

        df["vpin_proxy"] = vpin_values
        return df
