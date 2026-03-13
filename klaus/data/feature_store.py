"""Rolling feature computation for the algorithm layer."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureStore:
    """Computes and caches rolling indicators on OHLCV DataFrames."""

    @staticmethod
    def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add the full standard feature set to an OHLCV DataFrame."""
        df = df.copy()
        df = FeatureStore.add_returns(df)
        df = FeatureStore.add_sma(df, [20, 50, 200])
        df = FeatureStore.add_ema(df, [12, 26])
        df = FeatureStore.add_rsi(df)
        df = FeatureStore.add_atr(df)
        df = FeatureStore.add_bollinger(df)
        df = FeatureStore.add_zscore(df)
        df = FeatureStore.add_rolling_volatility(df)
        df = FeatureStore.add_macd(df)
        return df

    @staticmethod
    def add_returns(df: pd.DataFrame) -> pd.DataFrame:
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        return df

    @staticmethod
    def add_sma(df: pd.DataFrame, periods: list[int] = None) -> pd.DataFrame:
        periods = periods or [20, 50, 200]
        for p in periods:
            df[f"sma_{p}"] = df["close"].rolling(p).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, periods: list[int] = None) -> pd.DataFrame:
        periods = periods or [12, 26]
        for p in periods:
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(period).mean()
        return df

    @staticmethod
    def add_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        df["bb_upper"] = sma + num_std * std
        df["bb_middle"] = sma
        df["bb_lower"] = sma - num_std * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        return df

    @staticmethod
    def add_zscore(df: pd.DataFrame, period: int = 60) -> pd.DataFrame:
        rolling_mean = df["close"].rolling(period).mean()
        rolling_std = df["close"].rolling(period).std()
        df["zscore"] = (df["close"] - rolling_mean) / rolling_std.replace(0, np.nan)
        return df

    @staticmethod
    def add_rolling_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        if "returns" not in df.columns:
            df["returns"] = df["close"].pct_change()
        df["rolling_volatility"] = df["returns"].rolling(period).std()
        df["rolling_volatility_annualised"] = df["rolling_volatility"] * np.sqrt(252)
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    @staticmethod
    def add_spread_zscore(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        period: int = 60,
    ) -> pd.Series:
        """Compute z-score of the price spread between two instruments."""
        spread = df_a["close"] - df_b["close"]
        rolling_mean = spread.rolling(period).mean()
        rolling_std = spread.rolling(period).std()
        return (spread - rolling_mean) / rolling_std.replace(0, np.nan)

    @staticmethod
    def add_ratio_zscore(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        period: int = 60,
    ) -> pd.Series:
        """Compute z-score of the price ratio between two instruments."""
        ratio = df_a["close"] / df_b["close"].replace(0, np.nan)
        rolling_mean = ratio.rolling(period).mean()
        rolling_std = ratio.rolling(period).std()
        return (ratio - rolling_mean) / rolling_std.replace(0, np.nan)
