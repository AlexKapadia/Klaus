"""Fetch, cache, and resample market data from MT5."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from klaus.config.settings import PROJECT_ROOT
from klaus.data.mt5_client import MT5Client

CACHE_DIR = PROJECT_ROOT / "data" / "cache"


class MarketData:
    """Manages bar data retrieval with Parquet caching."""

    def __init__(self, client: MT5Client):
        self._client = client
        self._cache: dict[str, pd.DataFrame] = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        count: int = 500,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Get bars, using in-memory cache with disk fallback."""
        cache_key = f"{symbol}_{timeframe}"

        # Try in-memory cache first
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if len(cached) >= count:
                return cached.tail(count).copy()

        # Fetch from MT5
        df = self._client.get_bars(symbol, timeframe, count)

        if df.empty:
            # Fall back to disk cache
            disk_df = self._load_from_disk(symbol, timeframe)
            if not disk_df.empty:
                logger.info(f"Using disk cache for {symbol} {timeframe}")
                return disk_df.tail(count).copy()
            return df

        # Update caches
        self._cache[cache_key] = df
        self._save_to_disk(df, symbol, timeframe)

        return df

    def refresh(self, symbol: str, timeframe: str = "1h", count: int = 500) -> pd.DataFrame:
        """Force-refresh from MT5, bypassing cache."""
        return self.get_bars(symbol, timeframe, count, use_cache=False)

    def get_multi_timeframe(
        self,
        symbol: str,
        timeframes: list[str],
        count: int = 500,
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars for multiple timeframes."""
        return {tf: self.get_bars(symbol, tf, count) for tf in timeframes}

    def resample(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample OHLCV data to a lower frequency."""
        rule_map = {
            "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "4h": "4h", "1d": "1D", "1w": "1W",
        }
        rule = rule_map.get(target_tf)
        if rule is None:
            raise ValueError(f"Cannot resample to {target_tf}")

        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return resampled

    def _save_to_disk(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        path = CACHE_DIR / f"{symbol}_{timeframe}.parquet"
        try:
            df.to_parquet(path)
        except Exception as e:
            logger.warning(f"Failed to cache {symbol} {timeframe}: {e}")

    def _load_from_disk(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = CACHE_DIR / f"{symbol}_{timeframe}.parquet"
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Failed to load cache {symbol} {timeframe}: {e}")
            return pd.DataFrame()
