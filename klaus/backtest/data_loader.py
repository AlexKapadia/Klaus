"""Historical data loader for backtesting."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from klaus.config.settings import PROJECT_ROOT
from klaus.data.mt5_client import MT5Client

CACHE_DIR = PROJECT_ROOT / "data" / "cache"


class HistoricalDataLoader:
    """Loads historical bar data from MT5 or cached Parquet files."""

    def __init__(self, client: Optional[MT5Client] = None):
        self._client = client
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        count: int = 5000,
    ) -> pd.DataFrame:
        """Load historical data, preferring cache, falling back to MT5.

        Args:
            symbol: Instrument symbol.
            timeframe: Bar timeframe.
            start_date: Optional start date for the data.
            end_date: Optional end date (for trimming).
            count: Number of bars to fetch from MT5 if no cache.
        """
        # Try cached file first
        cache_path = CACHE_DIR / f"{symbol}_{timeframe}_hist.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded {len(df)} bars from cache: {cache_path.name}")
        elif self._client is not None and self._client.is_connected:
            df = self._client.get_bars(symbol, timeframe, count, from_date=start_date)
            if not df.empty:
                df.to_parquet(cache_path)
                logger.info(f"Fetched {len(df)} bars from MT5, cached to {cache_path.name}")
        else:
            logger.warning(f"No data available for {symbol} {timeframe}")
            return pd.DataFrame()

        # Trim to date range
        if start_date and not df.empty:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date and not df.empty:
            df = df[df.index <= pd.Timestamp(end_date)]

        return df

    def load_from_csv(self, path: str | Path) -> pd.DataFrame:
        """Load data from a CSV file (for external data sources)."""
        path = Path(path)
        if not path.exists():
            logger.error(f"CSV file not found: {path}")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=True, index_col=0)

        # Standardise column names
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if "open" in lower:
                col_map[col] = "open"
            elif "high" in lower:
                col_map[col] = "high"
            elif "low" in lower:
                col_map[col] = "low"
            elif "close" in lower:
                col_map[col] = "close"
            elif "vol" in lower:
                col_map[col] = "volume"
        df.rename(columns=col_map, inplace=True)

        return df
