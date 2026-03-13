"""Cross-instrument correlation filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from klaus.config.settings import get_settings
from klaus.core.types import Position


class CorrelationFilter:
    """Prevents excessive exposure to correlated instruments.

    Rule: Max N positions with pairwise correlation > threshold.
    """

    def __init__(self):
        self._settings = get_settings().risk
        self._correlation_matrix: pd.DataFrame = pd.DataFrame()
        self._returns_cache: dict[str, pd.Series] = {}

    def update_returns(self, symbol: str, returns: pd.Series) -> None:
        """Feed in recent returns for correlation computation."""
        self._returns_cache[symbol] = returns.dropna().tail(120)
        self._recompute_matrix()

    def _recompute_matrix(self) -> None:
        if len(self._returns_cache) < 2:
            return
        df = pd.DataFrame(self._returns_cache)
        self._correlation_matrix = df.corr()

    def check(self, new_symbol: str, open_positions: list[Position]) -> bool:
        """Return True if the new trade is allowed, False if blocked by correlation.

        Blocked if adding this position would result in more than
        max_correlated_positions with correlation > threshold.
        """
        if self._correlation_matrix.empty:
            return True

        if new_symbol not in self._correlation_matrix.columns:
            return True

        open_symbols = [p.symbol for p in open_positions]
        correlated_count = 0

        for sym in open_symbols:
            if sym not in self._correlation_matrix.columns:
                continue
            corr = abs(self._correlation_matrix.loc[new_symbol, sym])
            if corr > self._settings.correlation_threshold:
                correlated_count += 1

        if correlated_count >= self._settings.max_correlated_positions:
            logger.info(
                f"Correlation filter blocked {new_symbol}: "
                f"{correlated_count} correlated positions (threshold={self._settings.correlation_threshold})"
            )
            return False

        return True
