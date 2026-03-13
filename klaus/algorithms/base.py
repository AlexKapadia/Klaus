"""Abstract base class for all trading algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from klaus.core.types import Regime, Signal


class BaseAlgorithm(ABC):
    """Every algorithm must inherit from this and implement generate_signal().

    Class attributes (must be set by subclasses):
        name:                   Unique identifier, e.g. "sma_crossover"
        supported_instruments:  List of symbols, or ["*"] for all
        preferred_regimes:      Which regimes this algorithm is suited for
        min_bars_required:      Minimum rows of data needed
    """

    name: str = ""
    supported_instruments: list[str] = ["*"]
    preferred_regimes: list[Regime] = []
    min_bars_required: int = 200

    def __init__(self, params: dict = None):
        self.params = params or {}

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Analyse data and return a Signal, or None if no opinion.

        Args:
            data: OHLCV DataFrame with features already added.
            symbol: The instrument symbol.

        Returns:
            Signal with direction and strength, or None.
        """
        ...

    def fit(self, data: pd.DataFrame) -> None:
        """Train the algorithm on historical data (ML algos override this)."""
        pass

    def warm_up(self, data: pd.DataFrame) -> None:
        """Precompute indicators or state needed before signal generation."""
        pass

    def can_trade(self, symbol: str) -> bool:
        """Check if this algorithm supports the given instrument."""
        if "*" in self.supported_instruments:
            return True
        return symbol in self.supported_instruments

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
