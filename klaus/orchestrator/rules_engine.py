"""Rules engine: maps (instrument, regime) → list of algorithm names."""

from __future__ import annotations

from loguru import logger

from klaus.config.settings import get_settings
from klaus.core.types import Regime


class RulesEngine:
    """Determines which algorithms to run for a given instrument and regime.

    Reads the mapping from regimes.yaml:
        mapping:
          XAUUSD:
            TRENDING: [sma_crossover, lstm_signal]
            MEAN_REVERTING: [bollinger, gold_silver_ratio]
            VOLATILE: []
    """

    def __init__(self):
        self._mapping: dict[str, dict[str, list[str]]] = {}
        self._load_mapping()

    def _load_mapping(self) -> None:
        settings = get_settings()
        raw = settings.regime_algo_mapping

        for symbol, regime_map in raw.items():
            self._mapping[symbol] = {}
            for regime_str, algos in regime_map.items():
                self._mapping[symbol][regime_str] = algos or []

        logger.info(f"RulesEngine loaded mappings for {len(self._mapping)} instruments")

    def get_algorithms(self, symbol: str, regime: Regime) -> list[str]:
        """Return algorithm names to run for the given instrument and regime."""
        regime_str = regime.name  # e.g. "TRENDING"

        if symbol not in self._mapping:
            logger.debug(f"No mapping for {symbol}, skipping")
            return []

        algos = self._mapping[symbol].get(regime_str, [])

        if not algos:
            logger.debug(f"No algorithms for {symbol} in {regime_str} regime")

        return algos

    def get_all_symbols(self) -> list[str]:
        """Return all symbols that have regime mappings."""
        return list(self._mapping.keys())

    def reload(self) -> None:
        """Reload mapping from config (e.g. after YAML edit)."""
        self._mapping.clear()
        self._load_mapping()
