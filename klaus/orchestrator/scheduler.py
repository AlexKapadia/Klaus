"""Per-instrument timing control for the orchestrator."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from loguru import logger


class InstrumentScheduler:
    """Tracks when each instrument was last processed and when it's due next.

    - Signal generation runs every cycle (60s)
    - HMM regime prediction runs every H4 bar
    - HMM refit runs weekly
    """

    def __init__(
        self,
        cycle_interval: int = 60,
        regime_interval_hours: int = 4,
        refit_interval_days: int = 7,
    ):
        self.cycle_interval = cycle_interval
        self.regime_interval = timedelta(hours=regime_interval_hours)
        self.refit_interval = timedelta(days=refit_interval_days)

        self._last_regime_check: dict[str, datetime] = {}
        self._last_refit: dict[str, datetime] = {}
        self._last_signal: dict[str, datetime] = {}

    def should_check_regime(self, symbol: str) -> bool:
        """True if it's time to re-predict the regime for this instrument."""
        now = datetime.utcnow()
        last = self._last_regime_check.get(symbol)
        if last is None:
            return True
        return (now - last) >= self.regime_interval

    def mark_regime_checked(self, symbol: str) -> None:
        self._last_regime_check[symbol] = datetime.utcnow()

    def should_refit_hmm(self, symbol: str) -> bool:
        """True if the HMM model should be refit for this instrument."""
        now = datetime.utcnow()
        last = self._last_refit.get(symbol)
        if last is None:
            return True
        return (now - last) >= self.refit_interval

    def mark_hmm_refit(self, symbol: str) -> None:
        self._last_refit[symbol] = datetime.utcnow()

    def mark_signal_generated(self, symbol: str) -> None:
        self._last_signal[symbol] = datetime.utcnow()

    def get_status(self) -> dict:
        """Return timing status for all instruments (for monitoring)."""
        now = datetime.utcnow()
        status = {}
        all_symbols = set(
            list(self._last_regime_check.keys())
            + list(self._last_refit.keys())
            + list(self._last_signal.keys())
        )
        for symbol in all_symbols:
            regime_ago = (now - self._last_regime_check[symbol]).total_seconds() if symbol in self._last_regime_check else None
            refit_ago = (now - self._last_refit[symbol]).total_seconds() if symbol in self._last_refit else None
            status[symbol] = {
                "regime_seconds_ago": regime_ago,
                "refit_seconds_ago": refit_ago,
            }
        return status
