"""CPPI-based drawdown control."""

from __future__ import annotations

from loguru import logger

from klaus.config.settings import get_settings


class DrawdownControl:
    """Manages drawdown limits and CPPI exposure scaling.

    - Halt all trading if drawdown > max_drawdown_pct
    - Halve position sizes if drawdown > drawdown_reduce_pct
    - CPPI: exposure = multiplier * (equity - floor)
    """

    def __init__(self):
        self._settings = get_settings().risk
        self._peak_equity: float = 0.0

    def update_peak(self, equity: float) -> None:
        """Update the peak equity watermark."""
        if equity > self._peak_equity:
            self._peak_equity = equity

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    def current_drawdown_pct(self, equity: float) -> float:
        """Calculate current drawdown as a percentage of peak."""
        if self._peak_equity <= 0:
            return 0.0
        return ((self._peak_equity - equity) / self._peak_equity) * 100

    def should_halt(self, equity: float) -> bool:
        """True if drawdown exceeds the maximum — stop all trading."""
        dd = self.current_drawdown_pct(equity)
        if dd >= self._settings.max_drawdown_pct:
            logger.warning(f"DRAWDOWN HALT: {dd:.1f}% >= {self._settings.max_drawdown_pct}%")
            return True
        return False

    def should_reduce(self, equity: float) -> bool:
        """True if drawdown exceeds the reduction threshold — halve sizes."""
        dd = self.current_drawdown_pct(equity)
        return dd >= self._settings.drawdown_reduce_pct

    def cppi_multiplier(self, equity: float) -> float:
        """CPPI exposure multiplier.

        exposure = multiplier * (equity - floor)
        floor = cppi_floor_pct% of peak equity

        Returns a scaling factor 0-1 to apply to position sizes.
        """
        if self._peak_equity <= 0:
            return 1.0

        floor = self._peak_equity * (self._settings.cppi_floor_pct / 100)
        cushion = equity - floor

        if cushion <= 0:
            return 0.0

        exposure = self._settings.cppi_multiplier * cushion
        # Normalise to a 0-1 multiplier relative to equity
        multiplier = min(exposure / equity, 1.0) if equity > 0 else 0.0
        return multiplier

    def size_adjustment(self, equity: float) -> float:
        """Return a combined size adjustment factor (0 to 1).

        0.0 = halt (no trading)
        0.5 = drawdown reduction active
        Otherwise = CPPI-scaled factor
        """
        self.update_peak(equity)

        if self.should_halt(equity):
            return 0.0

        factor = self.cppi_multiplier(equity)

        if self.should_reduce(equity):
            factor *= 0.5
            logger.info(f"Drawdown reduction active: factor={factor:.3f}")

        return factor
