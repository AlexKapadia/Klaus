"""Performance metrics computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from klaus.execution.order_tracker import OrderTracker


@dataclass
class PerformanceSnapshot:
    total_trades: int = 0
    open_trades: int = 0
    closed_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0


class MetricsCalculator:
    """Computes trading performance metrics from order history."""

    def __init__(self, tracker: OrderTracker):
        self._tracker = tracker

    def snapshot(self) -> PerformanceSnapshot:
        """Compute current performance snapshot."""
        closed = self._tracker.closed_trades
        opened = self._tracker.open_trades

        snap = PerformanceSnapshot()
        snap.open_trades = len(opened)
        snap.closed_trades = len(closed)
        snap.total_trades = snap.open_trades + snap.closed_trades

        if not closed:
            return snap

        pnls = [t.pnl for t in closed]
        snap.total_pnl = sum(pnls)
        snap.avg_trade_pnl = snap.total_pnl / len(pnls)
        snap.best_trade = max(pnls)
        snap.worst_trade = min(pnls)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        snap.win_rate = len(wins) / len(pnls) if pnls else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        snap.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe (simplified)
        pnl_series = pd.Series(pnls)
        if pnl_series.std() > 0:
            snap.sharpe_ratio = float(pnl_series.mean() / pnl_series.std() * np.sqrt(252))

        # Max drawdown from cumulative PnL
        cum_pnl = pnl_series.cumsum()
        peak = cum_pnl.cummax()
        dd = cum_pnl - peak
        if peak.max() > 0:
            snap.max_drawdown_pct = float(abs(dd.min()) / peak.max() * 100)

        return snap

    def print_report(self) -> None:
        """Log a formatted performance report."""
        s = self.snapshot()
        logger.info("=" * 50)
        logger.info("  KLAUS PERFORMANCE REPORT")
        logger.info("=" * 50)
        logger.info(f"  Total trades:     {s.total_trades}")
        logger.info(f"  Open / Closed:    {s.open_trades} / {s.closed_trades}")
        logger.info(f"  Win rate:         {s.win_rate:.1%}")
        logger.info(f"  Total P&L:        {s.total_pnl:,.2f}")
        logger.info(f"  Avg trade P&L:    {s.avg_trade_pnl:,.2f}")
        logger.info(f"  Best trade:       {s.best_trade:,.2f}")
        logger.info(f"  Worst trade:      {s.worst_trade:,.2f}")
        logger.info(f"  Profit factor:    {s.profit_factor:.2f}")
        logger.info(f"  Sharpe ratio:     {s.sharpe_ratio:.2f}")
        logger.info(f"  Max drawdown:     {s.max_drawdown_pct:.1f}%")
        logger.info("=" * 50)
