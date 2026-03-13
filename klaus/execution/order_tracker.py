"""Tracks open orders, fills, and P&L history."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

from klaus.core.types import Direction, TradeResult, OrderStatus


@dataclass
class TradeRecord:
    """Complete record of a round-trip trade."""
    ticket: int
    symbol: str
    direction: Direction
    volume: float
    open_price: float
    open_time: datetime
    algo_name: str
    close_price: Optional[float] = None
    close_time: Optional[datetime] = None
    pnl: float = 0.0
    closed: bool = False


class OrderTracker:
    """Maintains history of all trades for performance analysis."""

    def __init__(self):
        self._open_trades: dict[int, TradeRecord] = {}
        self._closed_trades: list[TradeRecord] = []

    def record_open(self, result: TradeResult, algo_name: str) -> None:
        """Record a new trade fill."""
        if result.status != OrderStatus.FILLED:
            return

        record = TradeRecord(
            ticket=result.ticket,
            symbol=result.symbol,
            direction=result.direction,
            volume=result.volume,
            open_price=result.price,
            open_time=result.timestamp,
            algo_name=algo_name,
        )
        self._open_trades[result.ticket] = record
        logger.debug(f"OrderTracker: opened #{result.ticket} {result.symbol}")

    def record_close(self, ticket: int, close_price: float) -> Optional[TradeRecord]:
        """Record a trade closure and compute P&L."""
        if ticket not in self._open_trades:
            logger.warning(f"OrderTracker: ticket #{ticket} not found in open trades")
            return None

        record = self._open_trades.pop(ticket)
        record.close_price = close_price
        record.close_time = datetime.utcnow()
        record.closed = True

        # Compute P&L (simplified — does not account for contract size)
        if record.direction == Direction.LONG:
            record.pnl = (close_price - record.open_price) * record.volume
        else:
            record.pnl = (record.open_price - close_price) * record.volume

        self._closed_trades.append(record)
        logger.debug(f"OrderTracker: closed #{ticket} PnL={record.pnl:.2f}")
        return record

    @property
    def open_trades(self) -> dict[int, TradeRecord]:
        return dict(self._open_trades)

    @property
    def closed_trades(self) -> list[TradeRecord]:
        return list(self._closed_trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self._closed_trades)

    def win_rate(self) -> float:
        """Compute win rate from closed trades."""
        if not self._closed_trades:
            return 0.5
        wins = sum(1 for t in self._closed_trades if t.pnl > 0)
        return wins / len(self._closed_trades)

    def avg_win_loss(self) -> tuple[float, float]:
        """Return (average_win, average_loss) magnitudes."""
        wins = [t.pnl for t in self._closed_trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in self._closed_trades if t.pnl <= 0]
        avg_win = sum(wins) / len(wins) if wins else 1.0
        avg_loss = sum(losses) / len(losses) if losses else 1.0
        return avg_win, avg_loss
