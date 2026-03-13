"""Core type definitions for the Klaus trading platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from typing import Optional


class Regime(Enum):
    TRENDING = auto()
    MEAN_REVERTING = auto()
    VOLATILE = auto()
    UNKNOWN = auto()


class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()


class OrderStatus(Enum):
    PENDING = auto()
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()


class Timeframe(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


@dataclass(frozen=True)
class Signal:
    """Output from an algorithm indicating a trading opinion."""
    symbol: str
    direction: Direction
    strength: float  # 0.0 to 1.0
    algo_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Signal strength must be 0-1, got {self.strength}")


@dataclass
class TradeRequest:
    """Sized and risk-checked order ready for execution."""
    symbol: str
    direction: Direction
    volume: float
    stop_loss: float
    take_profit: float
    algo_name: str
    signal_strength: float
    comment: str = ""


@dataclass
class Position:
    """Tracks an open position."""
    ticket: int
    symbol: str
    direction: Direction
    volume: float
    open_price: float
    open_time: datetime
    stop_loss: float
    take_profit: float
    algo_name: str
    unrealized_pnl: float = 0.0


@dataclass
class TradeResult:
    """Result of an executed trade."""
    ticket: int
    symbol: str
    direction: Direction
    volume: float
    price: float
    status: OrderStatus
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_code: int = 0
    error_message: str = ""


@dataclass
class AccountInfo:
    """Snapshot of account state."""
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    open_positions: int
    peak_equity: float = 0.0
