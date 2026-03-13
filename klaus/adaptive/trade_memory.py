"""Persistent trade memory — SQLite database of every trade and its outcome.

Survives restarts. The system never forgets — it accumulates knowledge over
its entire lifetime. Old trades naturally lose influence via the exponential
decay in PerformanceTracker, but the raw data stays forever for analysis.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "adaptive" / "trade_memory.db"


class TradeMemory:
    """SQLite-backed persistent storage of all trade outcomes."""

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        count = self._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        logger.info(f"TradeMemory loaded: {count} historical trades from {self._db_path.name}")

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket      INTEGER,
                symbol      TEXT NOT NULL,
                direction   TEXT NOT NULL,
                algo_name   TEXT NOT NULL,
                regime      TEXT NOT NULL DEFAULT 'UNKNOWN',
                engine_type TEXT NOT NULL DEFAULT 'standard',

                -- Entry details
                volume          REAL NOT NULL,
                open_price      REAL NOT NULL,
                open_time       TEXT NOT NULL,
                signal_strength REAL NOT NULL DEFAULT 0.5,

                -- Exit details (filled when trade closes)
                close_price REAL,
                close_time  TEXT,
                pnl         REAL,
                duration_s  REAL,

                -- Context at time of trade
                atr         REAL,
                volatility  REAL,
                features    TEXT,  -- JSON blob of key features at signal time

                -- Outcome classification
                outcome     TEXT,  -- 'win', 'loss', 'breakeven', NULL if still open

                closed      INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_trades_algo ON trades(algo_name);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_regime ON trades(regime);
            CREATE INDEX IF NOT EXISTS idx_trades_closed ON trades(closed);
            CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades(open_time);
            CREATE INDEX IF NOT EXISTS idx_trades_composite ON trades(algo_name, symbol, regime, closed);
        """)
        self._conn.commit()

    def record_open(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        algo_name: str,
        volume: float,
        open_price: float,
        signal_strength: float = 0.5,
        regime: str = "UNKNOWN",
        engine_type: str = "standard",
        atr: float = None,
        volatility: float = None,
        features: dict = None,
    ) -> int:
        """Record a new trade opening. Returns the row ID."""
        cursor = self._conn.execute(
            """INSERT INTO trades
               (ticket, symbol, direction, algo_name, regime, engine_type,
                volume, open_price, open_time, signal_strength,
                atr, volatility, features, closed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                ticket, symbol, direction, algo_name, regime, engine_type,
                volume, open_price, datetime.utcnow().isoformat(),
                signal_strength, atr, volatility,
                json.dumps(features) if features else None,
            ),
        )
        self._conn.commit()
        row_id = cursor.lastrowid
        logger.debug(f"TradeMemory: recorded open #{ticket} {symbol} {algo_name}")
        return row_id

    def record_close(
        self,
        ticket: int,
        close_price: float,
        pnl: float,
    ) -> Optional[dict]:
        """Record trade closure with P/L. Returns the full trade record."""
        row = self._conn.execute(
            "SELECT * FROM trades WHERE ticket = ? AND closed = 0",
            (ticket,),
        ).fetchone()

        if row is None:
            logger.debug(f"TradeMemory: ticket #{ticket} not found or already closed")
            return None

        open_time = datetime.fromisoformat(row["open_time"])
        close_time = datetime.utcnow()
        duration_s = (close_time - open_time).total_seconds()

        if pnl > 0:
            outcome = "win"
        elif pnl < 0:
            outcome = "loss"
        else:
            outcome = "breakeven"

        self._conn.execute(
            """UPDATE trades
               SET close_price = ?, close_time = ?, pnl = ?,
                   duration_s = ?, outcome = ?, closed = 1
               WHERE ticket = ? AND closed = 0""",
            (close_price, close_time.isoformat(), pnl, duration_s, outcome, ticket),
        )
        self._conn.commit()

        logger.debug(
            f"TradeMemory: closed #{ticket} {row['symbol']} "
            f"{row['algo_name']} pnl={pnl:.2f} ({outcome})"
        )

        return dict(row) | {
            "close_price": close_price,
            "close_time": close_time.isoformat(),
            "pnl": pnl,
            "duration_s": duration_s,
            "outcome": outcome,
        }

    def get_algo_trades(
        self,
        algo_name: str,
        symbol: str = None,
        regime: str = None,
        closed_only: bool = True,
        limit: int = 500,
    ) -> list[dict]:
        """Get trades for an algorithm, optionally filtered by symbol/regime."""
        query = "SELECT * FROM trades WHERE algo_name = ?"
        params: list = [algo_name]

        if closed_only:
            query += " AND closed = 1"
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if regime:
            query += " AND regime = ?"
            params.append(regime)

        query += " ORDER BY open_time DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_recent_closed(self, limit: int = 100) -> list[dict]:
        """Get the N most recently closed trades."""
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE closed = 1 ORDER BY close_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_open_trades(self) -> list[dict]:
        """Get all currently open trades."""
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE closed = 0 ORDER BY open_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_algo_stats(self, algo_name: str, symbol: str = None, regime: str = None) -> dict:
        """Quick aggregate stats for an algorithm."""
        query = "SELECT * FROM trades WHERE algo_name = ? AND closed = 1"
        params: list = [algo_name]
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if regime:
            query += " AND regime = ?"
            params.append(regime)

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return {"trades": 0, "win_rate": 0.5, "avg_pnl": 0.0, "total_pnl": 0.0}

        pnls = [r["pnl"] for r in rows]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "trades": len(rows),
            "win_rate": wins / len(rows),
            "avg_pnl": sum(pnls) / len(pnls),
            "total_pnl": sum(pnls),
            "best": max(pnls),
            "worst": min(pnls),
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
