"""Trailing stop manager — moves SL to lock in profits on open HFT positions.

Each HFT cycle, checks all open positions and:
  1. Breakeven: once profit >= 25% of TP distance, move SL to entry price
  2. Trail: once profit >= 40% of TP distance, trail SL at 50% of current profit
  3. Stale exit: force-close positions open longer than max_hold_seconds
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from klaus.core.types import Direction, Position
from klaus.data.mt5_client import MT5Client


class TrailingStopManager:
    """Manages trailing stops for open HFT positions."""

    def __init__(self, client: MT5Client, config: dict = None):
        self._client = client
        cfg = config or {}
        self._enabled = cfg.get("enabled", True)
        self._activation_pct = cfg.get("activation_pct", 40) / 100.0
        self._trail_pct = cfg.get("trail_pct", 50) / 100.0
        self._breakeven_pct = cfg.get("breakeven_pct", 25) / 100.0
        self._max_hold_seconds = cfg.get("max_hold_seconds", 300)

        # Track which positions we've already moved to breakeven
        self._breakeven_set: set[int] = set()

    def manage(self) -> None:
        """Check all open HFT positions and adjust SL."""
        if not self._enabled:
            return

        positions = self._client.get_positions()
        hft_positions = [
            p for p in positions
            if p.algo_name.startswith("K|") or "Klaus" in p.algo_name or "hft" in p.algo_name.lower()
        ]

        now = datetime.utcnow()

        for pos in hft_positions:
            try:
                self._manage_position(pos, now)
            except Exception as e:
                logger.error(f"TrailingStop error ticket={pos.ticket}: {e}")

        # Clean up breakeven set for closed positions
        open_tickets = {p.ticket for p in hft_positions}
        self._breakeven_set = self._breakeven_set & open_tickets

    def _manage_position(self, pos: Position, now: datetime) -> None:
        """Manage a single position's trailing stop."""
        # Calculate distances
        tp_distance = abs(pos.take_profit - pos.open_price)
        if tp_distance <= 0:
            return

        tick = self._client.get_tick(pos.symbol)
        if tick is None:
            return

        current_price = tick["bid"] if pos.direction == Direction.LONG else tick["ask"]

        if pos.direction == Direction.LONG:
            current_profit_dist = current_price - pos.open_price
        else:
            current_profit_dist = pos.open_price - current_price

        profit_ratio = current_profit_dist / tp_distance if tp_distance > 0 else 0

        # 1. Stale trade exit — force close if held too long
        hold_time = (now - pos.open_time).total_seconds()
        if hold_time >= self._max_hold_seconds:
            logger.info(
                f"Trailing: force-closing stale trade ticket={pos.ticket} "
                f"{pos.symbol} held {hold_time:.0f}s (pnl={pos.unrealized_pnl:.2f})"
            )
            self._client.close_position(pos.ticket)
            return

        # Only trail if in profit
        if current_profit_dist <= 0:
            return

        symbol_info = self._client.get_symbol_info(pos.symbol)
        digits = symbol_info.get("digits", 2) if symbol_info else 2

        # 2. Breakeven — move SL to entry once 25% of TP reached
        if profit_ratio >= self._breakeven_pct and pos.ticket not in self._breakeven_set:
            # Add a small buffer (1 point) so we don't get stopped at exact entry
            point = symbol_info.get("point", 0.01) if symbol_info else 0.01
            if pos.direction == Direction.LONG:
                new_sl = round(pos.open_price + point, digits)
            else:
                new_sl = round(pos.open_price - point, digits)

            # Only move SL if it's better than current
            if self._is_better_sl(pos, new_sl):
                self._client.modify_position(pos.ticket, stop_loss=new_sl)
                self._breakeven_set.add(pos.ticket)
                logger.info(
                    f"Trailing: breakeven ticket={pos.ticket} {pos.symbol} "
                    f"SL→{new_sl} (profit {profit_ratio:.0%} of TP)"
                )

        # 3. Trail — lock in 50% of current profit once 40% of TP reached
        if profit_ratio >= self._activation_pct:
            lock_in = current_profit_dist * self._trail_pct
            if pos.direction == Direction.LONG:
                new_sl = round(pos.open_price + lock_in, digits)
            else:
                new_sl = round(pos.open_price - lock_in, digits)

            if self._is_better_sl(pos, new_sl):
                self._client.modify_position(pos.ticket, stop_loss=new_sl)
                logger.info(
                    f"Trailing: SL trailed ticket={pos.ticket} {pos.symbol} "
                    f"SL→{new_sl} (locking {self._trail_pct:.0%} of {current_profit_dist:.4f} profit)"
                )

    @staticmethod
    def _is_better_sl(pos: Position, new_sl: float) -> bool:
        """Check if new SL is tighter (more protective) than current."""
        if pos.stop_loss == 0:
            return True
        if pos.direction == Direction.LONG:
            return new_sl > pos.stop_loss
        else:
            return new_sl < pos.stop_loss
