"""Risk manager — orchestrates all risk checks as a hard gate."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from loguru import logger

from klaus.config.settings import get_settings
from klaus.core.types import Signal, TradeRequest, TradeResult, Direction, OrderStatus
from klaus.data.mt5_client import MT5Client
from klaus.risk.correlation_filter import CorrelationFilter
from klaus.risk.drawdown_control import DrawdownControl
from klaus.risk.position_sizer import PositionSizer


class RiskManager:
    """Evaluates signals against the full risk stack.

    A signal either passes (→ TradeRequest) or is blocked (→ None).
    No partial adjustments — this is a hard gate.

    Checks (in order):
      1. Free margin check (reject early if < 10% free)
      2. Drawdown halt
      3. Position limits (total and per-instrument)
      4. Correlation filter
      5. Position sizing (half-Kelly + ATR)
      6. SL/TP computation
      7. Margin pre-check (MT5 order_check before sending)
    """

    def __init__(self, client: MT5Client):
        self._client = client
        self._settings = get_settings().risk
        self._sizer = PositionSizer()
        self._drawdown = DrawdownControl()
        self._correlation = CorrelationFilter()
        # Trade history for Kelly updates
        self._trade_results: deque[float] = deque(maxlen=200)

    def evaluate(self, signal: Signal) -> Optional[TradeRequest]:
        """Evaluate a signal through the risk stack.

        Returns a TradeRequest if approved, None if blocked.
        """
        # Get account state
        account = self._client.get_account_info()
        equity = account.equity
        free_margin = account.free_margin

        # 1. Free margin pre-check — reject early if nearly out
        if equity > 0 and free_margin < equity * 0.05:
            logger.info(
                f"Signal blocked (low free margin ${free_margin:.2f} < 5% of "
                f"${equity:.2f}): {signal.symbol} {signal.algo_name}"
            )
            return None

        # 2. Drawdown check
        size_factor = self._drawdown.size_adjustment(equity)
        if size_factor <= 0:
            logger.info(f"Signal blocked (drawdown halt): {signal.symbol} {signal.algo_name}")
            return None

        # 3. Position limits
        positions = self._client.get_positions()

        if len(positions) >= self._settings.max_positions:
            logger.info(f"Signal blocked (max positions {self._settings.max_positions}): {signal.symbol}")
            return None

        # Per-instrument limit
        inst_positions = [p for p in positions if p.symbol == signal.symbol]
        if len(inst_positions) >= self._settings.max_per_instrument:
            logger.info(f"Signal blocked (max per instrument): {signal.symbol}")
            return None

        # 4. Correlation filter
        if not self._correlation.check(signal.symbol, positions):
            return None

        # 5. Position sizing
        symbol_info = self._client.get_symbol_info(signal.symbol)
        if symbol_info is None:
            logger.warning(f"Cannot get symbol info for {signal.symbol}")
            return None

        # Get ATR from latest data
        atr = signal.metadata.get("atr")
        if atr is None:
            # Fetch fresh data to compute ATR
            from klaus.data.market_data import MarketData
            from klaus.data.feature_store import FeatureStore
            md = MarketData(self._client)
            df = md.get_bars(signal.symbol, "1h", 20)
            if not df.empty:
                df = FeatureStore.add_atr(df)
                atr = df["atr"].iloc[-1]

        if atr is None or np.isnan(atr) or atr <= 0:
            logger.warning(f"Invalid ATR for {signal.symbol}, using fallback")
            tick = self._client.get_tick(signal.symbol)
            if tick is None:
                return None
            atr = tick["bid"] * 0.01  # 1% fallback

        volume = self._sizer.calculate_volume(equity, atr, signal.strength, symbol_info)

        # Apply drawdown size factor
        volume *= size_factor
        vol_step = symbol_info.get("volume_step", 0.01)
        volume = round(volume / vol_step) * vol_step
        volume = max(volume, symbol_info.get("volume_min", 0.01))

        if volume <= 0:
            logger.info(f"Signal blocked (zero volume after sizing): {signal.symbol}")
            return None

        # 6. Margin reservation — don't let one trade hog all the free margin.
        #    Each trade may use at most free_margin / remaining_slots so there's
        #    room for other positions. Estimate margin via contract size.
        open_count = len(positions)
        max_pos = self._settings.max_positions
        remaining_slots = max(max_pos - open_count, 1)
        max_margin_for_this_trade = free_margin / remaining_slots

        # Estimate margin required: volume * contract_size * price / leverage
        # Since we don't know leverage directly, use MT5's order_check.
        # But first do a rough cap: scale volume down if it would use too much.
        contract_size = symbol_info.get("trade_contract_size", 100_000)
        tick = self._client.get_tick(signal.symbol)
        if tick is None:
            return None
        est_price = tick["ask"] if signal.direction == Direction.LONG else tick["bid"]

        # Try progressively smaller volumes until margin check passes
        vol_min = symbol_info.get("volume_min", 0.01)
        vol_step = symbol_info.get("volume_step", 0.01)
        original_volume = volume

        while volume >= vol_min:
            ok, reason = self._client.check_margin(signal.symbol, signal.direction, volume)
            if ok:
                break
            volume -= vol_step
            volume = round(volume / vol_step) * vol_step

        if volume < vol_min and not ok:
            logger.info(
                f"Signal blocked (insufficient margin even at min vol): "
                f"{signal.symbol} {signal.direction.name} | "
                f"free_margin=${free_margin:.2f} reason={reason}"
            )
            return None

        if volume != original_volume:
            logger.info(
                f"Volume reduced {original_volume:.2f} -> {volume:.2f} lots "
                f"(margin constraint, {remaining_slots} slots remaining)"
            )

        # 7. SL / TP (reuse tick from margin check above)
        price = est_price
        point = symbol_info.get("point", 0.01)
        digits = symbol_info.get("digits", 2)

        sl_distance = atr * self._settings.default_sl_atr_mult
        tp_distance = atr * self._settings.default_tp_atr_mult

        if signal.direction == Direction.LONG:
            stop_loss = round(price - sl_distance, digits)
            take_profit = round(price + tp_distance, digits)
        else:
            stop_loss = round(price + sl_distance, digits)
            take_profit = round(price - tp_distance, digits)

        trade_request = TradeRequest(
            symbol=signal.symbol,
            direction=signal.direction,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            algo_name=signal.algo_name,
            signal_strength=signal.strength,
            comment=f"Klaus|{signal.algo_name}|{signal.strength:.2f}",
        )

        logger.info(
            f"Trade approved: {trade_request.symbol} {trade_request.direction.name} "
            f"vol={trade_request.volume} SL={trade_request.stop_loss} TP={trade_request.take_profit} "
            f"| free_margin=${free_margin:.2f}"
        )

        return trade_request

    def record_trade_result(self, pnl: float) -> None:
        """Record a closed trade's P&L and update Kelly statistics."""
        self._trade_results.append(pnl)
        if len(self._trade_results) >= 10:
            wins = [p for p in self._trade_results if p > 0]
            losses = [p for p in self._trade_results if p <= 0]
            win_rate = len(wins) / len(self._trade_results)
            avg_win = np.mean(wins) if wins else 1.0
            avg_loss = abs(np.mean(losses)) if losses else 1.0
            self._sizer.update_statistics(win_rate, avg_win, avg_loss)
            logger.info(
                f"Kelly updated: {len(self._trade_results)} trades, "
                f"WR={win_rate:.1%}, avg_win={avg_win:.2f}, avg_loss={avg_loss:.2f}"
            )

    def update_correlation_data(self, symbol: str, returns) -> None:
        """Feed returns data to the correlation filter."""
        self._correlation.update_returns(symbol, returns)
