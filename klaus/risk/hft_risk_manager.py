"""HFT micro-scalping risk manager — many tiny trades, conservative P/L.

Academic basis: Aldridge (2013), Cartea/Jaimungal/Penalva (2015).
- Micro lots (0.01-0.05) for minimal per-trade exposure
- Tight SL/TP (~0.5/0.6 ATR) for small, predictable P/L
- High trade frequency (hundreds/day) — statistical edge compounds
- Balanced long/short (sell profile = buy profile)
- Conservative daily loss cap (3%) to protect capital
- Spread-aware: TP must exceed transaction costs
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from loguru import logger

from klaus.config.settings import get_settings
from klaus.core.types import Signal, TradeRequest, Direction
from klaus.data.mt5_client import MT5Client
from klaus.risk.drawdown_control import DrawdownControl
from klaus.risk.position_sizer import PositionSizer


class HFTRiskManager:
    """Risk manager tailored for high-frequency commodity trading.

    Tighter risk per trade, max daily trade count, instrument cooldowns,
    and transaction cost budgeting.
    """

    def __init__(self, client: MT5Client, hft_config: dict = None):
        self._client = client
        self._settings = get_settings().risk
        self._hft = hft_config or {}

        self._sizer = PositionSizer()
        self._drawdown = DrawdownControl()

        # Initialize peak from current equity so we only track drawdown
        # from THIS session — not from historical balance highs
        try:
            account = self._client.get_account_info()
            self._drawdown.update_peak(account.equity)
            logger.info(f"HFT drawdown baseline set: equity=${account.equity:.2f}")
        except Exception:
            pass  # Will be set on first evaluate() call

        # HFT-specific parameters (from config)
        self._max_trades_per_day = self._hft.get("max_trades_per_day", 50)
        self._max_hft_positions_cfg = self._hft.get("max_hft_positions", 5)
        self._max_per_instrument_cfg = self._hft.get("max_per_instrument_hft", 2)
        self._cooldown_seconds = self._hft.get("cooldown_seconds", 60)
        self._max_risk_per_trade_pct = self._hft.get("max_risk_per_trade_pct", 0.2)
        self._default_sl_atr_mult = self._hft.get("default_sl_atr_mult", 0.8)
        self._default_tp_atr_mult = self._hft.get("default_tp_atr_mult", 1.2)
        self._max_daily_loss_pct = self._hft.get("max_daily_loss_pct", 2.0)
        self._spread_cost_buffer = self._hft.get("spread_cost_buffer", 1.5)
        self._max_volume_per_trade = self._hft.get("max_volume_per_trade", 2.0)
        self._min_atr_pct = self._hft.get("min_atr_pct", 0.5)

        # Growth mode config (auto-applied for small accounts)
        self._growth_mode = self._hft.get("growth_mode", {})
        self._growth_threshold = self._growth_mode.get("equity_threshold", 5000)
        self._min_signal_strength = self._hft.get("min_signal_strength", 0.0)

        # Auto-scale position limits based on account balance
        self._max_hft_positions = self._max_hft_positions_cfg
        self._max_per_instrument = self._max_per_instrument_cfg
        self._scale_limits_applied = False

        # Margin backoff config (configurable per engine)
        self._margin_backoff_base = self._hft.get("margin_backoff_base", 30)
        self._margin_backoff_max = self._hft.get("margin_backoff_max", 300)

        # Position prefix for filtering own positions (default "K|" for Klaus)
        self._position_prefix = self._hft.get("position_prefix", "K|")

        # Tracking state
        self._trades_today: list[datetime] = []
        self._last_trade_time: dict[str, datetime] = {}
        self._daily_pnl: float = 0.0
        self._day_start: datetime = datetime.utcnow().replace(hour=0, minute=0, second=0)

        # Margin rejection tracking — back off instruments that fail
        self._margin_rejects: dict[str, int] = {}  # symbol → consecutive reject count
        self._margin_blocked_until: dict[str, datetime] = {}  # symbol → unblock time

    def evaluate(self, signal: Signal) -> Optional[TradeRequest]:
        """Evaluate an HFT signal through the risk stack."""
        now = datetime.utcnow()

        # Reset daily counters
        if now.date() > self._day_start.date():
            self._trades_today.clear()
            self._daily_pnl = 0.0
            self._day_start = now.replace(hour=0, minute=0, second=0)

        account = self._client.get_account_info()
        equity = account.equity

        # Growth mode — auto-activate for small accounts, override risk params
        if not self._scale_limits_applied and equity > 0:
            if equity < self._growth_threshold and self._growth_mode:
                gm = self._growth_mode
                self._max_risk_per_trade_pct = gm.get("risk_per_trade_pct", self._max_risk_per_trade_pct)
                self._min_signal_strength = gm.get("min_signal_strength", 0.55)
                self._max_volume_per_trade = gm.get("max_volume_per_trade", 0.10)
                self._max_hft_positions = gm.get("max_hft_positions", 4)
                self._max_per_instrument = gm.get("max_per_instrument", 2)
                self._cooldown_seconds = gm.get("cooldown_seconds", 5)
                self._default_sl_atr_mult = gm.get("default_sl_atr_mult", 0.4)
                self._default_tp_atr_mult = gm.get("default_tp_atr_mult", 0.8)
                self._max_daily_loss_pct = gm.get("max_daily_loss_pct", 5.0)
                logger.info(
                    f"GROWTH MODE activated (equity=${equity:.0f} < ${self._growth_threshold}): "
                    f"risk={self._max_risk_per_trade_pct}%/trade, "
                    f"min_strength={self._min_signal_strength}, "
                    f"SL={self._default_sl_atr_mult}ATR, TP={self._default_tp_atr_mult}ATR, "
                    f"max_vol={self._max_volume_per_trade}, max_pos={self._max_hft_positions}"
                )
            self._scale_limits_applied = True

        # Check free margin before even evaluating — reject early if < 10% free
        if account.free_margin < equity * 0.10:
            logger.info(f"HFT blocked (low free margin ${account.free_margin:.2f} < ${equity*0.10:.2f}): {signal.symbol}")
            return None

        # Signal strength filter — skip weak signals in growth mode
        if signal.strength < self._min_signal_strength:
            logger.info(
                f"HFT blocked (strength {signal.strength:.3f} < {self._min_signal_strength}): "
                f"{signal.symbol} {signal.algo_name}"
            )
            return None

        # Check if this instrument is temporarily blocked due to margin rejections
        blocked_until = self._margin_blocked_until.get(signal.symbol)
        if blocked_until and now < blocked_until:
            remaining = (blocked_until - now).total_seconds()
            logger.info(f"HFT blocked (margin backoff {remaining:.0f}s remaining): {signal.symbol}")
            return None

        # 1. Drawdown check (shared with standard engine)
        size_factor = self._drawdown.size_adjustment(equity)
        if size_factor <= 0:
            logger.info(f"HFT blocked (drawdown halt): {signal.symbol}")
            return None

        # 2. Daily loss limit
        if equity > 0:
            daily_loss_pct = abs(min(self._daily_pnl, 0)) / equity * 100
            if daily_loss_pct >= self._max_daily_loss_pct:
                logger.info(f"HFT blocked (daily loss limit {daily_loss_pct:.1f}%): {signal.symbol}")
                return None

        # 3. Max trades per day
        self._trades_today = [t for t in self._trades_today if t.date() == now.date()]
        if len(self._trades_today) >= self._max_trades_per_day:
            logger.info(f"HFT blocked (max daily trades {self._max_trades_per_day})")
            return None

        # 4. Instrument cooldown
        last_trade = self._last_trade_time.get(signal.symbol)
        if last_trade and (now - last_trade).total_seconds() < self._cooldown_seconds:
            return None

        # 5. Position limits (only count THIS engine's positions)
        positions = self._client.get_positions()
        prefix = self._position_prefix
        hft_positions = [p for p in positions if p.algo_name.startswith(prefix)
                         or "hft" in p.algo_name.lower()]

        if len(hft_positions) >= self._max_hft_positions:
            logger.info(f"HFT blocked (max HFT positions {self._max_hft_positions})")
            return None

        inst_hft = [p for p in hft_positions if p.symbol == signal.symbol]
        if len(inst_hft) >= self._max_per_instrument:
            logger.info(f"HFT blocked (max per instrument): {signal.symbol}")
            return None

        # 6. Position sizing (tighter than standard)
        symbol_info = self._client.get_symbol_info(signal.symbol)
        if symbol_info is None:
            return None

        atr = signal.metadata.get("micro_atr") or signal.metadata.get("atr")
        tick = self._client.get_tick(signal.symbol)
        if tick is None:
            return None

        if atr is None or np.isnan(atr) or atr <= 0:
            atr = tick["bid"] * (self._min_atr_pct / 100)

        # Floor: ATR must be at least min_atr_pct% of price (prevents tiny SL → huge volume)
        min_atr = tick["bid"] * (self._min_atr_pct / 100)
        atr = max(atr, min_atr)

        # HFT risk budget: smaller per trade
        max_risk = equity * (self._max_risk_per_trade_pct / 100)
        contract_size = symbol_info.get("trade_contract_size", 100)

        # Use algorithm-specified SL multiplier if available
        sl_mult = signal.metadata.get("hft_sl_mult", self._default_sl_atr_mult)
        tp_mult = signal.metadata.get("hft_tp_mult", self._default_tp_atr_mult)

        sl_distance = atr * sl_mult
        risk_per_lot = sl_distance * contract_size

        if risk_per_lot <= 0:
            return None

        volume = max_risk / risk_per_lot
        volume *= size_factor * signal.strength

        # Hard cap on volume per trade
        volume = min(volume, self._max_volume_per_trade)

        # Round to volume step
        vol_step = symbol_info.get("volume_step", 0.01)
        vol_min = symbol_info.get("volume_min", 0.01)
        if vol_step > 0:
            volume = round(volume / vol_step) * vol_step
        volume = max(volume, vol_min)

        if volume <= 0:
            return None

        # 7. Spread cost check: TP must exceed spread to be profitable
        spread = tick["ask"] - tick["bid"]
        tp_distance = atr * tp_mult
        # If TP < spread cost, widen TP to just clear spread (still profitable)
        min_tp = spread * self._spread_cost_buffer
        if tp_distance < min_tp:
            tp_distance = min_tp
            # Also widen SL proportionally to maintain R:R
            sl_distance = tp_distance * (sl_mult / tp_mult) if tp_mult > 0 else tp_distance
            logger.info(
                f"HFT TP widened to clear spread: {signal.symbol} "
                f"TP={tp_distance:.5f} SL={sl_distance:.5f} spread={spread:.5f}"
            )

        # 7b. Margin pre-check (advisory only — don't block or trigger backoff)
        ok, reason = self._client.check_margin(signal.symbol, signal.direction, volume)
        if not ok:
            logger.info(f"HFT margin warning (may reject): {signal.symbol} vol={volume} reason={reason}")
            # Don't block — let the actual order attempt go through

        # 8. Compute SL/TP
        price = tick["ask"] if signal.direction == Direction.LONG else tick["bid"]
        digits = symbol_info.get("digits", 2)

        if signal.direction == Direction.LONG:
            stop_loss = round(price - sl_distance, digits)
            take_profit = round(price + tp_distance, digits)
        else:
            stop_loss = round(price + sl_distance, digits)
            take_profit = round(price - tp_distance, digits)

        # Record trade
        self._trades_today.append(now)
        self._last_trade_time[signal.symbol] = now

        trade_request = TradeRequest(
            symbol=signal.symbol,
            direction=signal.direction,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            algo_name=signal.algo_name,
            signal_strength=signal.strength,
            comment=f"K|{signal.algo_name[:20]}",
        )

        logger.info(
            f"HFT trade approved: {trade_request.symbol} {trade_request.direction.name} "
            f"vol={trade_request.volume} SL={trade_request.stop_loss} TP={trade_request.take_profit}"
        )

        return trade_request

    def record_rejection(self, symbol: str, error_code: int) -> None:
        """Record a margin/money rejection — back off this instrument exponentially."""
        if error_code in (10014, 10015, 10016, 10019, 10031):  # margin/money-related
            count = self._margin_rejects.get(symbol, 0) + 1
            self._margin_rejects[symbol] = count
            # Exponential backoff using configurable base/max
            backoff = min(self._margin_backoff_base * (2 ** (count - 1)), self._margin_backoff_max)
            self._margin_blocked_until[symbol] = datetime.utcnow() + timedelta(seconds=backoff)
            logger.info(
                f"Margin reject #{count} for {symbol} — backing off {backoff}s"
            )

    def clear_rejection(self, symbol: str) -> None:
        """Clear rejection state when a trade fills successfully."""
        self._margin_rejects.pop(symbol, None)
        self._margin_blocked_until.pop(symbol, None)

    def record_pnl(self, pnl: float) -> None:
        """Record realised PnL for daily tracking."""
        self._daily_pnl += pnl

    @property
    def daily_trade_count(self) -> int:
        return len(self._trades_today)

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl
