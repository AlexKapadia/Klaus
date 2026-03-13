"""Thin MetaTrader 5 wrapper — connect, fetch bars, send orders, get positions."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from klaus.config.settings import get_settings
from klaus.core.types import Direction, OrderStatus, TradeResult, Position, AccountInfo

# MT5 timeframe constants mapping
_TF_MAP: dict[str, int] = {}


def _init_tf_map():
    """Initialise timeframe map after MT5 import."""
    import MetaTrader5 as mt5
    global _TF_MAP
    _TF_MAP = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
        "1w": mt5.TIMEFRAME_W1,
    }


class MT5Client:
    """Wrapper around the MetaTrader5 Python package."""

    def __init__(self):
        self._connected = False

    def connect(self) -> bool:
        """Initialise MT5 and log in to the demo account."""
        import MetaTrader5 as mt5

        settings = get_settings().mt5

        init_kwargs = {}
        if settings.path:
            init_kwargs["path"] = settings.path

        if not mt5.initialize(**init_kwargs):
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        if settings.login:
            authorised = mt5.login(
                login=settings.login,
                password=settings.password,
                server=settings.server,
            )
            if not authorised:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

        _init_tf_map()
        self._connected = True
        info = mt5.account_info()
        logger.info(f"MT5 connected: account={info.login}, server={info.server}, balance={info.balance}")
        return True

    def disconnect(self) -> None:
        import MetaTrader5 as mt5
        mt5.shutdown()
        self._connected = False
        logger.info("MT5 disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        count: int = 500,
        from_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars as a DataFrame."""
        import MetaTrader5 as mt5

        tf = _TF_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe '{timeframe}'. Valid: {list(_TF_MAP.keys())}")

        if from_date:
            rates = mt5.copy_rates_from(symbol, tf, from_date, count)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            logger.warning(f"No bars returned for {symbol} {timeframe}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "volume",
        }, inplace=True)

        # Keep only standard OHLCV columns
        cols = [c for c in ["open", "high", "low", "close", "volume", "spread", "real_volume"] if c in df.columns]
        return df[cols]

    def get_tick(self, symbol: str) -> Optional[dict]:
        """Get latest tick (bid/ask) for a symbol."""
        import MetaTrader5 as mt5

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {"bid": tick.bid, "ask": tick.ask, "time": datetime.fromtimestamp(tick.time)}

    def check_margin(self, symbol: str, direction: Direction, volume: float) -> tuple[bool, str]:
        """Check if the account has enough margin for this order.

        Returns (ok, reason) — ok=True means margin is sufficient.
        Non-margin failures (market closed, filling mode, etc.) return (True, "")
        so the order can still be attempted via send_order.
        """
        import MetaTrader5 as mt5

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return True, "no_tick"  # let send_order handle it

        price = tick.ask if direction == Direction.LONG else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if direction == Direction.LONG else mt5.ORDER_TYPE_SELL

        # Detect supported filling mode (must match send_order logic)
        sym_info = mt5.symbol_info(symbol)
        fill_type = mt5.ORDER_FILLING_FOK
        if sym_info is not None:
            fm = sym_info.filling_mode
            if fm & 1:
                fill_type = mt5.ORDER_FILLING_FOK
            elif fm & 2:
                fill_type = mt5.ORDER_FILLING_IOC
            else:
                fill_type = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(volume, 2),
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "margin_check",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": fill_type,
        }

        result = mt5.order_check(request)
        if result is None:
            return True, "check_unavailable"  # let send_order decide

        # Only block on margin/money-specific retcodes
        # 10014=Bad volume, 10015=Bad price, 10019=No money, 10031=No connection
        MARGIN_REJECT_CODES = {10014, 10019}
        if result.retcode in MARGIN_REJECT_CODES:
            logger.debug(
                f"Margin check FAILED: {symbol} {direction.name} vol={volume} "
                f"retcode={result.retcode} comment={result.comment} "
                f"margin_needed={result.margin:.2f} free={result.margin_free:.2f}"
            )
            return False, result.comment

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.debug(
                f"Margin check OK: {symbol} vol={volume} "
                f"margin_needed={result.margin:.2f} free_after={result.margin_free:.2f}"
            )

        # For any other retcode (market closed, filling mode issue, etc.)
        # return True so the order attempt proceeds — send_order will handle it
        return True, ""

    def send_order(
        self,
        symbol: str,
        direction: Direction,
        volume: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        comment: str = "",
    ) -> TradeResult:
        """Send a market order via MT5."""
        import MetaTrader5 as mt5

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return TradeResult(
                ticket=0, symbol=symbol, direction=direction, volume=volume,
                price=0.0, status=OrderStatus.REJECTED,
                error_code=-1, error_message=f"No tick data for {symbol}",
            )

        price = tick.ask if direction == Direction.LONG else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if direction == Direction.LONG else mt5.ORDER_TYPE_SELL

        # Auto-detect supported filling mode for this symbol
        # filling_mode bitmask: bit0=FOK(1), bit1=IOC(2)
        sym_info = mt5.symbol_info(symbol)
        fill_type = mt5.ORDER_FILLING_FOK  # safest default
        if sym_info is not None:
            fm = sym_info.filling_mode
            if fm & 1:      # FOK supported
                fill_type = mt5.ORDER_FILLING_FOK
            elif fm & 2:    # IOC supported
                fill_type = mt5.ORDER_FILLING_IOC
            else:
                fill_type = mt5.ORDER_FILLING_RETURN

        # Clamp volume to broker limits
        if sym_info is not None:
            vol_min = sym_info.volume_min
            vol_max = sym_info.volume_max
            vol_step = sym_info.volume_step
            volume = max(vol_min, min(volume, vol_max))
            if vol_step > 0:
                volume = round(round(volume / vol_step) * vol_step, 6)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": round(volume, 2),
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 123456,
            "comment": comment or "Klaus",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": fill_type,
        }

        result = mt5.order_send(request)

        if result is None:
            err = mt5.last_error()
            logger.warning(f"order_send returned None for {symbol} {direction.name} vol={volume} — last_error={err}")
            return TradeResult(
                ticket=0, symbol=symbol, direction=direction, volume=volume,
                price=0.0, status=OrderStatus.REJECTED,
                error_code=err[0] if err else -1,
                error_message=f"order_send returned None: {err}",
            )

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning(f"Order rejected: {symbol} {direction.name} {volume} — {result.comment}")
            return TradeResult(
                ticket=0, symbol=symbol, direction=direction, volume=volume,
                price=result.price, status=OrderStatus.REJECTED,
                error_code=result.retcode, error_message=result.comment,
            )

        logger.info(f"Order filled: {symbol} {direction.name} {volume} @ {result.price} ticket={result.order}")
        return TradeResult(
            ticket=result.order, symbol=symbol, direction=direction,
            volume=volume, price=result.price, status=OrderStatus.FILLED,
        )

    def modify_position(self, ticket: int, stop_loss: float = 0.0, take_profit: float = 0.0) -> bool:
        """Modify SL/TP on an open position. Returns True on success."""
        import MetaTrader5 as mt5

        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning(f"modify_position: ticket {ticket} not found")
            return False

        pos = position[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": stop_loss if stop_loss else pos.sl,
            "tp": take_profit if take_profit else pos.tp,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.comment if result else mt5.last_error()
            logger.debug(f"modify_position failed ticket={ticket}: {err}")
            return False

        logger.info(f"Position modified: ticket={ticket} SL={request['sl']} TP={request['tp']}")
        return True

    def close_position(self, ticket: int) -> TradeResult:
        """Close an open position by ticket."""
        import MetaTrader5 as mt5

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return TradeResult(
                ticket=ticket, symbol="", direction=Direction.FLAT, volume=0,
                price=0, status=OrderStatus.REJECTED,
                error_message="Position not found",
            )

        pos = position[0]
        symbol = pos.symbol
        direction = Direction.SHORT if pos.type == mt5.ORDER_TYPE_BUY else Direction.LONG
        volume = pos.volume

        return self.send_order(symbol, direction, volume, comment=f"Close #{ticket}")

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        import MetaTrader5 as mt5

        positions = mt5.positions_get()
        if not positions:
            return []

        result = []
        for pos in positions:
            direction = Direction.LONG if pos.type == mt5.ORDER_TYPE_BUY else Direction.SHORT
            result.append(Position(
                ticket=pos.ticket,
                symbol=pos.symbol,
                direction=direction,
                volume=pos.volume,
                open_price=pos.price_open,
                open_time=datetime.fromtimestamp(pos.time),
                stop_loss=pos.sl,
                take_profit=pos.tp,
                algo_name=pos.comment or "",
                unrealized_pnl=pos.profit,
            ))
        return result

    def get_account_info(self) -> AccountInfo:
        """Get current account snapshot."""
        import MetaTrader5 as mt5

        info = mt5.account_info()
        positions = mt5.positions_total() or 0

        return AccountInfo(
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            margin_level=info.margin_level if info.margin_level else 0.0,
            open_positions=positions,
        )

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get symbol specification (point, digits, trade sizes, etc.)."""
        import MetaTrader5 as mt5

        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "symbol": info.name,
            "point": info.point,
            "digits": info.digits,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_contract_size": info.trade_contract_size,
        }
