"""MT5 order execution wrapper."""

from __future__ import annotations

from loguru import logger

from klaus.core.types import TradeRequest, TradeResult, OrderStatus
from klaus.data.mt5_client import MT5Client


class Executor:
    """Translates TradeRequests into MT5 orders."""

    def __init__(self, client: MT5Client):
        self._client = client

    def execute(self, request: TradeRequest) -> TradeResult:
        """Send a trade request to MT5."""
        logger.info(
            f"Executing: {request.symbol} {request.direction.name} "
            f"vol={request.volume} SL={request.stop_loss} TP={request.take_profit} "
            f"[{request.algo_name}]"
        )

        result = self._client.send_order(
            symbol=request.symbol,
            direction=request.direction,
            volume=request.volume,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            comment=request.comment,
        )

        if result.status == OrderStatus.FILLED:
            logger.info(f"Filled: ticket={result.ticket} @ {result.price}")
        else:
            logger.warning(
                f"Order not filled: {result.status.name} "
                f"err={result.error_code} {result.error_message}"
            )

        return result
