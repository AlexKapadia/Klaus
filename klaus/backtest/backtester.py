"""Walk-forward backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from klaus.algorithms.base import BaseAlgorithm
from klaus.core.types import Direction, Signal
from klaus.data.feature_store import FeatureStore


@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: Direction
    entry_price: float
    exit_price: float = 0.0
    volume: float = 1.0
    pnl: float = 0.0
    algo_name: str = ""
    closed: bool = False


@dataclass
class BacktestResult:
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    profit_factor: float = 0.0


class Backtester:
    """Walk-forward backtester using the same BaseAlgorithm interface.

    - Splits data into train + test windows
    - Runs algorithm.generate_signal() on each test bar
    - Simulates fills at next bar's open
    - Tracks equity curve, drawdown, Sharpe
    """

    def __init__(
        self,
        initial_equity: float = 100_000.0,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 3.0,
        position_size_pct: float = 0.5,
    ):
        self.initial_equity = initial_equity
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.position_size_pct = position_size_pct

    def run(
        self,
        algorithm: BaseAlgorithm,
        data: pd.DataFrame,
        symbol: str,
        train_window: int = 200,
        test_window: int = 50,
    ) -> BacktestResult:
        """Walk-forward backtest.

        Args:
            algorithm: Algorithm instance to test.
            data: Full OHLCV DataFrame.
            symbol: Instrument symbol.
            train_window: Bars for initial training.
            test_window: Bars per walk-forward step.
        """
        data = FeatureStore.add_all_features(data)

        if len(data) < train_window + test_window:
            logger.warning(f"Insufficient data for backtest: {len(data)} bars")
            return BacktestResult()

        equity = self.initial_equity
        equity_curve = [equity]
        trades: list[BacktestTrade] = []
        open_trade: Optional[BacktestTrade] = None

        start = train_window
        while start + test_window <= len(data):
            train_data = data.iloc[:start]
            test_data = data.iloc[start : start + test_window]

            # Fit algorithm on training data
            algorithm.fit(train_data)
            algorithm.warm_up(train_data)

            # Walk through test bars
            for i in range(len(test_data)):
                bar_idx = start + i
                lookback = data.iloc[:bar_idx + 1]
                current_bar = data.iloc[bar_idx]
                atr = current_bar.get("atr", 0)

                # Check open trade for SL/TP
                if open_trade is not None:
                    hit_sl, hit_tp = self._check_sl_tp(open_trade, current_bar)
                    if hit_sl or hit_tp:
                        exit_price = open_trade.exit_price if hit_sl else open_trade.exit_price
                        if hit_tp:
                            exit_price = open_trade.exit_price  # set below
                        # Determine exit price
                        if open_trade.direction == Direction.LONG:
                            if hit_sl:
                                exit_price = open_trade.entry_price - self.sl_atr_mult * atr if atr > 0 else current_bar["low"]
                            else:
                                exit_price = open_trade.entry_price + self.tp_atr_mult * atr if atr > 0 else current_bar["high"]
                            open_trade.pnl = (exit_price - open_trade.entry_price) * open_trade.volume
                        else:
                            if hit_sl:
                                exit_price = open_trade.entry_price + self.sl_atr_mult * atr if atr > 0 else current_bar["high"]
                            else:
                                exit_price = open_trade.entry_price - self.tp_atr_mult * atr if atr > 0 else current_bar["low"]
                            open_trade.pnl = (open_trade.entry_price - exit_price) * open_trade.volume

                        open_trade.exit_price = exit_price
                        open_trade.exit_time = current_bar.name
                        open_trade.closed = True
                        equity += open_trade.pnl
                        trades.append(open_trade)
                        open_trade = None

                equity_curve.append(equity)

                # Generate signal if no open trade
                if open_trade is not None:
                    continue

                signal = algorithm.generate_signal(lookback, symbol)
                if signal is None or signal.direction == Direction.FLAT:
                    continue

                # Open trade at next bar's open (if available)
                if bar_idx + 1 >= len(data):
                    continue

                next_bar = data.iloc[bar_idx + 1]
                entry_price = next_bar["open"]

                # Position sizing
                risk_amount = equity * (self.position_size_pct / 100)
                if atr > 0:
                    sl_dist = atr * self.sl_atr_mult
                    volume = risk_amount / sl_dist
                else:
                    volume = risk_amount / (entry_price * 0.02)

                open_trade = BacktestTrade(
                    entry_time=next_bar.name,
                    exit_time=None,
                    symbol=symbol,
                    direction=signal.direction,
                    entry_price=entry_price,
                    volume=volume,
                    algo_name=signal.algo_name,
                )

            start += test_window

        # Close any remaining open trade at last bar's close
        if open_trade is not None:
            last_bar = data.iloc[-1]
            open_trade.exit_price = last_bar["close"]
            open_trade.exit_time = last_bar.name
            if open_trade.direction == Direction.LONG:
                open_trade.pnl = (open_trade.exit_price - open_trade.entry_price) * open_trade.volume
            else:
                open_trade.pnl = (open_trade.entry_price - open_trade.exit_price) * open_trade.volume
            open_trade.closed = True
            equity += open_trade.pnl
            trades.append(open_trade)
            equity_curve.append(equity)

        # Compute metrics
        result = self._compute_metrics(trades, equity_curve)
        return result

    def _check_sl_tp(self, trade: BacktestTrade, bar) -> tuple[bool, bool]:
        """Check if a bar hits stop loss or take profit."""
        atr = bar.get("atr", 0)
        if atr <= 0:
            return False, False

        if trade.direction == Direction.LONG:
            sl_price = trade.entry_price - self.sl_atr_mult * atr
            tp_price = trade.entry_price + self.tp_atr_mult * atr
            hit_sl = bar["low"] <= sl_price
            hit_tp = bar["high"] >= tp_price
        else:
            sl_price = trade.entry_price + self.sl_atr_mult * atr
            tp_price = trade.entry_price - self.tp_atr_mult * atr
            hit_sl = bar["high"] >= sl_price
            hit_tp = bar["low"] <= tp_price

        return hit_sl, hit_tp

    def _compute_metrics(self, trades: list[BacktestTrade], equity_curve: list[float]) -> BacktestResult:
        result = BacktestResult()
        result.trades = trades
        result.equity_curve = equity_curve
        result.n_trades = len(trades)

        if not trades:
            return result

        pnls = [t.pnl for t in trades]
        result.total_pnl = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        result.win_rate = len(wins) / len(pnls) if pnls else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe ratio (annualised, assuming hourly bars)
        returns = pd.Series(pnls)
        if returns.std() > 0:
            result.sharpe_ratio = float((returns.mean() / returns.std()) * np.sqrt(252 * 24))

        # Max drawdown
        eq = pd.Series(equity_curve)
        peak = eq.cummax()
        dd = (eq - peak) / peak * 100
        result.max_drawdown_pct = float(abs(dd.min()))

        return result
