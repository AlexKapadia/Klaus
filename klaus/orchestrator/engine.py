"""Main orchestrator loop — the heart of Klaus."""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from klaus.config.settings import get_settings
from klaus.core.registry import get_algorithm, get_all_algorithms
from klaus.core.types import Direction, Regime, Signal, TradeRequest
from klaus.data.feature_store import FeatureStore
from klaus.data.market_data import MarketData
from klaus.data.mt5_client import MT5Client
from klaus.execution.executor import Executor
from klaus.orchestrator.rules_engine import RulesEngine
from klaus.orchestrator.scheduler import InstrumentScheduler
from klaus.regime.hmm_detector import HMMRegimeDetector
from klaus.risk.risk_manager import RiskManager


class Engine:
    """Orchestrates the full trading pipeline.

    Each cycle (default 60s):
      1. Refresh market data for all active instruments
      2. Compute features
      3. Check/update regime detection (every H4)
      4. Query rules engine for applicable algorithms
      5. Generate signals
      6. Pass signals through risk manager
      7. Execute approved trades
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = MT5Client()
        self._market_data = MarketData(self._client)
        self._rules = RulesEngine()
        self._scheduler = InstrumentScheduler()
        self._risk_manager = RiskManager(self._client)
        self._executor = Executor(self._client)

        # One HMM detector per instrument
        self._hmm_detectors: dict[str, HMMRegimeDetector] = {}
        # Current regime per instrument
        self._current_regimes: dict[str, Regime] = {}
        # Instantiated algorithm objects
        self._algo_instances: dict[str, object] = {}

        self._running = False

    def start(self) -> None:
        """Connect to MT5 and begin the orchestrator loop."""
        logger.info("Klaus Engine starting...")

        if not self._client.connect():
            logger.error("Failed to connect to MT5. Aborting.")
            return

        self._init_algorithms()
        self._running = True

        instruments = self._settings.instrument_list
        logger.info(f"Active instruments: {instruments}")

        try:
            while self._running:
                cycle_start = time.time()
                self._run_cycle(instruments)
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self._scheduler.cycle_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Shutdown requested via KeyboardInterrupt")
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully shut down."""
        self._running = False
        self._client.disconnect()
        logger.info("Klaus Engine stopped")

    def _init_algorithms(self) -> None:
        """Instantiate all registered algorithms with their config params."""
        algo_configs = self._settings.algorithms.get("algorithms", {})
        for name, cls in get_all_algorithms().items():
            params = algo_configs.get(name, {})
            self._algo_instances[name] = cls(params=params)
            logger.debug(f"Initialised algorithm: {name}")

    def _run_cycle(self, instruments: list[str]) -> None:
        """Execute one full orchestrator cycle."""
        all_signals: list[Signal] = []

        for symbol in instruments:
            try:
                signals = self._process_instrument(symbol)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Pass all signals through risk manager
        for signal in all_signals:
            try:
                trade_request = self._risk_manager.evaluate(signal)
                if trade_request is not None:
                    result = self._executor.execute(trade_request)
                    logger.info(
                        f"Trade executed: {result.symbol} {result.direction.name} "
                        f"{result.volume} @ {result.price} [{result.status.name}]"
                    )
            except Exception as e:
                logger.error(f"Error executing signal {signal}: {e}")

    def _process_instrument(self, symbol: str) -> list[Signal]:
        """Process a single instrument: data → regime → algorithms → signals."""
        # 1. Fetch data
        df = self._market_data.get_bars(symbol, "1h", 500)
        if df.empty or len(df) < 50:
            return []

        # 2. Compute features
        df = FeatureStore.add_all_features(df)

        # 3. Regime detection
        regime = self._update_regime(symbol, df)
        self._current_regimes[symbol] = regime

        # 4. Get applicable algorithms from rules engine
        algo_names = self._rules.get_algorithms(symbol, regime)
        if not algo_names:
            return []

        # 5. Generate signals
        signals = []
        for algo_name in algo_names:
            algo = self._algo_instances.get(algo_name)
            if algo is None:
                logger.warning(f"Algorithm '{algo_name}' not instantiated, skipping")
                continue
            if not algo.can_trade(symbol):
                continue
            if len(df) < algo.min_bars_required:
                continue

            signal = algo.generate_signal(df, symbol)
            if signal is not None:
                logger.info(
                    f"Signal: {symbol} {signal.direction.name} "
                    f"strength={signal.strength:.3f} algo={signal.algo_name} "
                    f"regime={regime.name}"
                )
                signals.append(signal)

        return signals

    def _update_regime(self, symbol: str, df) -> Regime:
        """Check and update regime detection for an instrument."""
        # Initialise detector if needed
        if symbol not in self._hmm_detectors:
            self._hmm_detectors[symbol] = HMMRegimeDetector()

        detector = self._hmm_detectors[symbol]

        # Refit if needed
        if self._scheduler.should_refit_hmm(symbol) or not detector.is_fitted:
            # Use H4 data for regime fitting
            h4_data = self._market_data.get_bars(symbol, "4h", 500)
            if not h4_data.empty and len(h4_data) >= 50:
                h4_data = FeatureStore.add_returns(h4_data)
                h4_data = FeatureStore.add_rolling_volatility(h4_data)
                detector.fit(h4_data)
                self._scheduler.mark_hmm_refit(symbol)

        # Predict regime if it's time
        if self._scheduler.should_check_regime(symbol) and detector.is_fitted:
            regime = detector.predict(df)
            self._scheduler.mark_regime_checked(symbol)
            return regime

        return self._current_regimes.get(symbol, Regime.UNKNOWN)

    def run_once(self) -> None:
        """Run a single cycle (useful for testing)."""
        if not self._client.is_connected:
            if not self._client.connect():
                logger.error("Cannot connect to MT5")
                return
        self._init_algorithms()
        instruments = self._settings.instrument_list
        self._run_cycle(instruments)
