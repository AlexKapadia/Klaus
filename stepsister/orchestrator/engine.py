"""Main orchestrator loop — the heart of Step Sister.

Same architecture as Klaus's standard engine but configured for the
forex market. Reads from stepsister's own config and registry,
uses Klaus's shared infrastructure (MT5Client, MarketData, FeatureStore,
Executor, RiskManager, HMMRegimeDetector, InstrumentScheduler).
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from stepsister.config.settings import get_fx_settings
from stepsister.core.registry import get_all_fx_algorithms
from klaus.core.types import Direction, Regime, Signal, TradeRequest
from klaus.data.feature_store import FeatureStore
from klaus.data.market_data import MarketData
from klaus.data.mt5_client import MT5Client
from klaus.execution.executor import Executor
from klaus.orchestrator.scheduler import InstrumentScheduler
from klaus.regime.hmm_detector import HMMRegimeDetector
from klaus.risk.risk_manager import RiskManager


class _FXRulesEngine:
    """Simple rules engine that maps (instrument, regime) -> algorithm names.

    Reads from fx_settings.regimes.get("mapping", {}).
    """

    def __init__(self):
        self._mapping: dict[str, dict[str, list[str]]] = {}
        self._load_mapping()

    def _load_mapping(self) -> None:
        settings = get_fx_settings()
        raw = settings.regimes.get("mapping", {})

        for symbol, regime_map in raw.items():
            self._mapping[symbol] = {}
            for regime_str, algos in regime_map.items():
                self._mapping[symbol][regime_str] = algos or []

        logger.info(f"FX RulesEngine loaded mappings for {len(self._mapping)} instruments")

    def get_algorithms(self, symbol: str, regime: Regime) -> list[str]:
        """Return algorithm names to run for the given instrument and regime."""
        regime_str = regime.name

        if symbol not in self._mapping:
            logger.debug(f"No FX mapping for {symbol}, skipping")
            return []

        algos = self._mapping[symbol].get(regime_str, [])

        if not algos:
            logger.debug(f"No FX algorithms for {symbol} in {regime_str} regime")

        return algos

    def get_all_symbols(self) -> list[str]:
        """Return all symbols that have regime mappings."""
        return list(self._mapping.keys())

    def reload(self) -> None:
        """Reload mapping from config."""
        self._mapping.clear()
        self._load_mapping()


class FXEngine:
    """Orchestrates the full forex trading pipeline.

    Each cycle (default 60s):
      1. Log account state (balance, equity, free margin, open positions)
      2. Refresh market data for all active FX instruments
      3. Compute features
      4. Check/update regime detection (every H4)
      5. Query rules engine for applicable algorithms
      6. Generate signals
      7. Pass signals through risk manager (incl. margin pre-check)
      8. Execute approved trades
      9. Track closed trades → update Kelly criterion
    """

    def __init__(self):
        self._settings = get_fx_settings()
        self._client = MT5Client()
        self._market_data = MarketData(self._client)
        self._rules = _FXRulesEngine()
        self._scheduler = InstrumentScheduler()
        self._risk_manager = RiskManager(self._client)
        self._executor = Executor(self._client)

        # One HMM detector per instrument
        self._hmm_detectors: dict[str, HMMRegimeDetector] = {}
        # Current regime per instrument
        self._current_regimes: dict[str, Regime] = {}
        # Instantiated algorithm objects
        self._algo_instances: dict[str, object] = {}
        # Track open tickets to detect closed trades
        self._known_tickets: set[int] = set()

        self._running = False

    def start(self) -> None:
        """Connect to MT5 and begin the orchestrator loop."""
        logger.info("Step Sister Engine starting...")

        if not self._client.connect():
            logger.error("Step Sister: Failed to connect to MT5. Aborting.")
            return

        self._init_algorithms()
        self._running = True

        instruments = self._settings.instrument_list
        logger.info(f"Step Sister active instruments: {instruments}")

        # Train ML models on startup with available data
        self._warmup_ml_models(instruments)

        # Snapshot existing positions
        for p in self._client.get_positions():
            self._known_tickets.add(p.ticket)

        try:
            while self._running:
                cycle_start = time.time()
                self._run_cycle(instruments)
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self._scheduler.cycle_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Step Sister: Shutdown requested via KeyboardInterrupt")
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully shut down."""
        self._running = False
        self._client.disconnect()
        logger.info("Step Sister Engine stopped")

    def _init_algorithms(self) -> None:
        """Instantiate all registered FX algorithms with their config params."""
        algo_configs = self._settings.algorithms.get("algorithms", {})
        for name, cls in get_all_fx_algorithms().items():
            params = algo_configs.get(name, {})
            self._algo_instances[name] = cls(params=params)
            logger.debug(f"Step Sister: Initialised algorithm: {name}")

    def _warmup_ml_models(self, instruments: list[str]) -> None:
        """Pre-train ML models on historical data so they're ready from cycle 1."""
        ml_algos = {
            name: algo for name, algo in self._algo_instances.items()
            if hasattr(algo, "fit") and name in (
                "fx_xgboost_signal", "fx_lstm_signal",
            )
        }
        if not ml_algos:
            return

        logger.info(f"Step Sister: warming up {len(ml_algos)} ML models...")
        # Use EURUSD as the primary training pair (most liquid)
        training_symbol = "EURUSD" if "EURUSD" in instruments else instruments[0]
        df = self._market_data.get_bars(training_symbol, "1h", 500)
        if df.empty or len(df) < 200:
            logger.warning("Step Sister: insufficient data for ML warmup")
            return

        from klaus.data.feature_store import FeatureStore
        df = FeatureStore.add_all_features(df)

        for name, algo in ml_algos.items():
            try:
                algo.fit(df)
                logger.info(f"Step Sister: ML model '{name}' trained on {len(df)} bars")
            except Exception as e:
                logger.warning(f"Step Sister: ML warmup failed for '{name}': {e}")

    def _check_closed_trades(self) -> None:
        """Detect trades that closed (hit TP/SL) and record P&L for Kelly updates."""
        current_positions = self._client.get_positions()
        current_tickets = {p.ticket for p in current_positions}

        closed_tickets = self._known_tickets - current_tickets
        if not closed_tickets:
            self._known_tickets = current_tickets
            return

        # Query MT5 deal history for closed tickets
        import MetaTrader5 as mt5
        from datetime import datetime, timedelta

        now = datetime.utcnow()
        yesterday = now - timedelta(days=1)

        for ticket in closed_tickets:
            try:
                deals = mt5.history_deals_get(yesterday, now, position=ticket)
                if deals and len(deals) >= 2:
                    # Sum P&L across all deals for this position
                    pnl = sum(d.profit + d.commission + d.swap for d in deals)
                    self._risk_manager.record_trade_result(pnl)
                    logger.info(f"Closed trade ticket={ticket} P&L={pnl:+.2f}")
            except Exception as e:
                logger.debug(f"Could not fetch history for ticket {ticket}: {e}")

        self._known_tickets = current_tickets

    def _run_cycle(self, instruments: list[str]) -> None:
        """Execute one full orchestrator cycle."""
        # Log account state at start of each cycle
        try:
            account = self._client.get_account_info()
            logger.info(
                f"Account: balance=${account.balance:.2f} equity=${account.equity:.2f} "
                f"free_margin=${account.free_margin:.2f} margin_level="
                f"{account.margin_level:.1f}% positions={account.open_positions}"
            )
        except Exception:
            pass

        # Check for closed trades and update Kelly
        self._check_closed_trades()

        all_signals: list[Signal] = []

        for symbol in instruments:
            try:
                signals = self._process_instrument(symbol)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Step Sister error processing {symbol}: {e}")

        # Pass all signals through risk manager
        for signal in all_signals:
            try:
                trade_request = self._risk_manager.evaluate(signal)
                if trade_request is not None:
                    result = self._executor.execute(trade_request)
                    logger.info(
                        f"Step Sister trade executed: {result.symbol} {result.direction.name} "
                        f"{result.volume} @ {result.price} [{result.status.name}]"
                    )
                    # Track new fills
                    if result.status.name == "FILLED" and result.ticket:
                        self._known_tickets.add(result.ticket)
            except Exception as e:
                logger.error(f"Step Sister execution error {signal}: {e}")

    def _process_instrument(self, symbol: str) -> list[Signal]:
        """Process a single instrument: data -> regime -> algorithms -> signals."""
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
                logger.warning(f"Step Sister: Algorithm '{algo_name}' not instantiated, skipping")
                continue
            if not algo.can_trade(symbol):
                continue
            if len(df) < algo.min_bars_required:
                continue

            signal = algo.generate_signal(df, symbol)
            if signal is not None:
                logger.info(
                    f"Step Sister Signal: {symbol} {signal.direction.name} "
                    f"strength={signal.strength:.3f} algo={signal.algo_name} "
                    f"regime={regime.name}"
                )
                signals.append(signal)

        return signals

    def _update_regime(self, symbol: str, df) -> Regime:
        """Check and update regime detection for an instrument."""
        if symbol not in self._hmm_detectors:
            self._hmm_detectors[symbol] = HMMRegimeDetector()

        detector = self._hmm_detectors[symbol]

        # Refit if needed
        if self._scheduler.should_refit_hmm(symbol) or not detector.is_fitted:
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
                logger.error("Step Sister: Cannot connect to MT5")
                return
        self._init_algorithms()
        instruments = self._settings.instrument_list
        self._run_cycle(instruments)
