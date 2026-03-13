"""High-Frequency Trading engine — fast-cycle orchestrator for M1 data.

Runs alongside the standard Engine on 5-10 second cycles using M1 bars.
Manages HFT-specific algorithms, tighter risk controls, and cross-asset
data sharing for lead-lag and spread algorithms.

Architecture:
  - 5-second main cycle (configurable)
  - M1 bars for signal generation (500 bars = ~8 hours)
  - M5 regime detection (every 15 minutes)
  - Cross-asset data broadcast for lead-lag and spread algos
  - HFT risk manager with daily limits and cooldowns
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from klaus.config.settings import get_settings
from klaus.core.registry import get_all_algorithms
from klaus.core.types import Direction, Regime, Signal, TradeRequest
from klaus.data.hft_feature_store import HFTFeatureStore
from klaus.data.market_data import MarketData
from klaus.data.mt5_client import MT5Client
from klaus.execution.executor import Executor
from klaus.orchestrator.rules_engine import RulesEngine
from klaus.regime.hmm_detector import HMMRegimeDetector
from klaus.risk.hft_risk_manager import HFTRiskManager
from klaus.risk.trailing_stop import TrailingStopManager


# HFT algorithm names (for filtering the registry)
_HFT_ALGO_NAMES = {
    "order_flow_imbalance",
    "tick_scalper",
    "vol_breakout_hft",
    "micro_reversion",
    "spread_scalper_hft",
    "cross_commodity_lead_lag",
    "dqn_scalper",
    "xgboost_hft",
    "hft_momentum",
}

# Known pair relationships for spread and lead-lag algos
_SPREAD_PAIRS = {
    "XTIUSD": "XBRUSD",
    "XBRUSD": "XTIUSD",
    "XAUUSD": "XAGUSD",
    "XAGUSD": "XAUUSD",
}

# Leader data needed by cross_commodity_lead_lag
_LEADER_MAP = {
    "XAGUSD": "XAUUSD",
    "XBRUSD": "XTIUSD",
    "XTIUSD": "XAUUSD",
}


class HFTEngine:
    """High-frequency orchestrator running on 5-second cycles with M1 data.

    Operates in parallel with the standard Engine. Each cycle:
      1. Fetch M1 data for all HFT instruments
      2. Compute HFT features (microstructure + standard)
      3. Update fast regime detection (every 15 min on M5 data)
      4. Broadcast cross-asset data to spread/lead-lag algos
      5. Generate signals from HFT algorithms
      6. Pass through HFT risk manager
      7. Execute approved trades
    """

    def __init__(self, hft_config: dict = None):
        self._settings = get_settings()
        self._client = MT5Client()
        self._market_data = MarketData(self._client)
        self._rules = RulesEngine()
        self._executor = Executor(self._client)

        cfg = hft_config or {}
        self._cycle_interval = cfg.get("cycle_interval", 5)
        self._data_bars = cfg.get("data_bars", 500)
        self._regime_interval_min = cfg.get("regime_interval_minutes", 15)
        self._refit_interval_hours = cfg.get("refit_interval_hours", 4)

        self._risk_manager = HFTRiskManager(self._client, cfg.get("risk", {}))
        trailing_cfg = cfg.get("risk", {}).get("trailing_stop", {})
        self._trailing_stop = TrailingStopManager(self._client, trailing_cfg)

        # Per-instrument HMM detectors (separate from standard engine)
        self._hmm_detectors: dict[str, HMMRegimeDetector] = {}
        self._current_regimes: dict[str, Regime] = {}
        self._last_regime_check: dict[str, datetime] = {}
        self._last_refit: dict[str, datetime] = {}

        # HFT algorithm instances
        self._algo_instances: dict[str, object] = {}

        # Cached M1 data for cross-asset sharing
        self._m1_cache: dict[str, object] = {}

        self._running = False

    @property
    def hft_instruments(self) -> list[str]:
        """Active instruments that have HFT algorithms mapped."""
        all_active = self._settings.instrument_list
        hft_mapped = set()
        mapping = self._settings.regimes.get("hft_mapping", {})
        for symbol in mapping:
            if symbol in all_active or symbol in [
                i["symbol"] for i in self._settings.instruments.get("instruments", [])
                if i.get("active", True)
            ]:
                hft_mapped.add(symbol)

        # Fallback: use all active instruments that support HFT algos
        if not hft_mapped:
            hft_mapped = {s for s in all_active if s in {
                "XAUUSD", "XTIUSD", "XBRUSD", "XNGUSD", "XAGUSD",
            }}

        return list(hft_mapped)

    def start(self) -> None:
        """Connect to MT5 and begin the HFT loop."""
        logger.info("=" * 60)
        logger.info("  KLAUS HFT Engine starting...")
        logger.info(f"  Cycle interval: {self._cycle_interval}s")
        logger.info("=" * 60)

        if not self._client.connect():
            logger.error("HFT Engine: Failed to connect to MT5. Aborting.")
            return

        self._init_hft_algorithms()
        self._running = True

        instruments = self.hft_instruments
        logger.info(f"HFT instruments: {instruments}")
        logger.info(f"HFT algorithms loaded: {list(self._algo_instances.keys())}")

        try:
            while self._running:
                cycle_start = time.time()
                self._run_hft_cycle(instruments)
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self._cycle_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("HFT Engine: Shutdown requested")
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully shut down."""
        self._running = False
        self._client.disconnect()
        logger.info(f"HFT Engine stopped. Daily trades: {self._risk_manager.daily_trade_count}")

    def _init_hft_algorithms(self) -> None:
        """Instantiate only HFT algorithms from the registry."""
        algo_configs = self._settings.algorithms.get("algorithms", {})
        for name, cls in get_all_algorithms().items():
            if name in _HFT_ALGO_NAMES:
                params = algo_configs.get(name, {})
                self._algo_instances[name] = cls(params=params)
                logger.debug(f"HFT: Initialised {name}")

    def _run_hft_cycle(self, instruments: list[str]) -> None:
        """Execute one HFT cycle."""
        # Phase 0: Manage trailing stops on existing positions
        try:
            self._trailing_stop.manage()
        except Exception as e:
            logger.error(f"Trailing stop error: {e}")

        # Phase 1: Fetch all M1 data (batch to reduce latency)
        self._m1_cache.clear()
        for symbol in instruments:
            try:
                df = self._market_data.get_bars(symbol, "1m", self._data_bars, use_cache=False)
                if not df.empty and len(df) >= 30:
                    df = HFTFeatureStore.add_all_hft_features(df)
                    self._m1_cache[symbol] = df
            except Exception as e:
                logger.error(f"HFT data error {symbol}: {e}")

        if not self._m1_cache:
            logger.info("HFT cycle: no data available")
            return

        logger.info(f"─── HFT cycle ─── {len(self._m1_cache)} instruments loaded ───")

        # Phase 2: Broadcast cross-asset data to pair/lead-lag algos
        self._broadcast_cross_asset_data()

        # Phase 3: Process each instrument
        all_signals: list[Signal] = []
        for symbol in instruments:
            if symbol not in self._m1_cache:
                logger.info(f"  {symbol}: no M1 data")
                continue
            try:
                signals = self._process_hft_instrument(symbol)
                all_signals.extend(signals)
                if not signals:
                    regime = self._current_regimes.get(symbol, Regime.UNKNOWN)
                    logger.info(f"  {symbol} [{regime.name}]: no signals this cycle")
            except Exception as e:
                logger.error(f"HFT processing error {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Phase 4: Risk-check and execute
        for signal in all_signals:
            try:
                trade_request = self._risk_manager.evaluate(signal)
                if trade_request is not None:
                    result = self._executor.execute(trade_request)
                    logger.info(
                        f"HFT executed: {result.symbol} {result.direction.name} "
                        f"{result.volume} @ {result.price} [{result.status.name}] "
                        f"algo={signal.algo_name}"
                    )
                    # Feed rejection/fill info back to risk manager
                    from klaus.core.types import OrderStatus
                    if result.status == OrderStatus.REJECTED and result.error_code in (
                        10019, 10014, 10015, 10016, 10019, 10031,  # margin/money-related MT5 codes
                    ):
                        self._risk_manager.record_rejection(result.symbol, result.error_code)
                    elif result.status == OrderStatus.FILLED:
                        self._risk_manager.clear_rejection(result.symbol)
            except Exception as e:
                logger.error(f"HFT execution error: {e}")

    def _process_hft_instrument(self, symbol: str) -> list[Signal]:
        """Process a single instrument through HFT algorithms."""
        df = self._m1_cache[symbol]

        # Fast regime detection on M5 data
        regime = self._update_hft_regime(symbol)
        self._current_regimes[symbol] = regime

        # Get HFT algorithms for this regime
        algo_names = self._get_hft_algorithms(symbol, regime)
        if not algo_names:
            return []

        signals = []
        for algo_name in algo_names:
            algo = self._algo_instances.get(algo_name)
            if algo is None:
                continue
            if not algo.can_trade(symbol):
                continue
            if len(df) < algo.min_bars_required:
                logger.info(f"    {algo_name}: need {algo.min_bars_required} bars, have {len(df)}")
                continue

            try:
                signal = algo.generate_signal(df, symbol)
            except Exception as e:
                logger.error(f"    {algo_name} ERROR: {e}")
                continue

            if signal is not None:
                logger.info(
                    f"  >>> SIGNAL: {symbol} {signal.direction.name} "
                    f"strength={signal.strength:.3f} algo={signal.algo_name}"
                )
                signals.append(signal)

        return signals

    def _broadcast_cross_asset_data(self) -> None:
        """Share M1 data between cross-asset algorithms."""
        # Spread scalper: pair data
        for algo_name in ("spread_scalper_hft",):
            algo = self._algo_instances.get(algo_name)
            if algo is None:
                continue
            for symbol, pair_symbol in _SPREAD_PAIRS.items():
                if pair_symbol in self._m1_cache:
                    algo.set_pair_data(self._m1_cache[pair_symbol])

        # Cross-commodity lead-lag: leader data
        lead_lag_algo = self._algo_instances.get("cross_commodity_lead_lag")
        if lead_lag_algo is not None:
            for lagger, leader in _LEADER_MAP.items():
                if leader in self._m1_cache:
                    lead_lag_algo.set_leader_data(leader, self._m1_cache[leader])

    def _get_hft_algorithms(self, symbol: str, regime: Regime) -> list[str]:
        """Get HFT algorithms for a given instrument and regime.

        First checks hft_mapping in regimes.yaml, then falls back
        to matching all HFT algos that support this instrument.
        """
        hft_mapping = self._settings.regimes.get("hft_mapping", {})
        regime_str = regime.name

        if symbol in hft_mapping and regime_str in hft_mapping[symbol]:
            return hft_mapping[symbol][regime_str]

        # Fallback: all HFT algos that support this instrument and regime
        result = []
        for name, algo in self._algo_instances.items():
            if algo.can_trade(symbol):
                if not algo.preferred_regimes or regime in algo.preferred_regimes:
                    result.append(name)
        return result

    def _update_hft_regime(self, symbol: str) -> Regime:
        """Fast regime detection on M5 data with 15-minute updates."""
        now = datetime.utcnow()

        if symbol not in self._hmm_detectors:
            self._hmm_detectors[symbol] = HMMRegimeDetector()

        detector = self._hmm_detectors[symbol]

        # Refit every N hours
        last_refit = self._last_refit.get(symbol)
        should_refit = last_refit is None or (now - last_refit) > timedelta(hours=self._refit_interval_hours)

        if should_refit or not detector.is_fitted:
            m5_data = self._market_data.get_bars(symbol, "5m", 500)
            if not m5_data.empty and len(m5_data) >= 50:
                from klaus.data.feature_store import FeatureStore
                m5_data = FeatureStore.add_returns(m5_data)
                m5_data = FeatureStore.add_rolling_volatility(m5_data)
                detector.fit(m5_data)
                self._last_refit[symbol] = now

        # Check regime every 15 minutes
        last_check = self._last_regime_check.get(symbol)
        should_check = last_check is None or (now - last_check) > timedelta(minutes=self._regime_interval_min)

        if should_check and detector.is_fitted:
            m1_df = self._m1_cache.get(symbol)
            if m1_df is not None:
                regime = detector.predict(m1_df)
                self._last_regime_check[symbol] = now
                return regime

        return self._current_regimes.get(symbol, Regime.UNKNOWN)

    def run_once(self) -> None:
        """Run a single HFT cycle (for testing)."""
        if not self._client.is_connected:
            if not self._client.connect():
                logger.error("HFT Engine: Cannot connect to MT5")
                return
        self._init_hft_algorithms()
        instruments = self.hft_instruments
        self._run_hft_cycle(instruments)
