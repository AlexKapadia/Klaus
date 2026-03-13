"""High-Frequency Trading engine for forex — fast-cycle orchestrator for M1 data.

The PRIMARY Step Sister engine, designed for maximum trades/minute.
Runs alongside the standard FXEngine on 5-second cycles using M1 bars.
Manages HFT-specific algorithms, tighter risk controls, and cross-asset
data sharing for spread and lead-lag algorithms.

Architecture:
  - 5-second main cycle (configurable)
  - M1 bars for signal generation (500 bars = ~8 hours)
  - M5 regime detection (every 15 minutes)
  - Cross-asset data broadcast for spread/lead-lag algos
  - HFT risk manager with daily limits and cooldowns
  - "S|" comment prefix for trade identification (vs Klaus's "K|")
  - Commodity data broadcast for commodity_fx_corr in HFT mode
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from stepsister.config.settings import get_fx_settings
from stepsister.core.registry import get_all_fx_algorithms
from klaus.core.types import Direction, Regime, Signal, TradeRequest
from klaus.data.feature_store import FeatureStore
from klaus.data.hft_feature_store import HFTFeatureStore
from klaus.data.market_data import MarketData
from klaus.data.mt5_client import MT5Client
from klaus.execution.executor import Executor
from klaus.regime.hmm_detector import HMMRegimeDetector
from klaus.risk.hft_risk_manager import HFTRiskManager
from klaus.risk.trailing_stop import TrailingStopManager


# FX HFT algorithm names (for filtering the registry)
_FX_HFT_ALGO_NAMES = {
    "fx_tick_scalper",
    "fx_order_flow",
    "fx_micro_reversion",
    "fx_momentum_hft",
    "fx_vol_breakout",
    "fx_cross_pair_lead_lag",
    "fx_spread_scalper",
    "fx_dqn_scalper",
    "fx_xgboost_hft",
}

# Known forex spread pair relationships
_FX_SPREAD_PAIRS = {
    "EURUSD": "GBPUSD",
    "GBPUSD": "EURUSD",
    "AUDUSD": "NZDUSD",
    "NZDUSD": "AUDUSD",
    "USDCAD": "USDNOK",
    "USDNOK": "USDCAD",
}

# Leader data needed by fx_cross_pair_lead_lag
_FX_LEADER_MAP = {
    "GBPUSD": "EURUSD",
    "NZDUSD": "AUDUSD",
    "USDCHF": "EURUSD",
    "EURJPY": "EURUSD",
    "GBPJPY": "GBPUSD",
}

# Commodity symbols to broadcast for commodity_fx_corr in HFT mode
_COMMODITY_SYMBOLS = ["XAUUSD", "XTIUSD"]


class FXHFTEngine:
    """High-frequency forex orchestrator running on 5-second cycles with M1 data.

    Operates in parallel with the standard FXEngine. Each cycle:
      1. Manage trailing stops on existing positions
      2. Fetch M1 data for all HFT FX instruments
      3. Compute HFT features (microstructure + standard)
      4. Update fast regime detection (every 15 min on M5 data)
      5. Broadcast cross-asset data to spread/lead-lag algos
      6. Broadcast commodity data for commodity_fx_corr (if in HFT mode)
      7. Generate signals from HFT algorithms
      8. Pass through HFT risk manager (comment prefix "S|")
      9. Execute approved trades
    """

    def __init__(self, hft_config: dict = None):
        self._settings = get_fx_settings()
        self._client = MT5Client()
        self._market_data = MarketData(self._client)
        self._executor = Executor(self._client)

        cfg = hft_config or {}
        self._cycle_interval = cfg.get("cycle_interval", 5)
        self._data_bars = cfg.get("data_bars", 500)
        self._regime_interval_min = cfg.get("regime_interval_minutes", 15)
        self._refit_interval_hours = cfg.get("refit_interval_hours", 4)

        # Use Klaus's HFTRiskManager — config comes from stepsister's risk.yaml
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
        """Active FX instruments that have HFT algorithms mapped."""
        all_active = self._settings.instrument_list
        hft_mapped = set()
        mapping = self._settings.regimes.get("hft_mapping", {})
        for symbol in mapping:
            if symbol in all_active or symbol in [
                i["symbol"] for i in self._settings.instruments.get("instruments", [])
                if i.get("active", True)
            ]:
                hft_mapped.add(symbol)

        # Fallback: use all active instruments (forex pairs typically support HFT)
        if not hft_mapped:
            hft_mapped = set(all_active)

        return list(hft_mapped)

    def start(self) -> None:
        """Connect to MT5 and begin the HFT forex loop."""
        logger.info("=" * 60)
        logger.info("  STEP SISTER HFT Engine starting...")
        logger.info(f"  Cycle interval: {self._cycle_interval}s")
        logger.info("=" * 60)

        if not self._client.connect():
            logger.error("Step Sister HFT: Failed to connect to MT5. Aborting.")
            return

        self._init_hft_algorithms()
        self._running = True

        instruments = self.hft_instruments
        logger.info(f"Step Sister HFT instruments: {instruments}")
        logger.info(f"Step Sister HFT algorithms loaded: {list(self._algo_instances.keys())}")

        try:
            while self._running:
                cycle_start = time.time()
                self._run_hft_cycle(instruments)
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self._cycle_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Step Sister HFT: Shutdown requested")
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully shut down."""
        self._running = False
        self._client.disconnect()
        logger.info(f"Step Sister HFT Engine stopped. Daily trades: {self._risk_manager.daily_trade_count}")

    def _init_hft_algorithms(self) -> None:
        """Instantiate only HFT algorithms from the FX registry."""
        algo_configs = self._settings.algorithms.get("algorithms", {})
        for name, cls in get_all_fx_algorithms().items():
            if name in _FX_HFT_ALGO_NAMES:
                params = algo_configs.get(name, {})
                self._algo_instances[name] = cls(params=params)
                logger.debug(f"Step Sister HFT: Initialised {name}")

    def _run_hft_cycle(self, instruments: list[str]) -> None:
        """Execute one HFT cycle."""
        # Phase 0: Manage trailing stops on existing Step Sister positions
        try:
            self._manage_trailing_stops()
        except Exception as e:
            logger.error(f"Step Sister trailing stop error: {e}")

        # Phase 1: Fetch all M1 data (batch to reduce latency)
        self._m1_cache.clear()
        for symbol in instruments:
            try:
                df = self._market_data.get_bars(symbol, "1m", self._data_bars, use_cache=False)
                if not df.empty and len(df) >= 30:
                    df = HFTFeatureStore.add_all_hft_features(df)
                    self._m1_cache[symbol] = df
            except Exception as e:
                logger.error(f"Step Sister HFT data error {symbol}: {e}")

        if not self._m1_cache:
            logger.info("Step Sister HFT cycle: no data available")
            return

        logger.info(f"--- Step Sister HFT cycle --- {len(self._m1_cache)} instruments loaded ---")

        # Phase 2: Broadcast cross-asset data to pair/lead-lag algos
        self._broadcast_cross_asset_data()

        # Phase 2b: Broadcast commodity data for commodity_fx_corr in HFT mode
        self._broadcast_commodity_data()

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
                logger.error(f"Step Sister HFT processing error {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Phase 4: Risk-check and execute — override comment prefix to "S|"
        for signal in all_signals:
            try:
                trade_request = self._risk_manager.evaluate(signal)
                if trade_request is not None:
                    # Override comment prefix from "K|" to "S|" for Step Sister identification
                    trade_request.comment = f"S|{signal.algo_name[:20]}"

                    result = self._executor.execute(trade_request)
                    logger.info(
                        f"Step Sister HFT executed: {result.symbol} {result.direction.name} "
                        f"{result.volume} @ {result.price} [{result.status.name}] "
                        f"algo={signal.algo_name}"
                    )
                    # Feed rejection/fill info back to risk manager
                    from klaus.core.types import OrderStatus
                    if result.status == OrderStatus.REJECTED and result.error_code in (
                        10019, 10014, 10015, 10016, 10019, 10031,
                    ):
                        self._risk_manager.record_rejection(result.symbol, result.error_code)
                    elif result.status == OrderStatus.FILLED:
                        self._risk_manager.clear_rejection(result.symbol)
            except Exception as e:
                logger.error(f"Step Sister HFT execution error: {e}")

    def _manage_trailing_stops(self) -> None:
        """Manage trailing stops only for Step Sister positions.

        The underlying TrailingStopManager checks for "K|" or "Klaus" —
        we intercept and filter to "S|" or "StepSis" positions first.
        """
        positions = self._client.get_positions()
        fx_hft_positions = [
            p for p in positions
            if p.algo_name.startswith("S|")
            or "StepSis" in p.algo_name
            or "fx_hft" in p.algo_name.lower()
            or p.algo_name.startswith("fx_")
        ]

        now = datetime.utcnow()

        for pos in fx_hft_positions:
            try:
                self._trailing_stop._manage_position(pos, now)
            except Exception as e:
                logger.error(f"Step Sister TrailingStop error ticket={pos.ticket}: {e}")

        # Clean up breakeven set for closed positions
        open_tickets = {p.ticket for p in fx_hft_positions}
        self._trailing_stop._breakeven_set = self._trailing_stop._breakeven_set & open_tickets

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
        """Share M1 data between cross-asset FX algorithms."""
        # FX spread scalper: pair data
        for algo_name in ("fx_spread_scalper",):
            algo = self._algo_instances.get(algo_name)
            if algo is None:
                continue
            for symbol, pair_symbol in _FX_SPREAD_PAIRS.items():
                if pair_symbol in self._m1_cache:
                    try:
                        algo.set_pair_data(self._m1_cache[pair_symbol])
                    except AttributeError:
                        pass

        # FX cross-pair lead-lag: leader data
        lead_lag_algo = self._algo_instances.get("fx_cross_pair_lead_lag")
        if lead_lag_algo is not None:
            for lagger, leader in _FX_LEADER_MAP.items():
                if leader in self._m1_cache:
                    try:
                        lead_lag_algo.set_leader_data(leader, self._m1_cache[leader])
                    except AttributeError:
                        pass

    def _broadcast_commodity_data(self) -> None:
        """Broadcast commodity data (XAUUSD, XTIUSD) to commodity_fx_corr algo.

        If commodity_fx_corr is registered in HFT mode, feed it M1
        commodity data for correlation-based forex signals.
        """
        corr_algo = self._algo_instances.get("commodity_fx_corr")
        if corr_algo is None:
            return

        for commodity_symbol in _COMMODITY_SYMBOLS:
            try:
                df = self._market_data.get_bars(commodity_symbol, "1m", self._data_bars, use_cache=False)
                if not df.empty and len(df) >= 30:
                    df = HFTFeatureStore.add_all_hft_features(df)
                    try:
                        corr_algo.set_commodity_data(commodity_symbol, df)
                    except AttributeError:
                        pass
            except Exception as e:
                logger.error(f"Step Sister HFT commodity data error {commodity_symbol}: {e}")

    def _get_hft_algorithms(self, symbol: str, regime: Regime) -> list[str]:
        """Get HFT algorithms for a given FX instrument and regime.

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

    def _count_fx_hft_positions(self) -> int:
        """Count open positions belonging to Step Sister HFT."""
        positions = self._client.get_positions()
        return len([
            p for p in positions
            if p.algo_name.startswith("S|")
            or "StepSis" in p.algo_name
            or "fx_hft" in p.algo_name.lower()
            or p.algo_name.startswith("fx_")
        ])

    def run_once(self) -> None:
        """Run a single HFT cycle (for testing)."""
        if not self._client.is_connected:
            if not self._client.connect():
                logger.error("Step Sister HFT: Cannot connect to MT5")
                return
        self._init_hft_algorithms()
        instruments = self.hft_instruments
        self._run_hft_cycle(instruments)
