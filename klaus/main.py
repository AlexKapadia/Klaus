"""Klaus — AI-Driven Algorithmic Trading Platform.

Entry points:
    Standard engine:  python -m klaus.main
    HFT engine:       python -m klaus.main --hft
    Both engines:     python -m klaus.main --both
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from klaus.monitoring.logger import setup_logger


def main():
    setup_logger()
    logger.info("=" * 60)
    logger.info("  KLAUS — AI-Driven Algorithmic Trading Platform")
    logger.info("=" * 60)

    # Import all algorithm modules so they auto-register via decorators
    _import_algorithms()

    mode = _parse_mode()

    if mode == "standard":
        _run_standard()
    elif mode == "hft":
        _run_hft()
    elif mode == "both":
        _run_both()


def _parse_mode() -> str:
    """Parse command-line arguments for engine mode."""
    args = sys.argv[1:]
    if "--hft" in args:
        return "hft"
    elif "--both" in args:
        return "both"
    return "standard"


def _run_standard():
    """Run only the standard (H1) engine."""
    from klaus.orchestrator.engine import Engine
    logger.info("Mode: STANDARD (H1 bars, 60s cycles)")
    engine = Engine()
    engine.start()


def _run_hft():
    """Run only the HFT (M1) engine."""
    from klaus.orchestrator.hft_engine import HFTEngine
    from klaus.config.settings import get_settings

    settings = get_settings()
    hft_config = settings.risk_yaml.get("hft", {})

    logger.info("Mode: HFT (M1 bars, 5s cycles)")
    engine = HFTEngine(hft_config)
    engine.start()


def _run_both():
    """Run both engines in parallel threads."""
    from klaus.orchestrator.engine import Engine
    from klaus.orchestrator.hft_engine import HFTEngine
    from klaus.config.settings import get_settings

    settings = get_settings()
    hft_config = settings.risk_yaml.get("hft", {})

    logger.info("Mode: DUAL (Standard + HFT running in parallel)")

    standard_engine = Engine()
    hft_engine = HFTEngine(hft_config)

    # Run standard engine in a background thread
    std_thread = threading.Thread(
        target=standard_engine.start,
        name="Klaus-Standard",
        daemon=True,
    )
    std_thread.start()

    # HFT engine runs in the main thread (for KeyboardInterrupt handling)
    try:
        hft_engine.start()
    except KeyboardInterrupt:
        logger.info("Shutdown: stopping both engines")
    finally:
        standard_engine.stop()
        hft_engine.stop()


def _import_algorithms():
    """Import all algorithm modules to trigger @register_algorithm decorators."""
    # ── Standard algorithms ──────────────────────────────────────
    import klaus.algorithms.momentum.sma_crossover
    import klaus.algorithms.mean_reversion.bollinger
    import klaus.algorithms.mean_reversion.zscore
    import klaus.algorithms.stat_arb.spread_ou
    import klaus.algorithms.stat_arb.gold_silver_ratio
    import klaus.algorithms.stat_arb.soybean_crush
    import klaus.algorithms.volatility.gjr_garch
    import klaus.algorithms.volatility.har
    import klaus.algorithms.seasonal.natgas_calendar
    import klaus.algorithms.seasonal.ag_growing
    import klaus.algorithms.geopolitical.gpr_index

    # ML algorithms (import only if dependencies are available)
    try:
        import klaus.algorithms.ml_signals.xgboost_signal
    except ImportError:
        logger.warning("XGBoost not available — xgboost_signal disabled")
    try:
        import klaus.algorithms.ml_signals.lstm_signal
        import klaus.algorithms.ml_signals.tcn_signal
    except ImportError:
        logger.warning("PyTorch not available — LSTM/TCN signals disabled")

    # ── HFT algorithms ───────────────────────────────────────────
    try:
        import klaus.algorithms.hft.order_flow_imbalance
        import klaus.algorithms.hft.tick_scalper
        import klaus.algorithms.hft.volatility_breakout_hft
        import klaus.algorithms.hft.microstructure_reversion
        import klaus.algorithms.hft.spread_scalper_hft
        import klaus.algorithms.hft.cross_commodity_lead_lag
        import klaus.algorithms.hft.hft_momentum
    except ImportError as e:
        logger.warning(f"HFT algorithms partially unavailable: {e}")

    # ML HFT algorithms
    try:
        import klaus.algorithms.ml_hft.dqn_scalper
    except ImportError:
        logger.warning("PyTorch not available — DQN scalper disabled")
    try:
        import klaus.algorithms.ml_hft.xgboost_hft
    except ImportError:
        logger.warning("XGBoost not available — xgboost_hft disabled")


if __name__ == "__main__":
    main()
