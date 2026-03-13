"""Step Sister — AI-Driven Forex Trading Platform (Klaus's Step Sister).

Entry points:
    Standard engine:  python -m stepsister.main
    HFT engine:       python -m stepsister.main --hft
    Both engines:     python -m stepsister.main --both
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
from stepsister.monitoring.logger import setup_fx_logger


def main():
    setup_fx_logger()
    logger.info("=" * 60)
    logger.info("  STEP SISTER — AI-Driven Forex Trading Platform")
    logger.info("  (Klaus's Step Sister)")
    logger.info("=" * 60)

    # Import all forex algorithm modules so they auto-register
    _import_fx_algorithms()

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
    """Run only the standard (H1) forex engine."""
    from stepsister.orchestrator.engine import FXEngine
    logger.info("Mode: STANDARD (H1 bars, 60s cycles)")
    engine = FXEngine()
    engine.start()


def _run_hft():
    """Run only the HFT (M1) forex engine."""
    from stepsister.orchestrator.hft_engine import FXHFTEngine
    from stepsister.config.settings import get_fx_settings

    settings = get_fx_settings()
    hft_config = settings.risk_yaml.get("hft", {})

    logger.info("Mode: HFT (M1 bars, 1s cycles)")
    engine = FXHFTEngine(hft_config)
    engine.start()


def _run_both():
    """Run both forex engines in parallel threads."""
    from stepsister.orchestrator.engine import FXEngine
    from stepsister.orchestrator.hft_engine import FXHFTEngine
    from stepsister.config.settings import get_fx_settings

    settings = get_fx_settings()
    hft_config = settings.risk_yaml.get("hft", {})

    logger.info("Mode: DUAL (Standard + HFT running in parallel)")

    standard_engine = FXEngine()
    hft_engine = FXHFTEngine(hft_config)

    # Run standard engine in a background thread
    std_thread = threading.Thread(
        target=standard_engine.start,
        name="StepSister-Standard",
        daemon=True,
    )
    std_thread.start()

    # HFT engine runs in the main thread (for KeyboardInterrupt handling)
    try:
        hft_engine.start()
    except KeyboardInterrupt:
        logger.info("Shutdown: stopping both forex engines")
    finally:
        standard_engine.stop()
        hft_engine.stop()


def _import_fx_algorithms():
    """Import all forex algorithm modules to trigger @register_fx_algorithm decorators."""
    # ── Standard forex algorithms ─────────────────────────────────
    import stepsister.algorithms.momentum.fx_sma_crossover
    import stepsister.algorithms.momentum.fx_momentum
    import stepsister.algorithms.carry.carry_trade
    import stepsister.algorithms.carry.feer_carry
    import stepsister.algorithms.mean_reversion.fx_bollinger
    import stepsister.algorithms.mean_reversion.fx_zscore
    import stepsister.algorithms.correlation.commodity_fx_corr

    # ML algorithms (import only if dependencies are available)
    try:
        import stepsister.algorithms.ml.fx_xgboost_signal
    except ImportError:
        logger.warning("XGBoost not available — fx_xgboost_signal disabled")
    try:
        import stepsister.algorithms.ml.fx_lstm_signal
    except ImportError:
        logger.warning("PyTorch not available — fx_lstm_signal disabled")

    # ── HFT forex algorithms ─────────────────────────────────────
    try:
        import stepsister.algorithms.hft.fx_tick_scalper
        import stepsister.algorithms.hft.fx_order_flow
        import stepsister.algorithms.hft.fx_micro_reversion
        import stepsister.algorithms.hft.fx_momentum_hft
        import stepsister.algorithms.hft.fx_vol_breakout
        import stepsister.algorithms.hft.fx_cross_pair_lead_lag
        import stepsister.algorithms.hft.fx_spread_scalper
    except ImportError as e:
        logger.warning(f"FX HFT algorithms partially unavailable: {e}")

    # ML HFT algorithms
    try:
        import stepsister.algorithms.ml.fx_dqn_scalper
    except ImportError:
        logger.warning("PyTorch not available — fx_dqn_scalper disabled")
    try:
        import stepsister.algorithms.ml.fx_xgboost_hft
    except ImportError:
        logger.warning("XGBoost not available — fx_xgboost_hft disabled")


if __name__ == "__main__":
    main()
