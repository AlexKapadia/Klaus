"""Adaptive online learning system for Klaus.

Continuously learns from trade outcomes while running — no resets needed.
All state persists to disk (SQLite + JSON) and resumes across restarts.

Components:
  - TradeMemory:        Persistent database of every trade + outcome
  - PerformanceTracker: EWM-scored per-algo/instrument/regime performance
  - AlgoSelector:       Thompson Sampling bandit for algo selection
  - OnlineLearner:      Incremental ML model updates from new data
  - AdaptiveRisk:       Dynamic position sizing based on algo performance
"""

from klaus.adaptive.trade_memory import TradeMemory
from klaus.adaptive.performance_tracker import PerformanceTracker
from klaus.adaptive.algo_selector import AdaptiveAlgoSelector
from klaus.adaptive.online_learner import OnlineLearner
from klaus.adaptive.adaptive_risk import AdaptiveRisk

__all__ = [
    "TradeMemory",
    "PerformanceTracker",
    "AdaptiveAlgoSelector",
    "OnlineLearner",
    "AdaptiveRisk",
]
