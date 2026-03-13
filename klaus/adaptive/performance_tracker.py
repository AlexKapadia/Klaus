"""Exponentially-weighted performance tracker.

Scores each (algorithm, instrument, regime) combination using recent trade
outcomes. More recent trades carry exponentially more weight, so the system
naturally adapts to changing market conditions without needing a reset.

Key metrics tracked per combination:
  - EWM win rate
  - EWM average P/L
  - EWM Sharpe ratio
  - Trade count (total and recent)
  - Confidence score (how sure we are about this algo's performance)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

_STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "adaptive" / "performance_state.json"


@dataclass
class AlgoScore:
    """Performance score for one (algo, instrument, regime) combination."""
    algo_name: str
    symbol: str
    regime: str

    # Exponentially-weighted stats
    ewm_win_rate: float = 0.5
    ewm_avg_pnl: float = 0.0
    ewm_pnl_variance: float = 1.0  # For Sharpe computation

    # Counts
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    streak: int = 0  # +N for win streak, -N for loss streak

    # Aggregate
    total_pnl: float = 0.0
    best_pnl: float = 0.0
    worst_pnl: float = 0.0

    # Timing
    last_trade_time: str = ""
    last_updated: str = ""

    @property
    def key(self) -> str:
        return f"{self.algo_name}|{self.symbol}|{self.regime}"

    @property
    def confidence(self) -> float:
        """How confident we are in this score (0-1). Grows with trade count."""
        # Sigmoid-like: reaches ~0.9 at 50 trades, ~0.95 at 100
        return 1.0 - math.exp(-self.total_trades / 30.0)

    @property
    def ewm_sharpe(self) -> float:
        """Sharpe-like ratio from EWM stats."""
        if self.ewm_pnl_variance <= 0:
            return 0.0
        std = math.sqrt(self.ewm_pnl_variance)
        if std < 1e-10:
            return 0.0
        return self.ewm_avg_pnl / std

    @property
    def composite_score(self) -> float:
        """Single composite performance score (0-1 range, 0.5 = neutral).

        Blends win rate, P/L, and Sharpe. Weighted by confidence so
        low-data algos stay near 0.5 (neutral).
        """
        # Normalise components to roughly 0-1
        wr_component = self.ewm_win_rate  # Already 0-1
        sharpe_component = min(max((self.ewm_sharpe + 2) / 4, 0), 1)  # Map [-2,2] → [0,1]
        pnl_sign = 0.5 + 0.5 * math.tanh(self.ewm_avg_pnl * 10)  # Sigmoid around 0

        raw = 0.4 * wr_component + 0.3 * sharpe_component + 0.3 * pnl_sign
        # Blend with 0.5 (neutral) based on confidence
        return self.confidence * raw + (1.0 - self.confidence) * 0.5


class PerformanceTracker:
    """Tracks and updates performance scores for all algo/instrument/regime combos.

    Uses exponential moving average (EMA) so recent outcomes matter more.
    Persists state to JSON on disk — survives restarts.
    """

    def __init__(self, decay: float = 0.05, state_path: Path = _STATE_PATH):
        """
        Args:
            decay: EMA decay factor. Higher = faster adaptation.
                   0.05 means ~20-trade half-life.
                   0.10 means ~10-trade half-life.
        """
        self._decay = decay
        self._state_path = state_path
        self._scores: dict[str, AlgoScore] = {}
        self._load_state()

    def _get_or_create(self, algo_name: str, symbol: str, regime: str) -> AlgoScore:
        key = f"{algo_name}|{symbol}|{regime}"
        if key not in self._scores:
            self._scores[key] = AlgoScore(
                algo_name=algo_name, symbol=symbol, regime=regime
            )
        return self._scores[key]

    def record_outcome(
        self,
        algo_name: str,
        symbol: str,
        regime: str,
        pnl: float,
        won: bool,
    ) -> AlgoScore:
        """Update the performance score after a trade closes.

        This is the core learning step. Called every time a trade completes.
        """
        score = self._get_or_create(algo_name, symbol, regime)
        alpha = self._decay

        # EWM updates
        score.ewm_win_rate = (1 - alpha) * score.ewm_win_rate + alpha * (1.0 if won else 0.0)
        score.ewm_avg_pnl = (1 - alpha) * score.ewm_avg_pnl + alpha * pnl

        # Welford-like online variance update for Sharpe
        diff = pnl - score.ewm_avg_pnl
        score.ewm_pnl_variance = (1 - alpha) * score.ewm_pnl_variance + alpha * (diff * diff)

        # Counts
        score.total_trades += 1
        if won:
            score.wins += 1
            score.streak = max(score.streak, 0) + 1
        else:
            score.losses += 1
            score.streak = min(score.streak, 0) - 1

        score.total_pnl += pnl
        score.best_pnl = max(score.best_pnl, pnl)
        score.worst_pnl = min(score.worst_pnl, pnl)

        now = datetime.utcnow().isoformat()
        score.last_trade_time = now
        score.last_updated = now

        # Auto-save periodically (every 10 trades across all combos)
        total = sum(s.total_trades for s in self._scores.values())
        if total % 10 == 0:
            self._save_state()

        logger.debug(
            f"PerfTracker: {algo_name}|{symbol}|{regime} "
            f"wr={score.ewm_win_rate:.2f} pnl={score.ewm_avg_pnl:.4f} "
            f"composite={score.composite_score:.3f} trades={score.total_trades}"
        )

        return score

    def get_score(self, algo_name: str, symbol: str, regime: str) -> AlgoScore:
        """Get current performance score for an algo/instrument/regime combo."""
        return self._get_or_create(algo_name, symbol, regime)

    def get_algo_scores(self, algo_name: str) -> list[AlgoScore]:
        """Get all scores for a given algorithm across instruments/regimes."""
        return [s for s in self._scores.values() if s.algo_name == algo_name]

    def get_instrument_scores(self, symbol: str, regime: str = None) -> list[AlgoScore]:
        """Get all algo scores for an instrument (optionally filtered by regime)."""
        results = [s for s in self._scores.values() if s.symbol == symbol]
        if regime:
            results = [s for s in results if s.regime == regime]
        return sorted(results, key=lambda s: s.composite_score, reverse=True)

    def get_signal_multiplier(self, algo_name: str, symbol: str, regime: str) -> float:
        """Get a signal strength multiplier based on learned performance.

        Returns:
            0.3 - 1.5 range. < 1.0 dampens poor performers, > 1.0 boosts good ones.
            Starts at 1.0 (neutral) and moves as data accumulates.
        """
        score = self._get_or_create(algo_name, symbol, regime)
        composite = score.composite_score  # 0-1, 0.5 = neutral

        # Map composite [0,1] → multiplier [0.3, 1.5]
        # 0.5 → 1.0 (neutral), 0.0 → 0.3 (dampen), 1.0 → 1.5 (boost)
        multiplier = 0.3 + 1.2 * composite
        return multiplier

    def rank_algorithms(self, symbol: str, regime: str, algo_names: list[str]) -> list[tuple[str, float]]:
        """Rank algorithms by composite score for a given context.

        Returns: list of (algo_name, composite_score) sorted best-first.
        """
        ranked = []
        for name in algo_names:
            score = self._get_or_create(name, symbol, regime)
            ranked.append((name, score.composite_score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _save_state(self) -> None:
        """Persist all scores to disk."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {key: asdict(score) for key, score in self._scores.items()}
        with open(self._state_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"PerfTracker: saved {len(data)} scores to disk")

    def _load_state(self) -> None:
        """Load scores from disk if available."""
        if not self._state_path.exists():
            logger.info("PerfTracker: no prior state found, starting fresh")
            return

        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)

            for key, score_dict in data.items():
                self._scores[key] = AlgoScore(**score_dict)

            logger.info(f"PerfTracker: restored {len(self._scores)} scores from disk")
        except Exception as e:
            logger.warning(f"PerfTracker: failed to load state: {e}, starting fresh")

    def save(self) -> None:
        """Public save (call on shutdown)."""
        self._save_state()

    def summary(self) -> str:
        """Human-readable summary of top/bottom performers."""
        if not self._scores:
            return "No performance data yet."

        scored = [(s.key, s.composite_score, s.total_trades, s.ewm_win_rate)
                  for s in self._scores.values() if s.total_trades >= 3]

        if not scored:
            return f"Tracking {len(self._scores)} combos, none with 3+ trades yet."

        scored.sort(key=lambda x: x[1], reverse=True)
        lines = ["=== Adaptive Performance Summary ==="]

        top = scored[:5]
        bottom = scored[-3:] if len(scored) > 5 else []

        lines.append("Top performers:")
        for key, comp, trades, wr in top:
            lines.append(f"  {key}: score={comp:.3f} wr={wr:.1%} trades={trades}")

        if bottom:
            lines.append("Bottom performers:")
            for key, comp, trades, wr in bottom:
                lines.append(f"  {key}: score={comp:.3f} wr={wr:.1%} trades={trades}")

        return "\n".join(lines)
