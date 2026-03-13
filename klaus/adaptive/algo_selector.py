"""Adaptive algorithm selector using Thompson Sampling.

Instead of blindly running every algorithm the rules engine says to run,
this module learns which algorithms actually perform well for each
(instrument, regime) pair and adjusts selection accordingly.

Uses Thompson Sampling (Bayesian bandit):
  - Each algo has a Beta(alpha, beta) distribution for its win probability
  - On each cycle, sample from each algo's distribution
  - Algos with higher sampled values are more likely to be selected
  - After each trade outcome, update the distribution

This naturally balances exploration (trying underperforming algos occasionally)
with exploitation (favouring proven winners).

Key principle: NEVER completely disable an algorithm. Even poor performers
get occasional chances — markets change, and an algo that failed last month
might be perfect next month.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from loguru import logger

_STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "adaptive" / "bandit_state.json"


@dataclass
class BanditArm:
    """Thompson Sampling arm for one (algo, symbol, regime) combo."""
    algo_name: str
    symbol: str
    regime: str

    # Beta distribution parameters (prior: Beta(1,1) = uniform)
    alpha: float = 1.0  # Successes + prior
    beta: float = 1.0   # Failures + prior

    # Decay parameters — slowly forget old results so the bandit adapts
    total_updates: int = 0

    @property
    def key(self) -> str:
        return f"{self.algo_name}|{self.symbol}|{self.regime}"

    @property
    def mean(self) -> float:
        """Expected win probability."""
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Draw a sample from Beta(alpha, beta)."""
        return random.betavariate(max(self.alpha, 0.01), max(self.beta, 0.01))

    def update(self, won: bool, decay: float = 0.998) -> None:
        """Update the arm after observing an outcome.

        Args:
            won: Whether the trade was profitable.
            decay: Multiplicative decay applied to alpha and beta before update.
                   This causes old observations to gradually lose influence,
                   allowing the bandit to adapt to changing markets.
                   0.998 ≈ half-life of ~350 updates (slow adaptation).
                   0.995 ≈ half-life of ~140 updates (moderate).
                   0.990 ≈ half-life of ~70 updates (fast adaptation).
        """
        # Decay old observations
        self.alpha *= decay
        self.beta *= decay

        # Clamp to prevent collapse (minimum prior strength)
        self.alpha = max(self.alpha, 0.5)
        self.beta = max(self.beta, 0.5)

        # Update with new observation
        if won:
            self.alpha += 1.0
        else:
            self.beta += 1.0

        self.total_updates += 1


class AdaptiveAlgoSelector:
    """Selects and ranks algorithms using Thompson Sampling.

    Sits between the rules engine and signal generation. The rules engine
    provides candidate algorithms; this module decides which ones to
    actually run and in what priority order.

    Design: Never completely removes an algorithm. Instead:
      - Strong performers: Always run, signal strength boosted
      - Average performers: Usually run, no modification
      - Weak performers: Run with reduced probability (but still occasionally)
    """

    def __init__(
        self,
        min_selection_prob: float = 0.15,
        decay: float = 0.998,
        state_path: Path = _STATE_PATH,
    ):
        """
        Args:
            min_selection_prob: Minimum probability of selecting any algo (exploration floor).
            decay: Bandit decay rate. Controls adaptation speed.
        """
        self._min_prob = min_selection_prob
        self._decay = decay
        self._state_path = state_path
        self._arms: dict[str, BanditArm] = {}
        self._load_state()

    def _get_or_create(self, algo_name: str, symbol: str, regime: str) -> BanditArm:
        key = f"{algo_name}|{symbol}|{regime}"
        if key not in self._arms:
            self._arms[key] = BanditArm(algo_name=algo_name, symbol=symbol, regime=regime)
        return self._arms[key]

    def select_algorithms(
        self,
        symbol: str,
        regime: str,
        candidate_algos: list[str],
    ) -> list[str]:
        """Select which algorithms to run this cycle.

        Uses Thompson Sampling to probabilistically select algos.
        Every algo has at least min_selection_prob chance of being selected.

        Returns:
            List of algorithm names to run (subset or all of candidates).
        """
        if not candidate_algos:
            return []

        selected = []
        samples = []

        for algo_name in candidate_algos:
            arm = self._get_or_create(algo_name, symbol, regime)
            sample = arm.sample()
            samples.append((algo_name, sample))

        # Sort by sampled value (best first)
        samples.sort(key=lambda x: x[1], reverse=True)

        for algo_name, sample in samples:
            # Always include if sample is above 0.5 (Thompson says it's likely good)
            if sample >= 0.5:
                selected.append(algo_name)
            else:
                # Below 0.5: still include with min_selection_prob
                if random.random() < self._min_prob:
                    selected.append(algo_name)

        # Safety: always run at least one algorithm
        if not selected and candidate_algos:
            best = samples[0][0]
            selected.append(best)

        return selected

    def get_strength_modifier(self, algo_name: str, symbol: str, regime: str) -> float:
        """Get a signal strength modifier based on the bandit's belief.

        Returns 0.5-1.3: dampens signals from poor algos, boosts good ones.
        """
        arm = self._get_or_create(algo_name, symbol, regime)
        mean = arm.mean  # Expected win rate

        # Map [0, 1] win probability → [0.5, 1.3] modifier
        # 0.5 win rate → 1.0 (neutral)
        # 0.8 win rate → 1.3 (boost)
        # 0.2 win rate → 0.5 (dampen)
        modifier = 0.5 + 0.8 * mean
        return min(max(modifier, 0.5), 1.3)

    def record_outcome(self, algo_name: str, symbol: str, regime: str, won: bool) -> None:
        """Update the bandit after a trade closes."""
        arm = self._get_or_create(algo_name, symbol, regime)
        arm.update(won, decay=self._decay)

        logger.debug(
            f"Bandit update: {arm.key} {'WIN' if won else 'LOSS'} "
            f"α={arm.alpha:.1f} β={arm.beta:.1f} mean={arm.mean:.3f}"
        )

        # Auto-save every 20 updates
        if sum(a.total_updates for a in self._arms.values()) % 20 == 0:
            self._save_state()

    def get_rankings(self, symbol: str, regime: str) -> list[tuple[str, float, int]]:
        """Get all algos ranked by expected win rate for a context.

        Returns: [(algo_name, expected_win_rate, total_updates), ...]
        """
        relevant = [
            a for a in self._arms.values()
            if a.symbol == symbol and a.regime == regime
        ]
        relevant.sort(key=lambda a: a.mean, reverse=True)
        return [(a.algo_name, a.mean, a.total_updates) for a in relevant]

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {key: asdict(arm) for key, arm in self._arms.items()}
        with open(self._state_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Bandit: saved {len(data)} arms to disk")

    def _load_state(self) -> None:
        if not self._state_path.exists():
            logger.info("Bandit: no prior state, starting with uniform priors")
            return
        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)
            for key, arm_dict in data.items():
                self._arms[key] = BanditArm(**arm_dict)
            logger.info(f"Bandit: restored {len(self._arms)} arms from disk")
        except Exception as e:
            logger.warning(f"Bandit: failed to load state: {e}, starting fresh")

    def save(self) -> None:
        """Public save (call on shutdown)."""
        self._save_state()

    def summary(self, symbol: str = None) -> str:
        """Human-readable summary."""
        arms = list(self._arms.values())
        if symbol:
            arms = [a for a in arms if a.symbol == symbol]

        if not arms:
            return "No bandit data yet."

        arms.sort(key=lambda a: a.mean, reverse=True)
        lines = ["=== Adaptive Algo Selection (Thompson Sampling) ==="]
        for arm in arms[:10]:
            lines.append(
                f"  {arm.key}: win_prob={arm.mean:.3f} "
                f"α={arm.alpha:.1f} β={arm.beta:.1f} "
                f"updates={arm.total_updates}"
            )
        return "\n".join(lines)
