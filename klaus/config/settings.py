"""Pydantic settings loader — reads .env + YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = Path(__file__).resolve().parent


def _load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


class MT5Settings(BaseSettings):
    login: int = 0
    password: str = ""
    server: str = ""
    path: str = ""

    model_config = {"env_prefix": "MT5_", "extra": "ignore"}


class RiskSettings(BaseSettings):
    max_drawdown_pct: float = 10.0
    drawdown_reduce_pct: float = 5.0
    cppi_floor_pct: float = 90.0
    cppi_multiplier: float = 3.0
    max_positions: int = 8
    max_per_instrument: int = 1
    max_correlated_positions: int = 3
    correlation_threshold: float = 0.7
    max_risk_per_trade_pct: float = 0.5
    default_sl_atr_mult: float = 2.0
    default_tp_atr_mult: float = 3.0
    kelly_fraction: float = 0.5  # half-Kelly

    model_config = {"env_prefix": "", "extra": "ignore"}


class Settings:
    """Central configuration aggregating .env and YAML files."""

    def __init__(self):
        self.mt5 = MT5Settings(_env_file=PROJECT_ROOT / ".env")
        self.risk = RiskSettings(_env_file=PROJECT_ROOT / ".env")
        self.instruments: dict[str, Any] = _load_yaml("instruments.yaml")
        self.algorithms: dict[str, Any] = _load_yaml("algorithms.yaml")
        self.regimes: dict[str, Any] = _load_yaml("regimes.yaml")
        self.risk_yaml: dict[str, Any] = _load_yaml("risk.yaml")

        # Merge YAML risk overrides into risk settings
        if self.risk_yaml:
            for key, val in self.risk_yaml.items():
                if hasattr(self.risk, key):
                    setattr(self.risk, key, val)

    @property
    def regime_algo_mapping(self) -> dict:
        """Return the regime → algorithm mapping from regimes.yaml."""
        return self.regimes.get("mapping", {})

    @property
    def instrument_list(self) -> list[str]:
        """Return list of active instrument symbols."""
        return [
            inst["symbol"]
            for inst in self.instruments.get("instruments", [])
            if inst.get("active", True)
        ]


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
