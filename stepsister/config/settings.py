"""Pydantic settings loader for the Step Sister forex platform.

Mirrors Klaus's settings architecture but reads from the stepsister
config directory and uses FX-specific defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STEPSISTER_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent


def _load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


class FXSettings:
    """Central configuration for the Step Sister forex platform.

    Reads YAML configs from stepsister/config/.
    Falls back to empty dicts when YAML files are absent.
    Reuses Klaus's MT5 connection settings (shared terminal).
    """

    def __init__(self):
        from klaus.config.settings import get_settings as _get_klaus_settings
        _klaus = _get_klaus_settings()
        self.mt5 = _klaus.mt5

        self.instruments: dict[str, Any] = _load_yaml("instruments.yaml")
        self.algorithms: dict[str, Any] = _load_yaml("algorithms.yaml")
        self.regimes: dict[str, Any] = _load_yaml("regimes.yaml")
        self.risk_yaml: dict[str, Any] = _load_yaml("risk.yaml")

    @property
    def regime_algo_mapping(self) -> dict:
        """Return the regime -> algorithm mapping from regimes.yaml."""
        return self.regimes.get("mapping", {})

    @property
    def instrument_list(self) -> list[str]:
        """Return list of active FX instrument symbols."""
        return [
            inst["symbol"]
            for inst in self.instruments.get("instruments", [])
            if inst.get("active", True)
        ]


# Singleton
_fx_settings: FXSettings | None = None


def get_fx_settings() -> FXSettings:
    global _fx_settings
    if _fx_settings is None:
        _fx_settings = FXSettings()
    return _fx_settings
