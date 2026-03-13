"""Decorator-based FX algorithm registry for the stepsister forex platform.

Usage:
    @register_fx_algorithm
    class MyFxAlgo(BaseAlgorithm):
        ...

    # Later:
    algos = get_all_fx_algorithms()
    algo = get_fx_algorithm("my_fx_algo")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klaus.algorithms.base import BaseAlgorithm

_FX_ALGORITHM_REGISTRY: dict[str, type[BaseAlgorithm]] = {}


def register_fx_algorithm(cls: type[BaseAlgorithm]) -> type[BaseAlgorithm]:
    """Class decorator that registers an FX algorithm by its `name` attribute."""
    name = getattr(cls, "name", None)
    if name is None:
        raise ValueError(f"{cls.__name__} must define a 'name' class attribute")
    if name in _FX_ALGORITHM_REGISTRY:
        raise ValueError(f"FX algorithm '{name}' already registered by {_FX_ALGORITHM_REGISTRY[name].__name__}")
    _FX_ALGORITHM_REGISTRY[name] = cls
    return cls


def get_fx_algorithm(name: str) -> type[BaseAlgorithm]:
    """Get a registered FX algorithm class by name."""
    if name not in _FX_ALGORITHM_REGISTRY:
        raise KeyError(f"FX algorithm '{name}' not found. Available: {list(_FX_ALGORITHM_REGISTRY.keys())}")
    return _FX_ALGORITHM_REGISTRY[name]


def get_all_fx_algorithms() -> dict[str, type[BaseAlgorithm]]:
    """Return a copy of the full FX registry."""
    return dict(_FX_ALGORITHM_REGISTRY)


def clear_fx_registry() -> None:
    """Clear all registered FX algorithms (for testing)."""
    _FX_ALGORITHM_REGISTRY.clear()
