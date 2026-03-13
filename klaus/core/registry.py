"""Decorator-based algorithm registry.

Usage:
    @register_algorithm
    class MyAlgo(BaseAlgorithm):
        ...

    # Later:
    algos = get_all_algorithms()
    algo = get_algorithm("my_algo")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klaus.algorithms.base import BaseAlgorithm

_ALGORITHM_REGISTRY: dict[str, type[BaseAlgorithm]] = {}


def register_algorithm(cls: type[BaseAlgorithm]) -> type[BaseAlgorithm]:
    """Class decorator that registers an algorithm by its `name` attribute."""
    name = getattr(cls, "name", None)
    if name is None:
        raise ValueError(f"{cls.__name__} must define a 'name' class attribute")
    if name in _ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{name}' already registered by {_ALGORITHM_REGISTRY[name].__name__}")
    _ALGORITHM_REGISTRY[name] = cls
    return cls


def get_algorithm(name: str) -> type[BaseAlgorithm]:
    """Get a registered algorithm class by name."""
    if name not in _ALGORITHM_REGISTRY:
        raise KeyError(f"Algorithm '{name}' not found. Available: {list(_ALGORITHM_REGISTRY.keys())}")
    return _ALGORITHM_REGISTRY[name]


def get_all_algorithms() -> dict[str, type[BaseAlgorithm]]:
    """Return a copy of the full registry."""
    return dict(_ALGORITHM_REGISTRY)


def clear_registry() -> None:
    """Clear all registered algorithms (for testing)."""
    _ALGORITHM_REGISTRY.clear()
