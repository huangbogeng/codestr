from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class UDFMeta:
    """Metadata for a registered user-defined function.

    Attributes:
        name: Unique function name used in DSL expressions.
        fn: The callable (must accept and return Polars expressions).
        category: Operator category — ``"math"``, ``"cs"`` (cross-section),
                  ``"ts"`` (time-series), or ``"user"`` for custom registrations.
        arity: Number of required positional arguments. Auto-inferred from the
               function signature if not provided.
    """

    name: str
    fn: Callable
    category: str = "math"
    arity: int | None = None


class UDFRegistry:
    """Singleton registry mapping function names to UDFMeta entries.

    Use ``UDFRegistry.get_instance()`` to obtain the global registry, or
    ``@udf`` / ``CodeStr.register_udf()`` for convenient registration.
    """

    _instance: UDFRegistry | None = None

    def __init__(self):
        self._registry: dict[str, UDFMeta] = {}

    @classmethod
    def get_instance(cls) -> UDFRegistry:
        """Return the global singleton UDFRegistry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton instance (useful for testing)."""
        cls._instance = None

    def register(self, meta: UDFMeta) -> UDFMeta:
        """Register a new UDF. Overwrites any existing entry with the same name."""
        self._registry[meta.name] = meta
        return meta

    def get(self, name: str) -> UDFMeta | None:
        """Look up a UDF by name, returning None if not found."""
        return self._registry.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __getitem__(self, name: str) -> UDFMeta:
        """Look up a UDF by name, raising KeyError if not found."""
        if name not in self._registry:
            raise KeyError(f"UDF not registered: {name}")
        return self._registry[name]

    def all(self) -> list[UDFMeta]:
        """Return all registered UDFs."""
        return list(self._registry.values())


def _infer_arity(fn: Callable) -> int:
    """Count required positional parameters (those without defaults)."""
    sig = inspect.signature(fn)
    return sum(1 for p in sig.parameters.values() if p.default is inspect.Parameter.empty)


def udf(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    category: str = "math",
    arity: int | None = None,
) -> Callable:
    """Decorator to register a function as a DSL operator.

    Can be used as ``@udf`` or ``@udf(category="ts", arity=2)``.

    Args:
        fn: The decorated function (when used without arguments).
        name: DSL name for the operator (defaults to the Python function name).
        category: Operator category — ``"math"``, ``"cs"``, ``"ts"``, or ``"user"``.
        arity: Number of required arguments. Auto-inferred if not provided.

    Example:
        >>> @udf(category="math")
        ... def triple(x: pl.Expr) -> pl.Expr:
        ...     return x * 3
    """

    def _decorator(f: Callable) -> Callable:
        meta = UDFMeta(
            name=name if name is not None else f.__name__,
            fn=f,
            category=category,
            arity=arity if arity is not None else _infer_arity(f),
        )
        UDFRegistry.get_instance().register(meta)
        return f

    if fn is not None:
        return _decorator(fn)
    return _decorator
