from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable


@dataclass
class UDFMeta:
    name: str
    fn: Callable
    category: str = "math"
    arity: int | None = None


class UDFRegistry:
    _instance: UDFRegistry | None = None

    def __init__(self):
        self._registry: dict[str, UDFMeta] = {}

    @classmethod
    def get_instance(cls) -> UDFRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton instance (useful for testing)."""
        cls._instance = None

    def register(self, meta: UDFMeta) -> UDFMeta:
        self._registry[meta.name] = meta
        return meta

    def get(self, name: str) -> UDFMeta | None:
        return self._registry.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __getitem__(self, name: str) -> UDFMeta:
        if name not in self._registry:
            raise KeyError(f"UDF not registered: {name}")
        return self._registry[name]

    def all(self) -> list[UDFMeta]:
        return list(self._registry.values())


def _infer_arity(fn: Callable) -> int:
    sig = inspect.signature(fn)
    return sum(1 for p in sig.parameters.values() if p.default is inspect.Parameter.empty)


def udf(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    category: str = "math",
    arity: int | None = None,
) -> Callable:
    """@udf or @udf(category="ts", arity=2)"""
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
