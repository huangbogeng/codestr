from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codestr.syntax import Call, ExprNode, KeywordArg

if TYPE_CHECKING:
    from codestr.udf.registry import UDFRegistry

WindowDomain = Literal["ts", "cs"]


@dataclass(frozen=True)
class WindowRef:
    call_name: str
    domain: WindowDomain


@dataclass(frozen=True)
class MixedWindowBoundary:
    outer_call: str
    outer_domain: WindowDomain
    inner_call: str
    inner_domain: WindowDomain


def _arg_value(arg):
    return arg.value if isinstance(arg, KeywordArg) else arg


def _call_domain(call: Call, registry: UDFRegistry) -> WindowDomain | None:
    meta = registry.get(call.fn_name)
    if meta is None or meta.category not in {"ts", "cs"}:
        return None
    return meta.category


def find_mixed_window_boundary(
    node: ExprNode,
    registry: UDFRegistry,
) -> MixedWindowBoundary | None:
    def visit(
        current: ExprNode,
        active_window: WindowRef | None,
    ) -> MixedWindowBoundary | None:
        if not isinstance(current, Call):
            return None

        # Unknown calls remain opaque so the compiler reports the original error.
        if registry.get(current.fn_name) is None:
            return None

        domain = _call_domain(current, registry)
        if domain is not None and active_window is not None and domain != active_window.domain:
            return MixedWindowBoundary(
                outer_call=active_window.call_name,
                outer_domain=active_window.domain,
                inner_call=current.fn_name,
                inner_domain=domain,
            )

        next_window = WindowRef(current.fn_name, domain) if domain is not None else active_window
        for arg in current.args:
            boundary = visit(_arg_value(arg), next_window)
            if boundary is not None:
                return boundary
        return None

    return visit(node, None)
