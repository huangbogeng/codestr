from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codestr.errors import CompileError
from codestr.syntax import Call, Column, ExprNode, KeywordArg
from codestr.syntax import Literal as LiteralNode

if TYPE_CHECKING:
    from collections.abc import Collection

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


@dataclass(frozen=True)
class PlanStep:
    node: ExprNode
    output_name: str
    internal: bool


@dataclass(frozen=True)
class ExecutionPlan:
    steps: tuple[PlanStep, ...]
    output_name: str

    @property
    def is_single_stage(self) -> bool:
        return len(self.steps) == 1


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


def _canonical_node(node: ExprNode):
    if isinstance(node, Column):
        return ["column", node.name]
    if isinstance(node, LiteralNode):
        value_type = f"{type(node.value).__module__}.{type(node.value).__qualname__}"
        return ["literal", value_type, repr(node.value)]
    if isinstance(node, Call):
        args = []
        for arg in node.args:
            if isinstance(arg, KeywordArg):
                args.append(["keyword", arg.name, _canonical_node(arg.value)])
            else:
                args.append(["positional", _canonical_node(arg)])
        return ["call", node.fn_name, args]
    raise TypeError(f"Unknown node type: {type(node)}")


def node_fingerprint(node: ExprNode) -> str:
    payload = json.dumps(
        _canonical_node(node),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class _LoweringContext:
    def __init__(
        self,
        registry: UDFRegistry,
        existing_columns: Collection[str],
    ):
        self.registry = registry
        self.used_names = set(existing_columns)
        self.steps: list[PlanStep] = []
        self.extracted: dict[ExprNode, str] = {}

    def _reserve_internal_name(self, node: ExprNode) -> str:
        base = f"__codestr_{node_fingerprint(node)[:16]}"
        candidate = base
        suffix = 1
        while candidate in self.used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self.used_names.add(candidate)
        return candidate

    def _rewrite_call(
        self,
        node: Call,
        active_domain: WindowDomain | None,
    ) -> Call:
        args = []
        for arg in node.args:
            if isinstance(arg, KeywordArg):
                args.append(
                    KeywordArg(
                        arg.name,
                        self.lower(arg.value, active_domain),
                    )
                )
            else:
                args.append(self.lower(arg, active_domain))
        return Call(node.fn_name, tuple(args), _alias=node.alias)

    def lower(
        self,
        node: ExprNode,
        active_domain: WindowDomain | None = None,
    ) -> ExprNode:
        if not isinstance(node, Call):
            return node

        # Unknown calls stay intact so compilation preserves its original error.
        if self.registry.get(node.fn_name) is None:
            return node

        domain = _call_domain(node, self.registry)
        if domain is not None and active_domain is not None and domain != active_domain:
            existing = self.extracted.get(node)
            if existing is not None:
                return Column(existing)

            rewritten = self._rewrite_call(node, domain)
            output_name = self._reserve_internal_name(node)
            self.steps.append(
                PlanStep(
                    node=rewritten,
                    output_name=output_name,
                    internal=True,
                )
            )
            self.extracted[node] = output_name
            return Column(output_name)

        next_domain = domain if domain is not None else active_domain
        return self._rewrite_call(node, next_domain)


def _column_names(node: ExprNode) -> set[str]:
    if isinstance(node, Column):
        return {node.name}
    if not isinstance(node, Call):
        return set()

    names = set()
    for arg in node.args:
        names.update(_column_names(_arg_value(arg)))
    return names


def _validate_step_dependencies(steps: tuple[PlanStep, ...]) -> None:
    internal_names = {step.output_name for step in steps if step.internal}
    available_internal = set()

    for step in steps:
        missing = (_column_names(step.node) & internal_names) - available_internal
        if missing:
            raise CompileError(f"Invalid plan dependency for {step.output_name}: {sorted(missing)}")
        if step.internal:
            available_internal.add(step.output_name)


def build_execution_plan(
    node: ExprNode,
    registry: UDFRegistry,
    existing_columns: Collection[str] = (),
) -> ExecutionPlan:
    context = _LoweringContext(registry, existing_columns)
    lowered = context.lower(node)
    steps = (
        *context.steps,
        PlanStep(
            node=lowered,
            output_name=node.alias,
            internal=False,
        ),
    )

    _validate_step_dependencies(steps)

    for step in steps:
        boundary = find_mixed_window_boundary(step.node, registry)
        if boundary is not None:
            raise CompileError(
                "Planner invariant failed; stage still contains mixed windows: "
                f"{boundary.outer_call} -> {boundary.inner_call}"
            )

    return ExecutionPlan(
        steps=steps,
        output_name=node.alias,
    )
