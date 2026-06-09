from __future__ import annotations

import inspect

import polars as pl

from codestr.syntax import Call, Column, ExprNode, Literal
from codestr.errors import CompileError
from codestr.udf.registry import UDFRegistry


def compile(
    node: ExprNode,
    registry: UDFRegistry | None = None,
    dims: list[int] | None = None,
) -> pl.Expr:
    """Pure: compile an AST to a Polars expression.  No side effects."""
    if registry is None:
        registry = UDFRegistry.get_instance()
    return _compile(node, registry, dims).alias(node.alias)


def _resolve(node: ExprNode, registry: UDFRegistry, dims: list[int] | None) -> pl.Expr | int | float | str:
    """Compile an AST node to a Polars expression or resolve to a Python scalar."""
    if isinstance(node, Column):
        return pl.col(node.name)
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, Call):
        return _compile(node, registry, dims)
    raise TypeError(f"Unknown node type: {type(node)}")


def _compile(
    node: ExprNode,
    registry: UDFRegistry,
    dims: list[int] | None = None,
) -> pl.Expr:
    from toolz import partial

    if isinstance(node, Column):
        return pl.col(node.name)

    if isinstance(node, Literal):
        return pl.lit(node.value)

    if isinstance(node, Call):
        if node.fn_name not in registry:
            raise CompileError(f"Unknown function: {node.fn_name}")

        meta = registry[node.fn_name]
        func = meta.fn

        sig_params = list(inspect.signature(func).parameters.keys())
        if "dims" in sig_params and dims is not None:
            func = partial(func, dims=dims)

        args: list = []
        kwargs: dict = {}

        for arg in node.args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    kwargs[k] = _resolve(v, registry, dims) if isinstance(v, ExprNode) else v
            else:
                args.append(_resolve(arg, registry, dims))

        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise CompileError(
                f"{node.fn_name}({', '.join(str(a) for a in node.args)})\n{e}"
            ) from e

    raise TypeError(f"Unknown node type: {type(node)}")
