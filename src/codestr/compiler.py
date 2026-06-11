from __future__ import annotations

import inspect

import polars as pl

from codestr.errors import CompileError
from codestr.syntax import Call, Column, ExprNode, Literal
from codestr.udf.registry import UDFRegistry

# Canonical window defaults for when the caller does not provide ts_over/cs_over.
# These match the default time_col="datetime", asset_col="asset" convention.
# When called through the CodeStr engine, these are ALWAYS overridden by the
# engine's per-instance _ts_over / _cs_over (derived from time_col/asset_col).
_TS_DEFAULT_OVER = {"partition_by": ["asset"], "order_by": ["datetime"]}
_CS_DEFAULT_OVER = {"partition_by": ["datetime"], "order_by": ["asset"]}


def compile(
    node: ExprNode,
    registry: UDFRegistry | None = None,
    dims: list[int] | None = None,
    ts_over: dict[str, list[str]] | None = None,
    cs_over: dict[str, list[str]] | None = None,
) -> pl.Expr:
    """Compile an AST node to a Polars expression.

    Pure function — no side effects, no state mutation.

    Args:
        node: The root AST node to compile.
        registry: UDF registry for function lookup. Uses the global singleton by default.
        dims: Dimension info (e.g. [num_datetimes, num_assets]) for operators that need
              array reshaping context.
        ts_over: Window config for time-series operators
                 (``{"partition_by": [...], "order_by": [...]}``).
        cs_over: Window config for cross-section operators.

    Returns:
        A Polars expression with the node's alias applied.
    """
    if registry is None:
        registry = UDFRegistry.get_instance()
    return _compile(node, registry, dims, ts_over, cs_over).alias(node.alias)


def _resolve(
    node: ExprNode,
    registry: UDFRegistry,
    dims: list[int] | None,
    ts_over: dict[str, list[str]] | None = None,
    cs_over: dict[str, list[str]] | None = None,
) -> pl.Expr | int | float | str:
    """Compile an AST node to a Polars expression or resolve to a Python scalar.

    Literal nodes return bare Python values (int/float), Column and Call nodes
    return pl.Expr. This is used so that UDF functions receive appropriate types
    for positional and keyword arguments.
    """
    if isinstance(node, Column):
        return pl.col(node.name)
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, Call):
        return _compile(node, registry, dims, ts_over, cs_over)
    raise TypeError(f"Unknown node type: {type(node)}")


def _compile(
    node: ExprNode,
    registry: UDFRegistry,
    dims: list[int] | None = None,
    ts_over: dict[str, list[str]] | None = None,
    cs_over: dict[str, list[str]] | None = None,
) -> pl.Expr:
    """Recursively compile an AST node to a Polars expression.

    Column → pl.col, Literal → pl.lit, Call → UDF invocation.

    Engine context is injected automatically by inspecting the UDF signature:
    * ``dims`` — injected if the function accepts it and dims is available.
    * ``partition_by`` / ``order_by`` — injected based on the UDF category
      (``"ts"`` → ts_over, ``"cs"`` → cs_over), only if the function
      signature includes those parameter names.
    """
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

        args: list = []
        kwargs: dict = {}

        sig_params = list(inspect.signature(func).parameters.keys())

        # Inject dims if the operator accepts it
        if "dims" in sig_params and dims is not None:
            func = partial(func, dims=dims)

        # Inject over config based on UDF category
        _inject_over(meta.category, sig_params, ts_over, cs_over, kwargs)

        for arg in node.args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    kwargs[k] = (
                        _resolve(v, registry, dims, ts_over, cs_over)
                        if isinstance(v, ExprNode)
                        else v
                    )
            else:
                args.append(_resolve(arg, registry, dims, ts_over, cs_over))

        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise CompileError(
                f"{node.fn_name}({', '.join(str(a) for a in node.args)})\n{e}"
            ) from e

    raise TypeError(f"Unknown node type: {type(node)}")


def _inject_over(
    category: str,
    sig_params: list[str],
    ts_over: dict[str, list[str]] | None,
    cs_over: dict[str, list[str]] | None,
    kwargs: dict,
) -> None:
    """Inject partition_by/order_by into kwargs based on operator category.

    TS/CS operators ALWAYS receive window config. The priority is:
    1. Caller-provided ts_over/cs_over (from the engine)
    2. Module-level canonical defaults (_TS_DEFAULT_OVER / _CS_DEFAULT_OVER)

    Math/user operators have no window concept and are skipped.
    """
    if category == "ts":
        over_config = ts_over if ts_over is not None else _TS_DEFAULT_OVER
    elif category == "cs":
        over_config = cs_over if cs_over is not None else _CS_DEFAULT_OVER
    else:
        return  # "math" and "user" have no window config

    if "partition_by" in sig_params:
        kwargs["partition_by"] = over_config["partition_by"]
    if "order_by" in sig_params:
        kwargs["order_by"] = over_config["order_by"]
