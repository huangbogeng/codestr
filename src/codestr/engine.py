from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

from codestr.compiler import compile as _pure_compile
from codestr.errors import CompileError, FailError, PolarsError
from codestr.parser import parse as _parse
from codestr.planner import build_execution_plan
from codestr.syntax import (
    Call,
    KeywordArg,
)
from codestr.syntax import (
    depth as _depth,
)
from codestr.syntax import (
    node_count as _node_count,
)
from codestr.syntax import (
    to_rpn as _to_rpn,
)
from codestr.udf.registry import UDFRegistry


class CodeStr:
    """Expression compute engine with DSL → Polars translation.

    State invariants
    ----------------
    - ``data`` : the most recently materialised DataFrame.  Only ``None`` before
      the first ``sql()`` call on a ``pure_lazy`` engine.
    - ``_data_`` : lazy compute graph that accumulates ``with_columns`` during
      a single ``sql()`` call.  Reset to ``None`` on ``clear_cache()``.
    - ``_last_query_cache`` : result of the last eager ``sql()``, used to merge
      new columns back into ``data`` on the next call.
    - ``_expr_cache`` : persistent cross-query cache (ExprNode → alias).
    - ``_cur_expr_cache`` : per-query cache, merged into ``_expr_cache`` after
      a successful eager ``sql()``.
    """

    def __init__(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        index: tuple[str, str] = ("datetime", "asset"),
        partition_by: list[str] | None = None,
        order_by: list[str] | None = None,
        align: bool = True,
        pure_lazy: bool = False,
    ):
        """Initialize the CodeStr engine.

        Args:
            data: Input Polars DataFrame or LazyFrame.
            index: A 2-tuple ``(time_col, entity_col)`` used for panel alignment
                   and result column selection.
            partition_by: Entity-axis columns for window grouping.
                          Defaults to ``[index[1]]`` (e.g. ``["asset"]``).
            order_by: Time-axis columns for window ordering.
                      Defaults to ``[index[0]]`` (e.g. ``["datetime"]``).
            align: If True (default), perform cross-join alignment to fill
                   missing index combinations with nulls.
            pure_lazy: If True, never materialize data — keep everything as
                       LazyFrame. ``sql()`` calls will not update ``self.data``.

        Window Semantics
        ----------------
        * **TS (time-series)** — ``over(partition_by=partition_by, order_by=order_by)``.
          Each entity's rolling window runs independently along the time axis.
        * **CS (cross-section)** — ``over(partition_by=order_by, order_by=partition_by)``.
          At each time slice, operators compute across all entities.
        """
        assert isinstance(data, (pl.LazyFrame, pl.DataFrame)), (
            "data 必须是 polars DataFrame 或 LazyFrame"
        )
        self.failed: list = []
        self._expr_cache: dict = {}
        self._cur_expr_cache: dict = {}

        self.data: pl.DataFrame | None = None
        self.index: tuple[str, str] = index
        self._data_: pl.LazyFrame | None = None
        self._last_query_cache: pl.DataFrame | None = None

        # Over-window config: user provides partition / order columns explicitly
        _partition = partition_by if partition_by is not None else [self.index[1]]
        _order = order_by if order_by is not None else [self.index[0]]

        self._ts_over = {
            "partition_by": _partition,  # entity columns
            "order_by": _order,  # time columns
        }
        self._cs_over = {
            "partition_by": _order,  # time columns  (swapped)
            "order_by": _partition,  # entity columns (swapped)
        }

        if pure_lazy:
            self._data_ = data
        else:
            self.data = data.with_columns(pl.col(pl.Decimal).cast(float))
            if isinstance(self.data, pl.LazyFrame):
                self.data = self.data.collect()

            if align:
                self.align()

    def align(self, on=None):
        """数据对齐

        Args:
            on: 2-tuple of columns to align on.  Defaults to ``self.index``.
        """
        if on is None:
            on = (self.index[0], self.index[1])
        lev_vals: list[pl.DataFrame] = [self.data.select(name).drop_nulls().unique() for name in on]
        full_index = lev_vals[0].unique()
        for lev_val in lev_vals[1:]:
            full_index = full_index.join(lev_val.unique(), how="cross")
        self.data = full_index.join(self.data, on=on, how="left").sort(self.index)

        self.dims = [self.data[name].drop_nulls().n_unique() for name in on]

    @property
    def cache_columns(self) -> list[str]:
        """Currently available column names (from materialised data or lazy graph)."""
        if self.data is not None:
            return self.data.columns
        if self._data_ is not None:
            return self._data_.collect_schema().names()
        return []

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def register_udf(self, func: Callable, name: str | None = None):
        """Register a user-defined function into the UDF registry."""
        from codestr.udf.registry import UDFMeta, UDFRegistry

        UDFRegistry.get_instance().register(
            UDFMeta(
                name=name if name is not None else func.__name__,
                fn=func,
                category="user",
            )
        )

    def check_expr(
        self,
        expr: str,
        max_depth: int | None = None,
        max_nodes: int | None = None,
        check_rpn: bool = True,
        check_redundant: bool = True,
    ):
        """Validate an expression string before execution.

        Does NOT execute the expression — only parses and runs structural checks.

        Args:
            expr: The DSL expression string to validate.
            max_depth: If set, reject expressions exceeding this AST depth.
            max_nodes: If set, reject expressions exceeding this node count.
            check_rpn: Validate reverse Polish notation stack balance.
            check_redundant: Detect redundant sub-expressions (e.g. ``a - a``).

        Returns:
            A dict with keys ``valid`` (bool), ``reasons`` (list[str]), and
            ``expr`` (str representation of the parsed node, or None on error).
        """
        result = {"valid": True, "reasons": [], "expr": None}
        try:
            node = _parse(expr)
            result["expr"] = str(node)
            if max_depth is not None and _depth(node) > max_depth:
                result["reasons"].append(f"max_depth:{_depth(node)}")
            if max_nodes is not None and _node_count(node) > max_nodes:
                result["reasons"].append(f"max_nodes:{_node_count(node)}")
            if check_redundant:
                self._check_redundant(node, result["reasons"])
            if check_rpn:
                self._check_rpn(_to_rpn(node), result["reasons"])
        except Exception as e:
            result["reasons"].append(str(e))
        result["valid"] = len(result["reasons"]) == 0
        return result

    def _check_redundant(self, node: Call, reasons: list[str]):
        if not isinstance(node, Call):
            return
        if node.fn_name in ("sub", "div") and len(node.args) == 2:
            left, right = (arg.value if isinstance(arg, KeywordArg) else arg for arg in node.args)
            if str(left) == str(right):
                reasons.append(f"redundant:{node.fn_name}")
        for arg in node.args:
            value = arg.value if isinstance(arg, KeywordArg) else arg
            if isinstance(value, Call):
                self._check_redundant(value, reasons)

    def _check_rpn(self, rpn, reasons: list[str]):
        from codestr.tokens import TokenType

        stack = 0
        for token in rpn:
            if token.type in (
                TokenType.FEATURE,
                TokenType.CONSTANT,
                TokenType.WINDOW,
                TokenType.BINS,
                TokenType.PARAM,
            ):
                stack += 1
            elif token.type == TokenType.OPERATOR:
                args_num = token.arity
                if stack < args_num:
                    reasons.append(f"rpn_args:{token.value}")
                    return
                stack = stack - args_num + 1
        if stack != 1:
            reasons.append("rpn_invalid")

    def compile(self, expr: str) -> pl.Expr:
        """Purely compile an expression string to a Polars Expression.

        No side effects.  The returned expression is bound to the column names
        and over-window config of this CodeStr instance.
        """
        try:
            node = _parse(expr)
            return _pure_compile(
                node,
                registry=UDFRegistry.get_instance(),
                dims=getattr(self, "dims", None),
                ts_over=self._ts_over,
                cs_over=self._cs_over,
            )
        except Exception as e:
            raise CompileError(f"Pure compilation failed for {expr}: {e}") from e

    def _compile_expr(self, expr: str, cover: bool):
        """Parse, plan, and append one expression to the current lazy graph."""
        if self._data_ is None:
            self._data_ = self.data.lazy()

        data_saved = self._data_
        cache_saved = dict(self._cur_expr_cache)

        try:
            node = _parse(expr)
            alias = node.alias
            current_cols = set(self._data_.collect_schema().names())

            if alias in current_cols and not cover:
                return pl.col(alias), alias
            if node in self._expr_cache and not cover:
                cached_alias = self._expr_cache[node]
                if cached_alias in current_cols:
                    expr_pl = pl.col(cached_alias).alias(alias)
                    self._data_ = self._data_.with_columns(expr_pl)
                    return pl.col(alias), alias
            if node in self._cur_expr_cache and not cover:
                cached_alias = self._cur_expr_cache[node]
                if cached_alias in current_cols:
                    expr_pl = pl.col(cached_alias).alias(alias)
                    self._data_ = self._data_.with_columns(expr_pl)
                    return pl.col(alias), alias

            registry = UDFRegistry.get_instance()
            plan = build_execution_plan(
                node,
                registry,
                existing_columns=current_cols,
            )

            if plan.is_single_stage:
                expr_pl = _pure_compile(
                    node,
                    registry=registry,
                    dims=getattr(self, "dims", None),
                    ts_over=self._ts_over,
                    cs_over=self._cs_over,
                )
                self._data_ = self._data_.with_columns(expr_pl.alias(alias))
            else:
                for step in plan.steps:
                    expr_pl = _pure_compile(
                        step.node,
                        registry=registry,
                        dims=getattr(self, "dims", None),
                        ts_over=self._ts_over,
                        cs_over=self._cs_over,
                    )
                    self._data_ = self._data_.with_columns(expr_pl.alias(step.output_name))

            self._cur_expr_cache[node] = alias
            return pl.col(alias), alias

        except Exception as e:
            self._data_ = data_saved
            self._cur_expr_cache = cache_saved
            raise CompileError(message=f"[表达式]: {expr}\n[编译器外层]\n{e}") from e

    def sql(
        self,
        *exprs: str,
        cover: bool = False,
        lazy: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        """Execute one or more DSL expressions interactively.

        This is the **stateful** API — results are cached in the engine and
        may be reused across subsequent ``sql()`` calls.

        Args:
            exprs: DSL expression strings, e.g. ``"ts_mean(close, 5) as ma5"``.
            cover: If True, recompute even if the alias already exists.
                   If False (default), skip computation on cache hits.
            lazy: If True, return a ``pl.LazyFrame`` without materializing.
                  If False (default), collect and return a ``pl.DataFrame``.

        Returns:
            A DataFrame or LazyFrame containing the index columns and all
            requested expression aliases.
        """
        self.failed = list()
        exprs_to_add = list()
        exprs_select = list()
        self._cur_expr_cache = {}

        # Snapshot _data_ so we can roll back on lazy-return or failure
        _data_saved = self._data_

        if self._last_query_cache is not None:
            if self.data is None:
                self.data = self._last_query_cache
            else:
                self.data = self.data.with_columns(
                    self._last_query_cache.select(
                        *[i for i in self._last_query_cache.columns if i not in self.data.columns]
                    )
                )

        for expr in exprs:
            try:
                compiled, alias = self._compile_expr(expr, cover)
                if compiled is not None:
                    exprs_to_add.append(compiled)
                    exprs_select.append(alias)
            except Exception as e:
                self.failed.append(FailError(expr, e))
        if self.failed:
            logger.warning(f"CodeStr.sql 失败：{len(self.failed)}/{len(exprs)}: \n {self.failed}")
        if self._data_ is None:
            self._data_ = self.data.lazy()
        self._data_ = self._data_.with_columns(*exprs_to_add)

        if lazy:
            self._expr_cache.update(self._cur_expr_cache)
            result = self._data_.select(*self.index, *exprs_select)
            self._data_ = _data_saved  # roll back: don't accumulate in lazy mode
            return result

        current_cols = set(self._data_.collect_schema().names())
        new_expr_cache = dict()
        try:
            self._last_query_cache = self._data_.select(*self.index, *exprs_select).collect()
            self._expr_cache.update(self._cur_expr_cache)
            for k, v in self._expr_cache.items():
                if v in current_cols:
                    new_expr_cache[k] = v
            self._expr_cache = new_expr_cache

            return self._last_query_cache
        except Exception as e:
            for k, v in self._expr_cache.items():
                if v in current_cols:
                    new_expr_cache[k] = v
            self._expr_cache = new_expr_cache
            self._data_ = _data_saved  # roll back failed with_columns
            raise PolarsError(message=f"LazyFrame.collect() 阶段出错\n{e}") from e

    def clear_cache(self):
        """清除缓存，重置计算图"""
        self._data_ = None
        self._expr_cache = {}
        self._cur_expr_cache = {}
        self._last_query_cache = None
