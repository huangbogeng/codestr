# -*- coding: utf-8 -*-

from __future__ import annotations

from loguru import logger

import polars as pl

from codestr.compiler import compile as _pure_compile
from codestr.errors import CalculateError, CompileError, PolarsError, FailError
from codestr.syntax import Call, depth as _depth, node_count as _node_count, to_rpn as _to_rpn
from codestr.parser import parse as _parse
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
        index: tuple[str] = ("date", "time", "datetime", "asset"),
        align: bool = True,
        pure_lazy: bool = False,
    ):
        assert isinstance(data, (pl.LazyFrame, pl.DataFrame)), (
            "data 必须是 polars DataFrame 或 LazyFrame"
        )
        self.failed: list = []
        self._expr_cache: dict = {}
        self._cur_expr_cache: dict = {}

        self.data: pl.DataFrame | None = None
        self.index = index
        self._data_: pl.LazyFrame | None = None
        self._last_query_cache: pl.DataFrame | None = None

        if pure_lazy:
            self._data_ = data
        else:
            self.data = data.with_columns(pl.col(pl.Decimal).cast(float))
            if isinstance(self.data, pl.LazyFrame):
                self.data = self.data.collect()

            if align:
                self.align()

    def align(self, on=("datetime", "asset")):
        """数据对齐"""
        lev_vals: list[pl.DataFrame] = [
            self.data.select(name).drop_nulls().unique() for name in on
        ]
        full_index = lev_vals[0].unique()
        for lev_val in lev_vals[1:]:
            full_index = full_index.join(lev_val.unique(), how="cross")
        self.data = full_index.join(self.data, on=on, how="left").sort(
            self.index
        )

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

    def register_udf(self, func: callable, name: str = None):
        """Register a user-defined function into the UDF registry."""
        from codestr.udf.registry import UDFMeta, UDFRegistry

        UDFRegistry.get_instance().register(UDFMeta(
            name=name if name is not None else func.__name__,
            fn=func,
            category="user",
        ))

    def check_expr(
        self,
        expr: str,
        max_depth: int | None = None,
        max_nodes: int | None = None,
        check_rpn: bool = True,
        check_redundant: bool = True,
    ):
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
            left, right = node.args
            if str(left) == str(right):
                reasons.append(f"redundant:{node.fn_name}")
        for arg in node.args:
            if isinstance(arg, Call):
                self._check_redundant(arg, reasons)

    def _check_rpn(self, rpn, reasons: list[str]):
        from codestr.tokens import TokenType

        stack = 0
        for token in rpn:
            if token.type in (TokenType.FEATURE, TokenType.CONSTANT, TokenType.WINDOW, TokenType.BINS, TokenType.PARAM):
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
        """
        Purely compile an expression string to a Polars Expression without side effects.
        Used for batch evaluation.
        """
        try:
            node = _parse(expr)
            return _pure_compile(node, registry=UDFRegistry.get_instance(), dims=getattr(self, "dims", None))
        except Exception as e:
            raise CompileError(f"Pure compilation failed for {expr}: {e}") from e

    def _compile_expr(self, expr: str, cover: bool):
        """str表达式 -> polars 表达式"""
        if self._data_ is None:
            self._data_ = self.data.lazy()

        try:
            node = _parse(expr)
            alias = node.alias
            current_cols = set(self.cache_columns)

            if alias in current_cols and not cover:
                return pl.col(alias), alias
            if node in self._expr_cache and not cover:
                expr_pl = pl.col(self._expr_cache[node]).alias(alias)
                self._data_ = self._data_.with_columns(expr_pl)
                return pl.col(alias), alias
            if node in self._cur_expr_cache and not cover:
                expr_pl = pl.col(self._cur_expr_cache[node]).alias(alias)
                self._data_ = self._data_.with_columns(expr_pl)
                return pl.col(alias), alias

            expr_pl = _pure_compile(node, registry=UDFRegistry.get_instance(), dims=getattr(self, "dims", None))
            self._data_ = self._data_.with_columns(expr_pl.alias(alias))
            self._cur_expr_cache[node] = alias
            return pl.col(alias), alias

        except Exception as e:
            raise CompileError(message=f"[表达式]: {expr}\n[编译器外层]\n{e}") from e

    def sql(
        self,
        *exprs: str,
        cover: bool = False,
        lazy: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        表达式查询
        Parameters
        ----------
        exprs: str
            表达式，比如 "ts_mean(close, 5) as close_ma5"
        cover: bool
            当遇到已经存在列名的时候，是否重新计算覆盖原来的列, 默认False，返回已经存在的列，跳过计算
            - True: 重新计算并且返回新的结果，覆盖掉原来的列
            - False, 返回已经存在的列，跳过计算
        lazy: bool
            是否返回 LazyFrame, 默认 False (返回 DataFrame)
            - True: 返回 polars.LazyFrame，不进行计算 (collect)
            - False: 返回 polars.DataFrame，立即计算
        Returns
        -------
            polars.DataFrame | polars.LazyFrame
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
                        *[
                            i
                            for i in self._last_query_cache.columns
                            if i not in self.data.columns
                        ]
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
            logger.warning(
                f"CodeStr.sql 失败：{len(self.failed)}/{len(exprs)}: \n {self.failed}"
            )
        if self._data_ is None:
            self._data_ = self.data.lazy()
        self._data_ = self._data_.with_columns(*exprs_to_add)

        if lazy:
            self._expr_cache.update(self._cur_expr_cache)
            result = self._data_.select(*self.index, *exprs_select)
            self._data_ = _data_saved   # roll back: don't accumulate in lazy mode
            return result

        current_cols = set(self._data_.collect_schema().names())
        new_expr_cache = dict()
        try:
            self._last_query_cache = self._data_.select(
                *self.index, *exprs_select
            ).collect()
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
            self._data_ = _data_saved   # roll back failed with_columns
            raise PolarsError(message=f"LazyFrame.collect() 阶段出错\n{e}") from e

    def clear_cache(self):
        """清除缓存，重置计算图"""
        self._data_ = None
        self._expr_cache = {}
        self._cur_expr_cache = {}
        self._last_query_cache = None
