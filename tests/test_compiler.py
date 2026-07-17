"""Tests for AST → Polars Expr compiler (compiler.py)."""

import polars as pl
import pytest

from codestr.compiler import compile as ast_compile
from codestr.errors import CompileError
from codestr.parser import parse
from codestr.syntax import Call, Column, Literal


class TestCompileLeaf:
    def test_compile_column(self):
        expr = ast_compile(Column("close"))
        df = pl.DataFrame({"close": [100.0, 101.0]})
        result = df.select(expr)
        assert result.columns == ["close"]
        assert result["close"].to_list() == [100.0, 101.0]

    def test_compile_literal_int(self):
        expr = ast_compile(Literal(5))
        df = pl.DataFrame({"dummy": [1]})
        result = df.select(expr)
        assert result["5"].to_list() == [5]

    def test_compile_literal_float(self):
        expr = ast_compile(Literal(3.14))
        df = pl.DataFrame({"dummy": [1]})
        result = df.select(expr)
        assert result["3.14"].to_list() == [3.14]


class TestCompileCall:
    def test_compile_binary_add(self):
        node = Call(fn_name="add", args=(Column("a"), Column("b")))
        expr = ast_compile(node)
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
        result = df.select(expr)
        assert result.columns == ["(a+b)"]

    def test_compile_binary_sub(self):
        node = Call(fn_name="sub", args=(Column("a"), Column("b")))
        expr = ast_compile(node)
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
        result = df.select(expr)
        assert result["(a-b)"].to_list() == [-9.0, -18.0]

    def test_compile_binary_mul(self):
        node = Call(fn_name="mul", args=(Column("a"), Column("b")))
        expr = ast_compile(node)
        df = pl.DataFrame({"a": [3.0, 4.0], "b": [2.0, 5.0]})
        result = df.select(expr)
        assert result["(a*b)"].to_list() == [6.0, 20.0]

    def test_compile_binary_div(self):
        node = Call(fn_name="div", args=(Column("a"), Column("b")))
        expr = ast_compile(node)
        df = pl.DataFrame({"a": [10.0, 10.0], "b": [2.0, 5.0]})
        result = df.select(expr)
        assert result["(a/b)"].to_list() == [5.0, 2.0]

    def test_compile_comparison(self):
        node = Call(fn_name="gt", args=(Column("a"), Column("b")))
        expr = ast_compile(node)
        df = pl.DataFrame({"a": [5.0, 3.0], "b": [2.0, 5.0]})
        result = df.select(expr)
        assert result["(a>b)"].to_list() == [True, False]

    def test_compile_unary_neg(self):
        node = Call(fn_name="neg", args=(Column("a"),))
        expr = ast_compile(node)
        df = pl.DataFrame({"a": [1.0, -2.0]})
        result = df.select(expr)
        assert result["-a"].to_list() == [-1.0, 2.0]

    def test_compile_ternary_if(self):
        node = Call(fn_name="if_", args=(Column("cond"), Column("a"), Column("b")))
        expr = ast_compile(node)
        df = pl.DataFrame({"cond": [True, False], "a": [1.0, 2.0], "b": [10.0, 20.0]})
        result = df.select(expr)
        assert result["cond?a:b"].to_list() == [1.0, 20.0]

    def test_compile_abs(self):
        node = Call(fn_name="abs", args=(Column("x"),))
        expr = ast_compile(node)
        df = pl.DataFrame({"x": [-1.0, 2.0, -3.0]})
        result = df.select(expr)
        assert result["abs(x)"].to_list() == [1.0, 2.0, 3.0]

    def test_compile_with_literal_args(self):
        node = Call(fn_name="add", args=(Column("a"), Literal(10)))
        expr = ast_compile(node)
        df = pl.DataFrame({"a": [1.0, 2.0]})
        result = df.select(expr)
        assert result["(a+10)"].to_list() == [11.0, 12.0]

    def test_compile_nested_call(self):
        """abs(a - b)"""
        inner = Call(fn_name="sub", args=(Column("a"), Column("b")))
        node = Call(fn_name="abs", args=(inner,))
        expr = ast_compile(node)
        df = pl.DataFrame({"a": [10.0, 5.0], "b": [5.0, 10.0]})
        result = df.select(expr)
        assert result["abs((a-b))"].to_list() == [5.0, 5.0]

    def test_compile_keyword_argument(self):
        df = pl.DataFrame({"close": [-1.0, 2.0]})
        result = df.select(ast_compile(parse("clip(close, lower_bound=0)")))

        assert result["clip(close, lower_bound=0)"].to_list() == [0.0, 2.0]


class TestCompileErrors:
    def test_unknown_function(self):
        node = Call(fn_name="nonexistent_fn", args=(Column("x"),))
        with pytest.raises(CompileError, match="Unknown function"):
            ast_compile(node)

    def test_wrong_node_type(self):
        with pytest.raises(TypeError):
            ast_compile(None)  # type: ignore


class TestMixedWindowCompile:
    @pytest.mark.parametrize(
        ("source", "path"),
        [
            ("ts_mean(cs_moderate(x), 60)", "ts_mean -> cs_moderate"),
            ("cs_mean(ts_mean(x, 60))", "cs_mean -> ts_mean"),
        ],
    )
    def test_pure_compile_rejects_mixed_window_domains(self, source, path):
        with pytest.raises(CompileError, match=path):
            ast_compile(parse(source))

    @pytest.mark.parametrize(
        "source",
        [
            "ts_mean(ts_mean(x, 2, min_samples=1), 2, min_samples=1)",
            "cs_rank(cs_moderate(x))",
        ],
    )
    def test_pure_compile_keeps_same_domain_nesting(self, source):
        assert isinstance(ast_compile(parse(source)), pl.Expr)

    def test_unknown_function_remains_the_primary_error(self):
        source = "ts_mean(not_registered(cs_moderate(x)), 60)"

        with pytest.raises(CompileError, match="Unknown function: not_registered"):
            ast_compile(parse(source))
