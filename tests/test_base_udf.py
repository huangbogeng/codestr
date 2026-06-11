"""Tests for base UDF operators (udf/base_udf.py)."""

import numpy as np
import polars as pl
import pytest

from codestr.compiler import compile as ast_compile
from codestr.syntax import Call, Column, Literal


def _eval_expr(node, df: pl.DataFrame) -> pl.DataFrame:
    expr = ast_compile(node)
    return df.select(expr)


def series_equal(result_col: pl.Series, expected: list) -> bool:
    return result_col.to_list() == expected


class TestUnaryMath:
    def test_abs(self):
        df = pl.DataFrame({"x": [-1.0, 0.0, 2.0]})
        node = Call("abs", (Column("x"),))
        assert df.select(ast_compile(node))["abs(x)"].to_list() == [1.0, 0.0, 2.0]

    def test_sqrt(self):
        df = pl.DataFrame({"x": [4.0, 9.0, 16.0]})
        node = Call("sqrt", (Column("x"),))
        result = df.select(ast_compile(node))["sqrt(x)"].to_list()
        assert pytest.approx(result) == [2.0, 3.0, 4.0]

    def test_square(self):
        df = pl.DataFrame({"x": [1.0, 2.0, -3.0]})
        node = Call("square", (Column("x"),))
        assert df.select(ast_compile(node))["square(x)"].to_list() == [1.0, 4.0, 9.0]

    def test_cube(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        node = Call("cube", (Column("x"),))
        assert df.select(ast_compile(node))["cube(x)"].to_list() == [1.0, 8.0, 27.0]

    def test_log(self):
        df = pl.DataFrame({"x": [1.0, np.e, np.e**2]})
        node = Call("log", (Column("x"),))
        result = df.select(ast_compile(node))["log(x)"].to_list()
        assert pytest.approx(result[1]) == 1.0
        assert pytest.approx(result[2]) == 2.0

    def test_sin(self):
        df = pl.DataFrame({"x": [0.0]})
        node = Call("sin", (Column("x"),))
        assert pytest.approx(df.select(ast_compile(node))["sin(x)"][0]) == 0.0

    def test_cos(self):
        df = pl.DataFrame({"x": [0.0]})
        node = Call("cos", (Column("x"),))
        assert pytest.approx(df.select(ast_compile(node))["cos(x)"][0]) == 1.0

    def test_exp(self):
        df = pl.DataFrame({"x": [0.0, 1.0]})
        node = Call("exp", (Column("x"),))
        result = df.select(ast_compile(node))["exp(x)"].to_list()
        assert pytest.approx(result) == [1.0, np.e]

    def test_sign(self):
        df = pl.DataFrame({"x": [-5.0, 0.0, 3.0]})
        node = Call("sign", (Column("x"),))
        assert df.select(ast_compile(node))["sign(x)"].to_list() == [-1.0, 0.0, 1.0]

    def test_sigmoid(self):
        df = pl.DataFrame({"x": [0.0]})
        node = Call("sigmoid", (Column("x"),))
        result = df.select(ast_compile(node))["sigmoid(x)"][0]
        assert pytest.approx(result) == 0.5

    def test_clip(self):
        df = pl.DataFrame({"x": [-10.0, 0.0, 10.0]})
        node = Call("clip", (Column("x"), Literal(-1), Literal(1)))
        assert df.select(ast_compile(node))["clip(x, -1, 1)"].to_list() == [-1.0, 0.0, 1.0]

    def test_between(self):
        df = pl.DataFrame({"x": [0.0, 5.0, 10.0]})
        node = Call("between", (Column("x"), Literal(2), Literal(8)))
        result = df.select(ast_compile(node))["between(x, 2, 8)"].to_list()
        assert result == [False, True, False]


class TestBinaryMath:
    def test_add(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
        node = Call("add", (Column("a"), Column("b")))
        assert df.select(ast_compile(node))["(a+b)"].to_list() == [11.0, 22.0]

    def test_sub(self):
        df = pl.DataFrame({"a": [10.0, 20.0], "b": [3.0, 5.0]})
        node = Call("sub", (Column("a"), Column("b")))
        assert df.select(ast_compile(node))["(a-b)"].to_list() == [7.0, 15.0]

    def test_mul(self):
        df = pl.DataFrame({"a": [3.0, 4.0], "b": [2.0, 5.0]})
        node = Call("mul", (Column("a"), Column("b")))
        assert df.select(ast_compile(node))["(a*b)"].to_list() == [6.0, 20.0]

    def test_div(self):
        df = pl.DataFrame({"a": [10.0, 20.0], "b": [2.0, 4.0]})
        node = Call("div", (Column("a"), Column("b")))
        assert df.select(ast_compile(node))["(a/b)"].to_list() == [5.0, 5.0]

    def test_gt(self):
        df = pl.DataFrame({"a": [5.0, 2.0], "b": [2.0, 5.0]})
        node = Call("gt", (Column("a"), Column("b")))
        assert df.select(ast_compile(node))["(a>b)"].to_list() == [True, False]

    def test_eq(self):
        df = pl.DataFrame({"a": [5.0, 3.0], "b": [5.0, 4.0]})
        node = Call("eq", (Column("a"), Column("b")))
        assert df.select(ast_compile(node))["(a==b)"].to_list() == [True, False]


class TestHorizontalOps:
    def test_max(self):
        df = pl.DataFrame({"a": [1.0, 5.0], "b": [3.0, 2.0]})
        node = Call("max", (Column("a"), Column("b")))
        assert df.select(ast_compile(node))["max(a, b)"].to_list() == [3.0, 5.0]

    def test_min(self):
        df = pl.DataFrame({"a": [1.0, 5.0], "b": [3.0, 2.0]})
        node = Call("min", (Column("a"), Column("b")))
        assert df.select(ast_compile(node))["min(a, b)"].to_list() == [1.0, 2.0]

    def test_sum_h(self):
        df = pl.DataFrame({"a": [1.0, 5.0], "b": [3.0, 2.0]})
        node = Call("sum", (Column("a"), Column("b")))
        assert df.select(ast_compile(node))["sum(a, b)"].to_list() == [4.0, 7.0]

    def test_mean_h(self):
        df = pl.DataFrame({"a": [1.0, 5.0], "b": [3.0, 3.0]})
        node = Call("mean", (Column("a"), Column("b")))
        assert df.select(ast_compile(node))["mean(a, b)"].to_list() == [2.0, 4.0]


class TestTernary:
    def test_if(self):
        df = pl.DataFrame(
            {
                "cond": [True, False, True],
                "a": [1.0, 2.0, 3.0],
                "b": [10.0, 20.0, 30.0],
            }
        )
        node = Call("if_", (Column("cond"), Column("a"), Column("b")))
        assert df.select(ast_compile(node))["cond?a:b"].to_list() == [1.0, 20.0, 3.0]

    def test_fib(self):
        df = pl.DataFrame({"high": [200.0, 100.0], "low": [100.0, 50.0]})
        node = Call("fib", (Column("high"), Column("low")))
        result = df.select(ast_compile(node))["fib(high, low)"].to_list()
        # low + (high - low) * 0.618
        assert pytest.approx(result) == [161.8, 80.9]


class TestLogicalOps:
    def test_and(self):
        df = pl.DataFrame({"a": [True, True, False], "b": [True, False, True]})
        node = Call("and_", (Column("a"), Column("b")))
        assert df.select(ast_compile(node))["(a&b)"].to_list() == [True, False, False]

    def test_or(self):
        df = pl.DataFrame({"a": [True, True, False], "b": [True, False, False]})
        node = Call("or_", (Column("a"), Column("b")))
        assert df.select(ast_compile(node))["(a|b)"].to_list() == [True, True, False]
