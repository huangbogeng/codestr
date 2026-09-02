"""Tests for base UDF operators (udf/base_udf.py)."""

import numpy as np
import polars as pl
import pytest

from codestr.compiler import compile as ast_compile
from codestr.syntax import Call, Column, Literal
from codestr.udf import base_udf


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

    @pytest.mark.parametrize(
        ("name", "value", "expected"),
        [
            ("cbrt", 8.0, 2.0),
            ("sinh", 0.0, 0.0),
            ("arcsin", 0.0, 0.0),
            ("arcsinh", 0.0, 0.0),
            ("cosh", 0.0, 1.0),
            ("arccos", 1.0, 0.0),
            ("arccosh", 1.0, 0.0),
            ("tan", 0.0, 0.0),
            ("tanh", 0.0, 0.0),
            ("arctan", 0.0, 0.0),
            ("arctanh", 0.0, 0.0),
            ("cot", np.pi / 4, 1.0),
            ("degrees", np.pi, 180.0),
            ("log1p", np.e - 1, 1.0),
        ],
    )
    def test_remaining_unary_numeric_contracts(self, name, value, expected):
        df = pl.DataFrame({"x": [value]})
        result = df.select(ast_compile(Call(name, (Column("x"),))))

        assert result.item() == pytest.approx(expected)

    def test_not(self):
        df = pl.DataFrame({"x": [True, False]})

        result = df.select(ast_compile(Call("not_", (Column("x"),))))

        assert result.to_series().to_list() == [False, True]

    def test_entropy(self):
        df = pl.DataFrame({"x": [0.25, 0.75]})

        result = df.select(ast_compile(Call("entropy", (Column("x"),))))

        assert result.item() == pytest.approx(0.5623351446)

    def test_trunc_closed_and_open_bounds(self):
        df = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
        closed = base_udf.trunc(pl.col("x"), 0, 2)
        opened = base_udf.trunc(pl.col("x"), 0, 2, left_closed=False, right_closed=False)

        result = df.select(closed.alias("closed"), opened.alias("opened"))

        assert result["closed"].to_list() == [0.0, 1.0, 2.0]
        assert result["opened"].to_list() == [None, 1.0, None]

    def test_cast_valid_and_invalid_dtype(self):
        df = pl.DataFrame({"x": [1.5]})

        result = df.select(base_udf.cast(pl.col("x"), "int"))

        assert result.schema["x"] == pl.Int64
        with pytest.raises(ValueError, match="not a valid type"):
            base_udf.cast(pl.col("x"), "date")

    def test_concat_and_null_type(self):
        df = pl.DataFrame({"a": [1], "b": [2]})

        result = df.select(
            base_udf.concat(pl.col("a"), pl.col("b")).alias("values"),
            base_udf.null_type(pl.col("a")).alias("null"),
        )

        assert result["values"].to_list() == [[1, 2]]
        assert result["null"].to_list() == [None]


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

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("floordiv", [2, 2]),
            ("mod", [1, 1]),
            ("lt", [False, False]),
            ("le", [False, False]),
            ("ge", [True, True]),
            ("neq", [True, True]),
        ],
    )
    def test_remaining_binary_contracts(self, name, expected):
        df = pl.DataFrame({"a": [5, 7], "b": [2, 3]})
        node = Call(name, (Column("a"), Column("b")))

        assert df.select(ast_compile(node)).to_series().to_list() == expected


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

    def test_arg_max_and_arg_min(self):
        df = pl.DataFrame({"a": [1.0, 5.0], "b": [3.0, 2.0]})

        result = df.select(
            ast_compile(Call("arg_max", (Column("a"), Column("b")))).alias("max"),
            ast_compile(Call("arg_min", (Column("a"), Column("b")))).alias("min"),
        )

        assert result["max"].to_list() == [1, 0]
        assert result["min"].to_list() == [0, 1]


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
