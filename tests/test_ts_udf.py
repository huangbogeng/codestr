"""Tests for time-series UDF operators (udf/ts_udf.py)."""

import polars as pl
import pytest

from codestr.compiler import compile as ast_compile
from codestr.errors import CompileError
from codestr.syntax import Call, Column, Literal


@pytest.fixture
def ts_df() -> pl.DataFrame:
    """Time-series panel: 1 asset × 5 days."""
    return pl.DataFrame(
        {
            "datetime": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "asset": ["A", "A", "A", "A", "A"],
            "close": [100.0, 102.0, 104.0, 103.0, 105.0],
        }
    )


class TestTSMean:
    def test_ts_mean_window3(self, ts_df):
        node = Call("ts_mean", (Column("close"), Literal(3)))
        expr = ast_compile(node)
        result = ts_df.select(expr)
        # Rolling mean over window=3: [NaN, NaN, 102.0, 103.0, 104.0]
        assert result["ts_mean(close, 3)"].to_list()[2] == 102.0  # (100+102+104)/3


class TestTSEma:
    def test_matches_recursive_polars_ema(self):
        df = pl.DataFrame(
            {
                "datetime": [3, 1, 2, 1, 3, 2],
                "asset": ["A", "A", "A", "B", "B", "B"],
                "value": [3.0, 1.0, None, 10.0, 14.0, 12.0],
            }
        )
        node = Call("ts_ema", (Column("value"), Literal(2)))

        actual = df.select(ast_compile(node))[node.alias]
        expected = df.select(
            pl.col("value")
            .ewm_mean(span=2, adjust=False, min_samples=1)
            .over(partition_by=["asset"], order_by=["datetime"])
            .alias("expected")
        )["expected"]

        assert actual.equals(expected)
        assert actual.to_list() == pytest.approx(
            [2.7142857142857144, 1.0, None, 10.0, 13.11111111111111, 11.333333333333334]
        )

    def test_span_one_returns_input(self, ts_df):
        node = Call("ts_ema", (Column("close"), Literal(1)))

        actual = ts_df.select(ast_compile(node))[node.alias]

        assert actual.to_list() == ts_df["close"].to_list()

    @pytest.mark.parametrize("span", [0, -1, 2.5, True])
    def test_rejects_invalid_span(self, span):
        node = Call("ts_ema", (Column("close"), Literal(span)))

        with pytest.raises(CompileError, match="positive integer"):
            ast_compile(node)


class TestTSMax:
    def test_ts_max_window3(self, ts_df):
        node = Call("ts_max", (Column("close"), Literal(3)))
        expr = ast_compile(node)
        result = ts_df.select(expr)
        # Rolling max over window=3: max(100,102,104)=104
        assert result["ts_max(close, 3)"].to_list()[2] == 104.0


class TestTSMin:
    def test_ts_min_window3(self, ts_df):
        node = Call("ts_min", (Column("close"), Literal(3)))
        expr = ast_compile(node)
        result = ts_df.select(expr)
        assert result["ts_min(close, 3)"].to_list()[2] == 100.0


class TestTSSum:
    def test_ts_sum_window3(self, ts_df):
        node = Call("ts_sum", (Column("close"), Literal(3)))
        expr = ast_compile(node)
        result = ts_df.select(expr)
        assert result["ts_sum(close, 3)"].to_list()[2] == 306.0  # 100+102+104


class TestTSStd:
    def test_ts_std_window3(self, ts_df):
        node = Call("ts_std", (Column("close"), Literal(3)))
        expr = ast_compile(node)
        result = ts_df.select(expr)
        assert result.height == 5


class TestTSDelay:
    def test_ts_delay_1(self, ts_df):
        node = Call("ts_delay", (Column("close"), Literal(1)))
        expr = ast_compile(node)
        result = ts_df.select(expr)
        vals = result["ts_delay(close, 1)"].to_list()
        assert vals[0] is None  # shifted back 1
        assert vals[1] == 100.0

    def test_ts_delay_2(self, ts_df):
        node = Call("ts_delay", (Column("close"), Literal(2)))
        expr = ast_compile(node)
        result = ts_df.select(expr)
        vals = result["ts_delay(close, 2)"].to_list()
        assert vals[0] is None
        assert vals[1] is None
        assert vals[2] == 100.0


class TestTSDelta:
    def test_ts_delta_1(self, ts_df):
        node = Call("ts_delta", (Column("close"), Literal(1)))
        expr = ast_compile(node)
        result = ts_df.select(expr)
        vals = result["ts_delta(close, 1)"].to_list()
        assert vals[0] is None  # diff of 1
        assert vals[1] == 2.0  # 102-100


class TestTSSkew:
    def test_ts_skew(self, ts_df):
        node = Call("ts_skew", (Column("close"), Literal(5)))
        expr = ast_compile(node)
        result = ts_df.select(expr)
        assert result.height == 5
