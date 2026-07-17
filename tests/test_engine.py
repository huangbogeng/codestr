"""Tests for CodeStr engine (engine.py)."""

import polars as pl
import pytest

from codestr.engine import CodeStr
from codestr.errors import CompileError


class TestCodeStrInit:
    def test_init_with_dataframe(self, sample_df):
        cs = CodeStr(sample_df)
        assert cs.data is not None
        assert "close" in cs.cache_columns

    def test_init_pure_lazy(self, sample_df):
        lazy = sample_df.lazy()
        cs = CodeStr(lazy, pure_lazy=True)
        assert cs.data is None
        assert "close" in cs.cache_columns

    def test_init_invalid_data_raises(self):
        with pytest.raises(AssertionError):
            CodeStr([1, 2, 3])  # type: ignore


class TestCompilePureMode:
    def test_compile_simple(self, sample_df):
        cs = CodeStr(sample_df)
        expr = cs.compile("close")
        assert isinstance(expr, pl.Expr)

    def test_compile_arithmetic(self, sample_df):
        cs = CodeStr(sample_df)
        expr = cs.compile("close + volume")
        result = sample_df.with_columns(expr)
        assert "(close+volume)" in result.columns

    def test_compile_no_side_effects(self, sample_df):
        cs = CodeStr(sample_df)
        initial_cols = len(cs.cache_columns)
        cs.compile("close + volume")
        cs.compile("high - low")
        # compile() should not mutate internal state
        assert len(cs.cache_columns) == initial_cols

    def test_compile_invalid_expr_raises(self, sample_df):
        cs = CodeStr(sample_df)
        with pytest.raises(CompileError):
            cs.compile("")


class TestSQLInteractiveMode:
    def test_sql_basic(self, sample_df):
        cs = CodeStr(sample_df)
        result = cs.sql("close + volume as result", cover=True)
        assert "result" in result.columns
        assert result.height == 8

    def test_sql_without_cover_skips_existing(self, sample_df):
        cs = CodeStr(sample_df)
        _result1 = cs.sql("close + volume as result", cover=True)
        # Second call without cover should use existing column
        result2 = cs.sql("close + volume as result", cover=False)
        assert "result" in result2.columns

    def test_sql_lazy(self, sample_df):
        cs = CodeStr(sample_df)
        result = cs.sql("close + volume as result", cover=True, lazy=True)
        assert isinstance(result, pl.LazyFrame)
        # Lazy should not mutate data
        materialized = result.collect()
        assert "result" in materialized.columns

    def test_sql_multiple_expressions(self, sample_df):
        cs = CodeStr(sample_df)
        result = cs.sql(
            "close + volume as total",
            "high - low as spread",
            cover=True,
        )
        assert "total" in result.columns
        assert "spread" in result.columns

    def test_sql_caching_reuses_expr(self, sample_df):
        """Subsequent identical expressions should reuse cached columns."""
        cs = CodeStr(sample_df)
        _result1 = cs.sql("close + 1 as offset", cover=True)
        # Same expression again — should be a cache hit
        result2 = cs.sql("close + 1 as offset", cover=False)
        assert "offset" in result2.columns

    def test_sql_keyword_argument_reuses_cache_with_new_alias(self, sample_df):
        cs = CodeStr(sample_df)

        first = cs.sql("clip(close, lower_bound=101) as clipped")
        cache_size = len(cs._expr_cache)
        second = cs.sql("clip(close, lower_bound=101) as clipped_again")

        assert cs.failed == []
        assert first["clipped"].to_list() == [
            101.0,
            200.0,
            101.0,
            198.0,
            102.0,
            202.0,
            103.0,
            204.0,
        ]
        assert second["clipped_again"].to_list() == first["clipped"].to_list()
        assert len(cs._expr_cache) == cache_size

    def test_sql_min_samples_keyword_reuses_cache(self, sample_df):
        cs = CodeStr(sample_df)

        first = cs.sql("ts_mean(close, 3, min_samples=1) as mean_fast")
        cache_size = len(cs._expr_cache)
        second = cs.sql("ts_mean(close, 3, min_samples=1) as mean_fast_again")

        assert cs.failed == []
        assert first["mean_fast"].null_count() == 0
        assert second["mean_fast_again"].to_list() == first["mean_fast"].to_list()
        assert len(cs._expr_cache) == cache_size

    def test_sql_cache_does_not_bypass_min_samples_type_validation(self, sample_df):
        cs = CodeStr(sample_df)

        cs.sql("ts_mean(close, 3, min_samples=1) as valid")
        result = cs.sql("ts_mean(close, 3, min_samples=1.0) as invalid")

        assert len(cs.failed) == 1
        assert "min_samples must be a positive integer" in str(cs.failed[0])
        assert "invalid" not in result.columns

    def test_ts_ema_can_feed_later_expression_in_same_sql(self, sample_df):
        cs = CodeStr(sample_df)

        result = cs.sql(
            "ts_ema(close, 2) as ema2",
            "ts_delta(ema2, 1) as ema_delta",
        )

        assert cs.failed == []
        assert {"ema2", "ema_delta"} <= set(result.columns)
        assert result["ema_delta"].null_count() == 2

    def test_sql_cover_overwrites(self, sample_df):
        cs = CodeStr(sample_df)
        cs.sql("close + 1 as offset", cover=True)
        result = cs.sql("close + 2 as offset", cover=True)
        # cover=True should recalculate
        assert "offset" in result.columns

    def test_sql_lowers_ts_over_cs(self, mixed_window_df):
        actual_engine = CodeStr(mixed_window_df, align=False)
        actual = actual_engine.sql("ts_mean(cs_moderate(x), 60) as factor").sort(
            ["asset", "datetime"]
        )

        expected_engine = CodeStr(mixed_window_df, align=False)
        expected_engine.sql("cs_moderate(x) as moderate")
        expected = expected_engine.sql("ts_mean(moderate, 60) as factor").sort(
            ["asset", "datetime"]
        )

        assert actual_engine.failed == []
        assert actual["factor"].equals(expected["factor"])
        assert actual.filter(pl.col("asset") == "A")["factor"].tail(3).to_list() == [
            33.5,
            34.5,
            35.5,
        ]
        assert actual.columns == ["datetime", "asset", "factor"]

    def test_sql_lowers_cs_over_ts(self, mixed_window_df):
        actual_engine = CodeStr(mixed_window_df, align=False)
        actual = actual_engine.sql("cs_mean(ts_mean(x, 2, min_samples=1)) as factor").sort(
            ["asset", "datetime"]
        )

        expected_engine = CodeStr(mixed_window_df, align=False)
        expected_engine.sql("ts_mean(x, 2, min_samples=1) as rolling")
        expected = expected_engine.sql("cs_mean(rolling) as factor").sort(["asset", "datetime"])

        assert actual_engine.failed == []
        assert actual["factor"].equals(expected["factor"])


class TestCheckExpr:
    def test_valid_expr(self, sample_df):
        cs = CodeStr(sample_df)
        result = cs.check_expr("close + volume")
        assert result["valid"] is True
        assert result["reasons"] == []

    def test_max_depth_exceeded(self, sample_df):
        cs = CodeStr(sample_df)
        # Nested expression with depth > 1
        result = cs.check_expr("close + volume + high", max_depth=1)
        assert result["valid"] is False
        assert any("max_depth" in r for r in result["reasons"])

    def test_max_nodes_exceeded(self, sample_df):
        cs = CodeStr(sample_df)
        result = cs.check_expr("close + volume + high + low", max_nodes=1)
        assert result["valid"] is False
        assert any("max_nodes" in r for r in result["reasons"])

    def test_redundant_detection(self, sample_df):
        cs = CodeStr(sample_df)
        result = cs.check_expr("close - close", check_redundant=True)
        assert result["valid"] is False
        assert any("redundant" in r for r in result["reasons"])

    def test_invalid_expression(self, sample_df):
        cs = CodeStr(sample_df)
        result = cs.check_expr("")
        assert result["valid"] is False

    def test_keyword_argument_expression(self, sample_df):
        cs = CodeStr(sample_df)

        result = cs.check_expr("clip(abs(close), lower_bound=0)")

        assert result == {
            "valid": True,
            "reasons": [],
            "expr": "clip(abs(close), lower_bound=0)",
        }

    def test_mixed_window_is_structurally_valid(self, sample_df):
        cs = CodeStr(sample_df)

        result = cs.check_expr("ts_mean(cs_moderate(close), 60)")

        assert result["valid"] is True
        assert result["reasons"] == []


class TestClearCache:
    def test_clear_cache_resets_state(self, sample_df):
        cs = CodeStr(sample_df)
        cs.sql("close + 1 as offset", cover=True)
        cs.clear_cache()
        # After clear, internal state should be reset
        assert cs._data_ is None
        assert cs._expr_cache == {}


class TestRegisterUDF:
    def test_register_custom_udf(self, sample_df):
        import polars as pl

        cs = CodeStr(sample_df)

        def triple(x: pl.Expr) -> pl.Expr:
            return x * 3

        cs.register_udf(triple, name="triple")
        expr = cs.compile("triple(close)")
        result = sample_df.with_columns(expr)
        assert "triple(close)" in result.columns


class TestPerInstanceOverConfig:
    """Verify that over-window config is per-instance, explicitly provided."""

    def test_custom_partition_and_order(self):
        """TS/CS operators use partition_by/order_by passed at construction."""
        df = pl.DataFrame(
            {
                "trade_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                "stock_code": ["X", "Y", "X", "Y"],
                "close": [10.0, 20.0, 15.0, 25.0],
            }
        )
        cs = CodeStr(
            df,
            index=("trade_date", "stock_code"),
            partition_by=["stock_code"],
            order_by=["trade_date"],
        )

        result = cs.sql("ts_mean(close, 2) as ma2", cover=True)
        assert "ma2" in result.columns
        assert "trade_date" in result.columns
        assert "stock_code" in result.columns

    def test_multi_column_partition_and_order(self):
        """partition_by and order_by accept multiple columns."""
        df = pl.DataFrame(
            {
                "date": ["D1", "D1", "D2", "D2"],
                "tick": [1, 2, 1, 2],
                "sector": ["TECH", "FIN", "TECH", "FIN"],
                "stock": ["A", "B", "A", "B"],
                "close": [10.0, 20.0, 15.0, 25.0],
            }
        )
        cs = CodeStr(
            df,
            index=("date", "stock"),
            partition_by=["sector", "stock"],
            order_by=["date", "tick"],
        )

        result = cs.sql("ts_mean(close, 2) as ma2", cover=True)
        assert "ma2" in result.columns

    def test_two_instances_independent(self):
        """Two instances with different configs do not interfere."""
        df1 = pl.DataFrame(
            {
                "date": ["D1", "D1", "D2", "D2"],
                "sym": ["A", "B", "A", "B"],
                "val": [1.0, 2.0, 3.0, 4.0],
            }
        )
        df2 = pl.DataFrame(
            {
                "day": ["D1", "D1"],
                "code": ["X", "Y"],
                "val": [10.0, 20.0],
            }
        )

        cs1 = CodeStr(df1, index=("date", "sym"))
        cs2 = CodeStr(df2, index=("day", "code"))

        r1 = cs1.sql("ts_mean(val, 2) as m", cover=True)
        r2 = cs2.sql("ts_mean(val, 2) as m", cover=True)

        assert "m" in r1.columns
        assert "m" in r2.columns

    def test_compile_with_custom_config(self):
        """compile() also pipes the over config."""
        df = pl.DataFrame(
            {
                "dt": ["2024-01-01", "2024-01-01"],
                "sym": ["A", "B"],
                "x": [10.0, 20.0],
            }
        )
        cs = CodeStr(df, index=("dt", "sym"))
        expr = cs.compile("cs_rank(x)")
        result = df.with_columns(expr)
        assert "cs_rank(x)" in result.columns
