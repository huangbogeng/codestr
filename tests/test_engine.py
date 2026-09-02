"""Tests for CodeStr engine (engine.py)."""

import polars as pl
import pytest

from codestr.engine import CodeStr
from codestr.errors import CompileError, PolarsError
from codestr.udf.registry import UDFMeta, UDFRegistry


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

    def test_init_collects_lazy_input_in_eager_mode(self, sample_df):
        cs = CodeStr(sample_df.lazy())

        assert isinstance(cs.data, pl.DataFrame)
        assert cs.data.height == sample_df.height

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

    def test_same_query_reuses_expression_with_new_alias(self, sample_df):
        cs = CodeStr(sample_df)

        result = cs.sql(
            "close + 1 as first",
            "close + 1 as second",
        )

        assert cs.failed == []
        assert result["first"].equals(result["second"])

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

    def test_mixed_window_uses_preceding_alias_in_same_sql(self, mixed_window_df):
        actual_engine = CodeStr(mixed_window_df, align=False)
        actual = actual_engine.sql(
            "x * 2 as scaled",
            "ts_mean(cs_moderate(scaled), 2, min_samples=1) as factor",
        ).sort(["asset", "datetime"])

        expected_engine = CodeStr(mixed_window_df, align=False)
        expected = expected_engine.sql(
            "x * 2 as scaled",
            "cs_moderate(scaled) as moderate",
            "ts_mean(moderate, 2, min_samples=1) as factor",
        ).sort(["asset", "datetime"])

        assert actual_engine.failed == []
        assert actual["factor"].equals(expected["factor"])
        assert actual.filter(pl.col("asset") == "A")["factor"].head(4).to_list() == [
            2.0,
            3.0,
            5.0,
            7.0,
        ]

    def test_mixed_window_lazy_matches_eager(self, mixed_window_df):
        eager_engine = CodeStr(mixed_window_df, align=False)
        eager = eager_engine.sql("ts_mean(cs_moderate(x), 2, min_samples=1) as factor").sort(
            ["asset", "datetime"]
        )

        lazy_engine = CodeStr(mixed_window_df, align=False)
        lazy = (
            lazy_engine.sql(
                "ts_mean(cs_moderate(x), 2, min_samples=1) as factor",
                lazy=True,
            )
            .collect()
            .sort(["asset", "datetime"])
        )

        assert lazy["factor"].equals(eager["factor"])
        assert lazy_engine._expr_cache == {}

    def test_mixed_window_respects_custom_over_columns(self):
        df = pl.DataFrame(
            {
                "trade_date": [1, 1, 2, 2, 3, 3],
                "symbol": ["A", "B", "A", "B", "A", "B"],
                "x": [1.0, 3.0, 2.0, 6.0, 4.0, 8.0],
            }
        )
        actual_engine = CodeStr(
            df,
            index=("trade_date", "symbol"),
            partition_by=["symbol"],
            order_by=["trade_date"],
            align=False,
        )
        actual = actual_engine.sql("ts_mean(cs_moderate(x), 2, min_samples=1) as factor").sort(
            ["symbol", "trade_date"]
        )

        expected = (
            df.lazy()
            .with_columns(
                (
                    pl.col("x")
                    - pl.col("x")
                    .mean()
                    .over(
                        partition_by=["trade_date"],
                        order_by=["symbol"],
                    )
                )
                .abs()
                .alias("moderate")
            )
            .with_columns(
                pl.col("moderate")
                .rolling_mean(2, min_samples=1)
                .over(
                    partition_by=["symbol"],
                    order_by=["trade_date"],
                )
                .alias("factor")
            )
            .select("trade_date", "symbol", "factor")
            .collect()
            .sort(["symbol", "trade_date"])
        )

        assert actual["factor"].equals(expected["factor"])

    def test_mixed_window_reuses_root_cache_with_new_alias(self, mixed_window_df):
        cs = CodeStr(mixed_window_df, align=False)

        first = cs.sql("ts_mean(cs_moderate(x), 2, min_samples=1) as first")
        cache_size = len(cs._expr_cache)
        second = cs.sql("ts_mean(cs_moderate(x), 2, min_samples=1) as second")

        assert cs.failed == []
        assert second["second"].to_list() == first["first"].to_list()
        assert len(cs._expr_cache) == cache_size

    def test_mixed_window_cover_recomputes_internal_stages(self, mixed_window_df):
        cs = CodeStr(mixed_window_df, align=False)
        first = cs.sql("ts_mean(cs_moderate(x), 2, min_samples=1) as factor")
        initial_internal = {
            name for name in cs._data_.collect_schema().names() if name.startswith("__codestr_")
        }

        cs._data_ = cs._data_.with_columns((pl.col("x") * 2).alias("x"))
        second = cs.sql(
            "ts_mean(cs_moderate(x), 2, min_samples=1) as factor",
            cover=True,
        )

        assert second["factor"].to_list() != first["factor"].to_list()
        assert second["factor"].fill_null(0).sum() == pytest.approx(
            2 * first["factor"].fill_null(0).sum()
        )
        current_internal = {
            name for name in cs._data_.collect_schema().names() if name.startswith("__codestr_")
        }
        assert current_internal == initial_internal

    def test_failed_mixed_plan_does_not_leak_internal_columns(self, mixed_window_df):
        cs = CodeStr(mixed_window_df, align=False)
        before = set(cs.data.columns)

        def fail_after_child(
            expr,
            partition_by=None,
            order_by=None,
        ):
            raise ValueError("planned failure")

        UDFRegistry.get_instance().register(
            UDFMeta(
                name="ts_fail_after_child",
                fn=fail_after_child,
                category="ts",
            )
        )
        result = cs.sql("ts_fail_after_child(cs_moderate(x)) as invalid")

        assert len(cs.failed) == 1
        assert "planned failure" in str(cs.failed[0])
        assert set(cs._data_.collect_schema().names()) == before
        assert not any(name.startswith("__codestr_") for name in cs._data_.collect_schema().names())
        assert "invalid" not in result.columns

    def test_missing_column_mixed_plan_rolls_back_and_allows_later_query(
        self,
        mixed_window_df,
    ):
        cs = CodeStr(mixed_window_df, align=False)
        before = set(cs.data.columns)

        result = cs.sql("ts_mean(cs_moderate(missing), 2, min_samples=1) as invalid")

        assert len(cs.failed) == 1
        assert set(cs._data_.collect_schema().names()) == before
        assert cs._cur_expr_cache == {}
        assert "invalid" not in result.columns

        valid = cs.sql("x + 1 as valid")

        assert cs.failed == []
        assert "valid" in valid.columns

    def test_collect_failure_restores_query_state(self, mixed_window_df):
        cs = CodeStr(mixed_window_df, align=False)

        def raise_during_collect(value):
            raise ValueError("collect failure")

        def fail_during_collect(
            expr,
            partition_by=None,
            order_by=None,
        ):
            return expr.map_elements(
                raise_during_collect,
                return_dtype=pl.Float64,
            )

        UDFRegistry.get_instance().register(
            UDFMeta(
                name="ts_fail_during_collect",
                fn=fail_during_collect,
                category="ts",
            )
        )

        with pytest.raises(PolarsError, match="collect failure"):
            cs.sql("ts_fail_during_collect(cs_moderate(x)) as invalid")

        assert cs._data_ is None
        assert cs._expr_cache == {}
        assert cs._cur_expr_cache == {}
        assert cs._internal_columns == set()


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


class TestValidateExpr:
    def test_validates_expression_against_current_schema(self, sample_df):
        cs = CodeStr(sample_df)

        assert cs.validate_expr("ts_mean(close, 2) as mean_close") == [
            {
                "expr": "ts_mean(close, 2) as mean_close",
                "valid": True,
                "stage": "schema",
                "error_type": None,
                "message": None,
            }
        ]

    def test_reports_structural_compile_and_schema_failures(self, sample_df):
        cs = CodeStr(sample_df)

        structural = cs.validate_expr("")[0]
        compile_failure = cs.validate_expr("sin(1.0) as factor")[0]
        schema_failure = cs.validate_expr("missing + 1 as factor")[0]

        assert structural["valid"] is False
        assert structural["stage"] == "structural"
        assert structural["error_type"] == "ParseError"
        assert structural["message"]
        assert compile_failure["valid"] is False
        assert compile_failure["stage"] == "compile"
        assert compile_failure["error_type"] == "CompileError"
        assert "sin" in compile_failure["message"]
        assert schema_failure["valid"] is False
        assert schema_failure["stage"] == "schema"
        assert schema_failure["error_type"] == "ColumnNotFoundError"
        assert "missing" in schema_failure["message"]

    def test_reports_non_parse_structural_failure(self, sample_df):
        cs = CodeStr(sample_df)

        result = cs.validate_expr("close - close as invalid")[0]

        assert result["valid"] is False
        assert result["stage"] == "structural"
        assert result["error_type"] == "StructuralError"
        assert "redundant:sub" in result["message"]

    def test_reports_invalid_base_lazy_schema(self):
        invalid = pl.DataFrame({"x": [1.0]}).lazy().select("missing")
        cs = CodeStr(invalid, pure_lazy=True)

        result = cs.validate_expr("x + 1 as factor")[0]

        assert result["valid"] is False
        assert result["stage"] == "schema"
        assert result["error_type"] == "ColumnNotFoundError"

    def test_validates_mixed_window_through_planner(self, mixed_window_df):
        cs = CodeStr(mixed_window_df, align=False)

        result = cs.validate_expr("ts_mean(cs_moderate(x), 2, min_samples=1) as factor")

        assert result[0]["valid"] is True
        assert result[0]["stage"] == "schema"

    def test_batch_expressions_are_independent(self, sample_df):
        cs = CodeStr(sample_df)

        results = cs.validate_expr(
            "close * 2 as scaled",
            "scaled + 1 as shifted",
        )

        assert results[0]["valid"] is True
        assert results[1]["valid"] is False
        assert results[1]["stage"] == "schema"

    def test_does_not_collect_or_mutate_engine_state(self, sample_df):
        cs = CodeStr(sample_df)
        cs.sql("close + 1 as cached")

        def raise_during_collect(value):
            raise ValueError("must not collect")

        def lazy_only(expr):
            return expr.map_elements(
                raise_during_collect,
                return_dtype=pl.Float64,
            )

        cs.register_udf(lazy_only)
        state = {
            "data": cs.data,
            "lazy": cs._data_,
            "expr_cache": dict(cs._expr_cache),
            "cur_expr_cache": dict(cs._cur_expr_cache),
            "internal_columns": set(cs._internal_columns),
            "last_query_cache": cs._last_query_cache,
            "failed": list(cs.failed),
        }

        result = cs.validate_expr("lazy_only(close) as validated")

        assert result[0]["valid"] is True
        assert cs.data is state["data"]
        assert cs._data_ is state["lazy"]
        assert cs._expr_cache == state["expr_cache"]
        assert cs._cur_expr_cache == state["cur_expr_cache"]
        assert cs._internal_columns == state["internal_columns"]
        assert cs._last_query_cache is state["last_query_cache"]
        assert cs.failed == state["failed"]


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
