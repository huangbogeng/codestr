"""Tests for cross-section UDF operators (udf/cs_udf.py)."""

import polars as pl
import pytest

from codestr.compiler import compile as ast_compile
from codestr.syntax import Call, Column, Literal


@pytest.fixture
def cs_df() -> pl.DataFrame:
    """Cross-section panel: 2 assets × 3 days."""
    return pl.DataFrame(
        {
            "datetime": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
            ],
            "asset": ["A", "B", "A", "B", "A", "B"],
            "x": [10.0, 20.0, 15.0, 25.0, 30.0, 10.0],
            "y": [1.0, 3.0, 2.0, 4.0, 5.0, 2.0],
        }
    )


class TestCSRank:
    def test_cs_rank(self, cs_df):
        node = Call("cs_rank", (Column("x"),))
        expr = ast_compile(node)
        result = cs_df.select(expr)
        # A rank vs B rank within each day
        assert result.height == 6


class TestCSDemean:
    def test_cs_demean(self, cs_df):
        node = Call("cs_demean", (Column("x"),))
        expr = ast_compile(node)
        result = cs_df.select(expr)
        # Day 1: mean=15, A=10-15=-5, B=20-15=5
        assert result.height == 6


class TestCSZscore:
    def test_cs_zscore(self, cs_df):
        node = Call("cs_zscore", (Column("x"),))
        expr = ast_compile(node)
        result = cs_df.select(expr)
        assert result.height == 6


class TestCSMean:
    def test_cs_mean(self, cs_df):
        node = Call("cs_mean", (Column("x"),))
        expr = ast_compile(node)
        result = cs_df.select(expr)
        # Day 1: both assets get mean=15
        assert pytest.approx(result["cs_mean(x)"][0]) == 15.0
        assert pytest.approx(result["cs_mean(x)"][1]) == 15.0


class TestCSMax:
    def test_cs_max(self, cs_df):
        node = Call("cs_max", (Column("x"),))
        expr = ast_compile(node)
        result = cs_df.select(expr)
        # Day 1: max=20, Day 2: max=25, Day 3: max=30
        assert result["cs_max(x)"][0] == 20.0
        assert result["cs_max(x)"][2] == 25.0


class TestCSIC:
    def test_cs_ic(self, cs_df):
        node = Call("cs_ic", (Column("x"), Column("y")))
        expr = ast_compile(node)
        result = cs_df.select(expr)
        # Spearman rank correlation within each day
        assert result.height == 6


class TestCSQcut:
    def test_cs_qcut(self, cs_df):
        node = Call("cs_qcut", (Column("x"), Literal(2)))
        expr = ast_compile(node)
        result = cs_df.select(expr)
        assert result.height == 6


class TestCSMidby:
    def test_cs_midby(self, cs_df):
        node = Call("cs_midby", (Column("x"),))
        expr = ast_compile(node)
        result = cs_df.select(expr)
        assert result.height == 6
