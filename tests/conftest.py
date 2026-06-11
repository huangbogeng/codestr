"""Shared test fixtures for CodeStr tests."""

import polars as pl
import pytest

from codestr.udf.registry import UDFRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset UDF registry before each test to avoid cross-test pollution.

    UDF modules are re-imported to re-register all @udf-decorated functions
    into the fresh singleton.
    """
    UDFRegistry.reset()
    import importlib

    from codestr.udf import base_udf, cs_udf, ts_udf

    importlib.reload(base_udf)
    importlib.reload(cs_udf)
    importlib.reload(ts_udf)


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Standard financial panel: 2 assets × 4 days."""
    return pl.DataFrame(
        {
            "datetime": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
            ],
            "asset": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "close": [100.0, 200.0, 101.0, 198.0, 102.0, 202.0, 103.0, 204.0],
            "volume": [1000.0, 2000.0, 1100.0, 1900.0, 1200.0, 2100.0, 1300.0, 2200.0],
            "high": [102.0, 205.0, 103.0, 202.0, 104.0, 206.0, 105.0, 208.0],
            "low": [98.0, 195.0, 99.0, 194.0, 100.0, 198.0, 101.0, 200.0],
        }
    )
