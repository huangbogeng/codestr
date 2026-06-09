"""
时间序列算子 (Time Series Operators)
"""

import polars as pl

from codestr.udf.registry import udf

over = dict(partition_by=["asset"], order_by=["datetime"])


def configure_over(
    *, partition_by: list[str] | None = None, order_by: list[str] | None = None
) -> None:
    """Update the default OVER window for all TS (time-series) operators."""
    if partition_by is not None:
        over["partition_by"] = list(partition_by)
    if order_by is not None:
        over["order_by"] = list(order_by)


@udf(category="ts")
def ts_max(expr: pl.Expr, windows):
    return expr.rolling_max(windows).over(**over)


@udf(category="ts")
def ts_min(expr: pl.Expr, windows):
    return expr.rolling_min(windows).over(**over)


@udf(category="ts")
def ts_mean(expr: pl.Expr, windows):
    return expr.rolling_mean(windows).over(**over)


@udf(category="ts")
def ts_std(expr: pl.Expr, windows):
    return expr.rolling_std(windows).over(**over)


@udf(category="ts")
def ts_skew(expr: pl.Expr, windows):
    return expr.rolling_skew(windows).over(**over)


@udf(category="ts")
def ts_kurt(expr: pl.Expr, windows):
    return expr.rolling_kurtosis(windows).over(**over)


@udf(category="ts")
def ts_sum(expr: pl.Expr, windows):
    return expr.rolling_sum(windows).over(**over)


@udf(category="ts")
def ts_var(expr: pl.Expr, windows):
    return expr.rolling_var(windows).over(**over)


@udf(category="ts")
def ts_mid(expr: pl.Expr, windows):
    return expr.rolling_median(windows).over(**over)


@udf(category="ts")
def ts_mad(expr: pl.Expr, windows):
    return 1.4826 * (expr - expr.rolling_median(windows).over(**over)).abs().rolling_median(
        windows
    ).over(**over)


@udf(category="ts")
def ts_delay(expr: pl.Expr, windows):
    """Lag operator: X_{t-d}"""
    return expr.shift(windows).over(**over)


@udf(category="ts")
def ts_delta(expr: pl.Expr, windows):
    """Delta operator: X_t - X_{t-d}"""
    return expr.diff(windows).over(**over)
