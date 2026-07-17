"""
时间序列算子 (Time Series Operators)

All TS operators accept optional ``partition_by`` and ``order_by`` keyword
arguments. When called through the CodeStr engine, these are automatically
injected from the instance's ``time_col`` / ``asset_col`` configuration.
When called directly through the compiler, canonical defaults
(``["asset"]`` / ``["datetime"]``) are used.
"""

import polars as pl

from codestr.udf.registry import udf

__all__ = [
    "ts_delay",
    "ts_delta",
    "ts_ema",
    "ts_kurt",
    "ts_mad",
    "ts_max",
    "ts_mean",
    "ts_mid",
    "ts_min",
    "ts_skew",
    "ts_std",
    "ts_sum",
    "ts_var",
]


def _validate_min_samples(min_samples: int | None) -> int | None:
    if min_samples is not None and (type(min_samples) is not int or min_samples < 1):
        raise ValueError("min_samples must be a positive integer")
    return min_samples


@udf(category="ts")
def ts_max(
    expr: pl.Expr,
    windows,
    partition_by=None,
    order_by=None,
    *,
    min_samples=None,
):
    min_samples = _validate_min_samples(min_samples)
    return expr.rolling_max(windows, min_samples=min_samples).over(
        partition_by=partition_by,
        order_by=order_by,
    )


@udf(category="ts")
def ts_min(
    expr: pl.Expr,
    windows,
    partition_by=None,
    order_by=None,
    *,
    min_samples=None,
):
    min_samples = _validate_min_samples(min_samples)
    return expr.rolling_min(windows, min_samples=min_samples).over(
        partition_by=partition_by,
        order_by=order_by,
    )


@udf(category="ts")
def ts_mean(
    expr: pl.Expr,
    windows,
    partition_by=None,
    order_by=None,
    *,
    min_samples=None,
):
    min_samples = _validate_min_samples(min_samples)
    return expr.rolling_mean(windows, min_samples=min_samples).over(
        partition_by=partition_by,
        order_by=order_by,
    )


@udf(category="ts")
def ts_ema(
    expr: pl.Expr,
    span: int,
    partition_by=None,
    order_by=None,
    *,
    min_samples: int = 1,
):
    """Recursive exponential moving average over each ordered entity series."""
    if type(span) is not int or span < 1:
        raise ValueError("span must be a positive integer")
    min_samples = _validate_min_samples(min_samples)
    return expr.ewm_mean(span=span, adjust=False, min_samples=min_samples).over(
        partition_by=partition_by,
        order_by=order_by,
    )


@udf(category="ts")
def ts_std(
    expr: pl.Expr,
    windows,
    partition_by=None,
    order_by=None,
    *,
    min_samples=None,
):
    min_samples = _validate_min_samples(min_samples)
    return expr.rolling_std(windows, min_samples=min_samples).over(
        partition_by=partition_by,
        order_by=order_by,
    )


@udf(category="ts")
def ts_skew(
    expr: pl.Expr,
    windows,
    partition_by=None,
    order_by=None,
    *,
    min_samples=None,
):
    min_samples = _validate_min_samples(min_samples)
    return expr.rolling_skew(windows, min_samples=min_samples).over(
        partition_by=partition_by,
        order_by=order_by,
    )


@udf(category="ts")
def ts_kurt(
    expr: pl.Expr,
    windows,
    partition_by=None,
    order_by=None,
    *,
    min_samples=None,
):
    min_samples = _validate_min_samples(min_samples)
    return expr.rolling_kurtosis(windows, min_samples=min_samples).over(
        partition_by=partition_by,
        order_by=order_by,
    )


@udf(category="ts")
def ts_sum(
    expr: pl.Expr,
    windows,
    partition_by=None,
    order_by=None,
    *,
    min_samples=None,
):
    min_samples = _validate_min_samples(min_samples)
    return expr.rolling_sum(windows, min_samples=min_samples).over(
        partition_by=partition_by,
        order_by=order_by,
    )


@udf(category="ts")
def ts_var(
    expr: pl.Expr,
    windows,
    partition_by=None,
    order_by=None,
    *,
    min_samples=None,
):
    min_samples = _validate_min_samples(min_samples)
    return expr.rolling_var(windows, min_samples=min_samples).over(
        partition_by=partition_by,
        order_by=order_by,
    )


@udf(category="ts")
def ts_mid(
    expr: pl.Expr,
    windows,
    partition_by=None,
    order_by=None,
    *,
    min_samples=None,
):
    min_samples = _validate_min_samples(min_samples)
    return expr.rolling_median(windows, min_samples=min_samples).over(
        partition_by=partition_by,
        order_by=order_by,
    )


@udf(category="ts")
def ts_mad(
    expr: pl.Expr,
    windows,
    partition_by=None,
    order_by=None,
    *,
    min_samples=None,
):
    """Median Absolute Deviation over a rolling window.

    MAD = 1.4826 * median(|x - rolling_median(x)|)
    The constant 1.4826 scales MAD to be consistent with standard deviation
    for normally distributed data.
    """
    min_samples = _validate_min_samples(min_samples)
    return 1.4826 * (
        expr
        - expr.rolling_median(windows, min_samples=min_samples).over(
            partition_by=partition_by,
            order_by=order_by,
        )
    ).abs().rolling_median(windows, min_samples=min_samples).over(
        partition_by=partition_by,
        order_by=order_by,
    )


@udf(category="ts")
def ts_delay(expr: pl.Expr, windows, partition_by=None, order_by=None):
    """Lag operator: X_{t-d}.  Returns the value *windows* periods ago."""
    return expr.shift(windows).over(partition_by=partition_by, order_by=order_by)


@udf(category="ts")
def ts_delta(expr: pl.Expr, windows, partition_by=None, order_by=None):
    """Delta operator: X_t - X_{t-d}.  Difference over *windows* periods."""
    return expr.diff(windows).over(partition_by=partition_by, order_by=order_by)
