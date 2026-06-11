"""
截面算子 (Cross-Section Operators)

All CS operators accept optional ``partition_by`` and ``order_by`` keyword
arguments. When called through the CodeStr engine, these are automatically
injected from the instance's ``time_col`` / ``asset_col`` configuration.
When called directly through the compiler, canonical defaults
(``["datetime"]`` / ``["asset"]``) are used.
"""

import polars as pl

from codestr.udf.registry import udf

__all__ = [
    "cs_corr",
    "cs_demean",
    "cs_ic",
    "cs_max",
    "cs_mean",
    "cs_meanby",
    "cs_mid",
    "cs_midby",
    "cs_min",
    "cs_moderate",
    "cs_peakmax",
    "cs_peakmin",
    "cs_qcut",
    "cs_rank",
    "cs_resid",
    "cs_skew",
    "cs_slope",
    "cs_std",
    "cs_ufit",
    "cs_var",
    "cs_zscore",
]


@udf(category="cs")
def cs_ufit(expr: pl.Expr, partition_by=None, order_by=None):
    """Cross-section median absolute deviation: |x - median(x)|."""
    return (expr - expr.median().over(partition_by=partition_by, order_by=order_by)).abs()


@udf(category="cs")
def cs_rank(expr: pl.Expr, partition_by=None, order_by=None):
    return expr.rank().over(partition_by=partition_by, order_by=order_by)


@udf(category="cs")
def cs_demean(expr: pl.Expr, partition_by=None, order_by=None):
    return expr - expr.mean().over(partition_by=partition_by, order_by=order_by)


@udf(category="cs")
def cs_mean(expr: pl.Expr, partition_by=None, order_by=None):
    return expr.mean().over(partition_by=partition_by, order_by=order_by)


@udf(category="cs")
def cs_mid(expr: pl.Expr, partition_by=None, order_by=None):
    return expr.median().over(partition_by=partition_by, order_by=order_by)


@udf(category="cs")
def cs_moderate(expr: pl.Expr, partition_by=None, order_by=None):
    return (expr - expr.mean().over(partition_by=partition_by, order_by=order_by)).abs()


@udf(category="cs")
def cs_qcut(expr: pl.Expr, n_bins=10, partition_by=None, order_by=None):
    return (
        expr.qcut(n_bins, labels=[str(i) for i in range(1, n_bins + 1)], allow_duplicates=True)
        .over(partition_by=partition_by, order_by=order_by)
        .cast(pl.Int32)
    )


@udf(category="cs")
def cs_ic(left: pl.Expr, right: pl.Expr, partition_by=None, order_by=None):
    """Cross-section Information Coefficient (Spearman rank correlation)."""
    return pl.corr(left, right, method="spearman").over(
        partition_by=partition_by, order_by=order_by
    )


@udf(category="cs")
def cs_corr(left: pl.Expr, right: pl.Expr, partition_by=None, order_by=None):
    """Cross-section Pearson correlation."""
    return pl.corr(left, right, method="pearson").over(partition_by=partition_by, order_by=order_by)


@udf(category="cs")
def cs_std(expr: pl.Expr, partition_by=None, order_by=None):
    return expr.std().over(partition_by=partition_by, order_by=order_by)


@udf(category="cs")
def cs_var(expr: pl.Expr, partition_by=None, order_by=None):
    return expr.var().over(partition_by=partition_by, order_by=order_by)


@udf(category="cs")
def cs_skew(expr: pl.Expr, partition_by=None, order_by=None):
    return expr.skew().over(partition_by=partition_by, order_by=order_by)


@udf(category="cs")
def cs_slope(left: pl.Expr, right: pl.Expr, partition_by=None, order_by=None):
    """Cross-section regression slope: corr(left, right) * std(left) / std(right)."""
    return (
        cs_corr(left, right, partition_by=partition_by, order_by=order_by)
        * cs_std(left, partition_by=partition_by, order_by=order_by)
        / cs_std(right, partition_by=partition_by, order_by=order_by)
    )


@udf(category="cs")
def cs_resid(left: pl.Expr, right: pl.Expr, partition_by=None, order_by=None):
    """Cross-section residual: left - slope(left, right) * right."""
    return left - cs_slope(left, right, partition_by=partition_by, order_by=order_by) * right


@udf(category="cs")
def cs_zscore(expr: pl.Expr, partition_by=None, order_by=None):
    """Cross-section z-score: (x - mean(x)) / std(x)."""
    return (expr - cs_mean(expr, partition_by=partition_by, order_by=order_by)) / cs_std(
        expr, partition_by=partition_by, order_by=order_by
    )


@udf(category="cs")
def cs_midby(expr: pl.Expr, *by: pl.Expr, partition_by=None, order_by=None):
    return expr.median().over(
        partition_by=(partition_by or []) + list(by),
        order_by=order_by,
    )


@udf(category="cs")
def cs_meanby(expr: pl.Expr, *by: pl.Expr, partition_by=None, order_by=None):
    return expr.mean().over(
        partition_by=(partition_by or []) + list(by),
        order_by=order_by,
    )


@udf(category="cs")
def cs_max(expr: pl.Expr, partition_by=None, order_by=None):
    return expr.max().over(partition_by=partition_by, order_by=order_by)


@udf(category="cs")
def cs_min(expr: pl.Expr, partition_by=None, order_by=None):
    return expr.min().over(partition_by=partition_by, order_by=order_by)


@udf(category="cs")
def cs_peakmax(expr: pl.Expr, partition_by=None, order_by=None):
    return expr.peak_max().over(partition_by=partition_by, order_by=order_by)


@udf(category="cs")
def cs_peakmin(expr: pl.Expr, partition_by=None, order_by=None):
    return expr.peak_min().over(partition_by=partition_by, order_by=order_by)
