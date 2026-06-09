"""
截面算子
"""

import polars as pl

from codestr.udf.registry import udf

over = dict(partition_by=["datetime"], order_by=["asset"])


def configure_over(
    *, partition_by: list[str] | None = None, order_by: list[str] | None = None
) -> None:
    """Update the default OVER window for all CS (cross-section) operators."""
    if partition_by is not None:
        over["partition_by"] = list(partition_by)
    if order_by is not None:
        over["order_by"] = list(order_by)


@udf(category="cs")
def cs_ufit(expr: pl.Expr):
    return (expr - expr.median().over(**over)).abs()


@udf(category="cs")
def cs_rank(expr: pl.Expr):
    return expr.rank().over(**over)


@udf(category="cs")
def cs_demean(expr: pl.Expr):
    return expr - expr.mean().over(**over)


@udf(category="cs")
def cs_mean(expr: pl.Expr):
    return expr.mean().over(**over)


@udf(category="cs")
def cs_mid(expr: pl.Expr):
    return expr.median().over(**over)


@udf(category="cs")
def cs_moderate(expr: pl.Expr):
    return (expr - expr.mean().over(**over)).abs()


@udf(category="cs")
def cs_qcut(expr: pl.Expr, n_bins=10):
    return (
        expr.qcut(n_bins, labels=[str(i) for i in range(1, n_bins + 1)], allow_duplicates=True)
        .over(**over)
        .cast(pl.Int32)
    )


@udf(category="cs")
def cs_ic(left: pl.Expr, right: pl.Expr):
    return pl.corr(left, right, method="spearman").over(**over)


@udf(category="cs")
def cs_corr(left: pl.Expr, right: pl.Expr):
    return pl.corr(left, right, method="pearson").over(**over)


@udf(category="cs")
def cs_std(expr: pl.Expr):
    return expr.std().over(**over)


@udf(category="cs")
def cs_var(expr: pl.Expr):
    return expr.var().over(**over)


@udf(category="cs")
def cs_skew(expr: pl.Expr):
    return expr.skew().over(**over)


@udf(category="cs")
def cs_slope(left: pl.Expr, right: pl.Expr):
    return cs_corr(left, right) * cs_std(left) / cs_std(right)


@udf(category="cs")
def cs_resid(left: pl.Expr, right: pl.Expr):
    return left - cs_slope(left, right) * right


@udf(category="cs")
def cs_zscore(expr: pl.Expr):
    return (expr - cs_mean(expr)) / cs_std(expr)


@udf(category="cs")
def cs_midby(expr: pl.Expr, *by: pl.Expr):
    return expr.median().over([*over, *by])


@udf(category="cs")
def cs_meanby(expr: pl.Expr, *by: pl.Expr):
    return expr.mean().over([*over, *by])


@udf(category="cs")
def cs_max(expr: pl.Expr):
    return expr.max().over(**over)


@udf(category="cs")
def cs_min(expr: pl.Expr):
    return expr.min().over(**over)


@udf(category="cs")
def cs_peakmax(expr: pl.Expr):
    return expr.peak_max().over(**over)


@udf(category="cs")
def cs_peakmin(expr: pl.Expr):
    return expr.peak_min().over(**over)
