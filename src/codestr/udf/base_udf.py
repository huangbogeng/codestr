import math

import numpy as np
import polars as pl

from codestr.udf.registry import udf

__all__ = [
    "abs",
    "add",
    "and_",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "arg_max",
    "arg_min",
    "between",
    "cast",
    "cbrt",
    "clip",
    "concat",
    "cos",
    "cosh",
    "cot",
    "cube",
    "degrees",
    "div",
    "entropy",
    "eq",
    "exp",
    "fib",
    "floordiv",
    "ge",
    "gt",
    "if_",
    "le",
    "log",
    "log1p",
    "lt",
    "max",
    "mean",
    "min",
    "mod",
    "mul",
    "neg",
    "neq",
    "not_",
    "null_type",
    "or_",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "sub",
    "sum",
    "tan",
    "tanh",
    "trunc",
]

"""
基础算子集合：一元、二元、三元算子及 Polars 表达式，排除含未来信息的算子。
"""
# ======================== 一元算子 ========================


@udf(category="math")
def not_(expr: pl.Expr):
    return ~expr


@udf(category="math")
def neg(expr: pl.Expr):
    return -expr


@udf(category="math")
def abs(expr: pl.Expr):
    return expr.abs()


@udf(category="math")
def log(expr: pl.Expr, base=math.e):
    return expr.log(base=base)


@udf(category="math")
def sqrt(expr: pl.Expr):
    return expr.sqrt()


@udf(category="math")
def square(expr: pl.Expr):
    return expr**2


@udf(category="math")
def cube(expr: pl.Expr):
    return expr**3


@udf(category="math")
def cbrt(expr: pl.Expr):
    return expr ** (1 / 3)


@udf(category="math")
def sin(expr: pl.Expr):
    return expr.sin()


@udf(category="math")
def sinh(expr: pl.Expr):
    return expr.sinh()


@udf(category="math")
def arcsin(expr: pl.Expr):
    return expr.arcsin()


@udf(category="math")
def arcsinh(expr: pl.Expr):
    return expr.arcsinh()


@udf(category="math")
def cos(expr: pl.Expr):
    return expr.cos()


@udf(category="math")
def cosh(expr: pl.Expr):
    return expr.cosh()


@udf(category="math")
def arccos(expr: pl.Expr):
    return expr.arccos()


@udf(category="math")
def arccosh(expr: pl.Expr):
    return expr.arccosh()


@udf(category="math")
def tan(expr: pl.Expr):
    return expr.tan()


@udf(category="math")
def tanh(expr: pl.Expr):
    return expr.tanh()


@udf(category="math")
def arctan(expr: pl.Expr):
    return expr.arctan()


@udf(category="math")
def arctanh(expr: pl.Expr):
    return expr.arctanh()


@udf(category="math")
def sign(expr: pl.Expr):
    return expr.sign()


@udf(category="math")
def sigmoid(expr: pl.Expr):
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    return 1 / (1 + (-expr).exp())


@udf(category="math")
def cot(expr: pl.Expr):
    return expr.cot()


@udf(category="math")
def degrees(expr: pl.Expr):
    return expr.degrees()


@udf(category="math")
def entropy(expr: pl.Expr):
    return expr.entropy()


@udf(category="math")
def exp(expr: pl.Expr):
    return expr.exp()


@udf(category="math")
def log1p(expr: pl.Expr):
    return expr.log1p()


@udf(category="math")
def clip(expr: pl.Expr, lower_bound=-np.inf, upper_bound=np.inf):
    """Clip values to [lower_bound, upper_bound] (inclusive)."""
    return expr.clip(lower_bound, upper_bound)


@udf(category="math")
def trunc(
    expr: pl.Expr,
    lower_bound=-np.inf,
    upper_bound=np.inf,
    left_closed=True,
    right_closed=True,
):
    """Truncate values to a range, setting out-of-bound values to null.

    Differs from ``clip``: outliers become None rather than being clamped to bounds.
    """
    lower = expr >= lower_bound if left_closed else expr > lower_bound
    upper = expr <= upper_bound if right_closed else expr < upper_bound
    return pl.when(lower, upper).then(expr).otherwise(None)


@udf(category="math")
def between(
    expr: pl.Expr,
    lower_bound=-np.inf,
    upper_bound=np.inf,
    close="both",
):
    """Return a boolean mask: whether each value falls within [lower, upper]."""
    return expr.is_between(lower_bound, upper_bound, close)


@udf(category="math")
def cast(expr: pl.Expr, dtype: str):
    """Cast an expression to a target dtype: ``"int"``, ``"float"``, ``"cat"``, or ``"str"``."""
    map = {
        "int": pl.Int64,
        "float": pl.Float64,
        "cat": pl.Catalog,
        "str": pl.String,
    }

    p = map.get(dtype)
    if p is None:
        raise ValueError(f"{expr}, {dtype} is not a valid type, must in {map.keys()}")

    return expr.cast(p)


@udf(category="math")
def concat(*exprs: pl.Expr):
    return pl.concat_list(*exprs)


@udf(category="math")
def null_type(expr: pl.Expr):
    return pl.lit(None)


# ======================== 二元算子 ========================


@udf(category="math")
def add(left: pl.Expr, right: pl.Expr):
    return left + right


@udf(category="math")
def sub(left: pl.Expr, right: pl.Expr):
    return left - right


@udf(category="math")
def mul(left: pl.Expr, right: pl.Expr):
    return left * right


@udf(category="math")
def div(left: pl.Expr, right: pl.Expr):
    return left / right


@udf(category="math")
def floordiv(left: pl.Expr, right: pl.Expr):
    return left // right


@udf(category="math")
def mod(left: pl.Expr, right: pl.Expr):
    return left % right


@udf(category="math")
def lt(left: pl.Expr, right: pl.Expr):
    return left < right


@udf(category="math")
def le(left: pl.Expr, right: pl.Expr):
    return left <= right


@udf(category="math")
def gt(left: pl.Expr, right: pl.Expr):
    return left > right


@udf(category="math")
def ge(left: pl.Expr, right: pl.Expr):
    return left >= right


@udf(category="math")
def eq(left: pl.Expr, right: pl.Expr):
    return left == right


@udf(category="math")
def neq(left: pl.Expr, right: pl.Expr):
    return left != right


@udf(category="math")
def and_(left: pl.Expr, right: pl.Expr):
    return left & right


@udf(category="math")
def or_(left: pl.Expr, right: pl.Expr):
    return left | right


@udf(category="math")
def max(*exprs: pl.Expr):
    return pl.max_horizontal(*exprs)


@udf(category="math")
def min(*exprs: pl.Expr):
    return pl.min_horizontal(*exprs)


@udf(category="math")
def sum(*exprs: pl.Expr):
    return pl.sum_horizontal(*exprs)


@udf(category="math")
def arg_max(*exprs: pl.Expr):
    """Return the index (0-based) of the maximum value across expressions."""
    return pl.concat_list(*exprs).list.arg_max()


@udf(category="math")
def arg_min(*exprs: pl.Expr):
    """Return the index (0-based) of the minimum value across expressions."""
    return pl.concat_list(*exprs).list.arg_min()


@udf(category="math")
def mean(*exprs: pl.Expr):
    return pl.mean_horizontal(*exprs)


# ======================== 三元 ========================


@udf(category="math")
def if_(cond: pl.Expr, body: pl.Expr, or_else: pl.Expr):
    """Ternary conditional: cond ? body : or_else."""
    return pl.when(cond).then(body).otherwise(or_else)


@udf(category="math")
def fib(high: pl.Expr, low: pl.Expr, ratio: float = 0.618):
    """
    计算裴波那契回调比率
    ratio: 0.236 | 0.382 | 0.618 等黄金分割比例
    """
    return low + (high - low) * ratio
