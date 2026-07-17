# Time-Series `min_samples` Design

## Scope

Expose Polars' `min_samples` option through every CodeStr time-series operator
that is backed by a rolling or exponentially weighted window.

This change covers:

- `ts_max`
- `ts_min`
- `ts_mean`
- `ts_std`
- `ts_skew`
- `ts_kurt`
- `ts_sum`
- `ts_var`
- `ts_mid`
- `ts_mad`
- `ts_ema`

`ts_delay` and `ts_delta` are excluded because their Polars implementations do
not have a `min_samples` parameter.

## API Contract

`min_samples` is an optional keyword argument in the DSL:

```text
ts_mean(close, 20, min_samples=5)
ts_std(close, 20, min_samples=10)
ts_ema(close, 10, min_samples=3)
```

The default follows the corresponding Polars 1.42.1 API rather than imposing
one CodeStr-wide value:

| CodeStr operators | Default | Polars behavior |
|---|---:|---|
| Rolling operators | `None` | Require the full rolling window |
| `ts_ema` | `1` | Produce a value after one valid observation |

Omitting `min_samples` therefore preserves all existing CodeStr results.

The Python UDF signatures add `min_samples` after the existing
`partition_by`/`order_by` parameters as a keyword-only option. This keeps the
existing positional parameter order available to direct Python callers while
allowing the compiler to pass DSL keyword arguments by name.

Examples:

```python
def ts_mean(
    expr: pl.Expr,
    windows,
    partition_by=None,
    order_by=None,
    *,
    min_samples=None,
):
    return expr.rolling_mean(
        windows,
        min_samples=min_samples,
    ).over(
        partition_by=partition_by,
        order_by=order_by,
    )


def ts_ema(
    expr: pl.Expr,
    span: int,
    partition_by=None,
    order_by=None,
    *,
    min_samples: int = 1,
):
    return expr.ewm_mean(
        span=span,
        adjust=False,
        min_samples=min_samples,
    ).over(
        partition_by=partition_by,
        order_by=order_by,
    )
```

## Validation

An explicit `min_samples` value must be a positive integer. Boolean values are
rejected even though Python treats `bool` as a subclass of `int`.

Rolling operators accept `None` internally as their default. The DSL does not
need a `None` literal because callers obtain the default by omitting the
keyword.

Invalid values are raised from the UDF and wrapped in the existing
`CompileError` boundary:

```text
min_samples must be a positive integer
```

Polars remains responsible for validating relationships such as
`min_samples <= windows`; CodeStr does not duplicate every backend constraint.

## Operator Semantics

Each rolling operator passes the validated value directly to its corresponding
Polars method:

```python
expr.rolling_mean(windows, min_samples=min_samples)
```

`ts_ema` keeps its established recursive EMA settings and exposes only the
existing `min_samples` constant:

```python
expr.ewm_mean(
    span=span,
    adjust=False,
    min_samples=min_samples,
)
```

`ts_mad` contains two rolling median stages. The same `min_samples` value is
passed to both stages so the operator has one predictable threshold:

```python
inner = expr.rolling_median(windows, min_samples=min_samples)
result = (
    expr
    - inner.over(
        partition_by=partition_by,
        order_by=order_by,
    )
).abs().rolling_median(
    windows,
    min_samples=min_samples,
).over(
    partition_by=partition_by,
    order_by=order_by,
)
```

All operators continue to use the compiler-injected `partition_by` and
`order_by` window configuration.

## Cache And Parsing

No parser or AST changes are required. The existing immutable `KeywordArg`
representation already supports numeric keyword values, structural hashing,
canonical rendering, and stateful `CodeStr.sql()` cache reuse.

For example, repeated calls to:

```text
ts_mean(close, 20, min_samples=5)
```

share the same structural cache identity, independent of their output aliases.

## Tests

Add coverage for:

- omitted rolling `min_samples` matching Polars' `None` default;
- omitted EMA `min_samples` matching Polars' `1` default;
- explicit thresholds on every covered operator;
- early rolling output when `min_samples` is below `windows`;
- grouped and ordered multi-asset execution;
- null handling delegated to Polars;
- `ts_mad` applying one threshold to both rolling stages;
- rejection of zero, negative, floating-point, and boolean values;
- DSL keyword compilation and stateful cache reuse;
- unchanged behavior for `ts_delay` and `ts_delta`.

Run the full pytest suite, Ruff lint and format checks, `uv lock --check`, and
`uv build` after implementation.

## Documentation

Update the README operator examples and the time-series operator reference to
document:

- the `min_samples` keyword;
- rolling default `None`;
- EMA default `1`;
- the fact that defaults follow Polars and preserve existing results.

## Non-Goals

- Do not expose `weights`, `center`, `ddof`, `bias`, `fisher`, `adjust`, or
  `ignore_nulls`.
- Do not add boolean, string, list, or `None` literals to the DSL.
- Do not change rolling-window defaults.
- Do not change EMA numerical defaults.
- Do not modify window injection, data loading, `yd`, or TA2 expressions.
- Do not add `d_*` compatibility aliases.
