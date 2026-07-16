# EMA And Keyword Arguments Design

## Scope

Add a standard time-series EMA operator and make keyword arguments safe in the stateful
`CodeStr.sql()` cache. Raise the Polars dependency floor so the implementation can use the current
Polars API directly.

This design supersedes the dependency-only implementation scope in
`2026-07-16-polars-minimum-version-design.md`.

## Polars Dependency

Set the published dependency to `polars>=1.42.1` and refresh `uv.lock` to Polars 1.42.1. Do not add
an upper bound.

## `ts_ema`

Add the public DSL operator:

```text
ts_ema(expr, span)
```

The operator is registered in the `ts` category, so the compiler injects the current CodeStr
instance's `partition_by` and `order_by` configuration. Its Polars implementation is:

```python
expr.ewm_mean(
    span=span,
    adjust=False,
    min_samples=1,
).over(
    partition_by=partition_by,
    order_by=order_by,
)
```

`span` must be a positive integer. Invalid values fail during compilation with the existing
`CompileError` wrapping behavior. Other null handling follows Polars defaults.

The operator must work as a top-level result, a nested expression, and an intermediate column used
by later expressions in the same `CodeStr.sql()` call.

## Immutable Keyword Arguments

The parser currently stores each keyword argument as a mutable `dict` inside `Call.args`. `Call` is
used as a dictionary key by the stateful expression cache, so this representation violates the AST's
hashability and immutability requirements.

Introduce an immutable internal syntax value:

```python
@dataclass(frozen=True)
class KeywordArg:
    name: str
    value: ExprNode
```

The parser emits `KeywordArg` instead of `dict`. The compiler resolves `KeywordArg.value` and passes
it under `KeywordArg.name`. Call rendering displays normal DSL syntax such as
`clip(close, lower_bound=0)`. `Call` equality and hashing continue to ignore aliases and use the
ordered argument tuple; keyword argument order is therefore preserved as written.

Syntax analysis helpers treat `KeywordArg` as a transparent wrapper around its value. Depth, node
count, descendants, and RPN validation continue to count the underlying expression without adding a
synthetic AST level.

This representation is preferred over freezing a `dict` only at hash time because no mutable object
remains inside a cache key.

## Tests

Add test coverage for:

- EMA span 1, constant values, changing values, nulls, and invalid spans;
- independent assets and explicit ordering;
- nested EMA and same-call intermediate-expression reuse;
- immutable, hashable keyword arguments and correct DSL rendering;
- keyword arguments through pure compilation and stateful `CodeStr.sql()`;
- repeated stateful queries hitting the expression cache without errors;
- syntax analysis helpers on expressions containing keyword arguments.

Run the complete pytest suite, Ruff checks, formatting checks, and package build after implementation.

## Documentation

Add `ts_ema` to the README operator list and the time-series operator reference, including recursive
EMA semantics and the positive-integer `span` contract.

## Non-Goals

- Do not add `d_std`, `d_mean`, `d_ref`, `d_ewmmean`, or any other compatibility aliases.
- Do not modify TA2 expressions, `yd`, datacenter integrations, or data-loading code.
- Do not add TA2 data or numerical comparison tests.
- Do not redesign positional argument parsing or unrelated cache behavior.
