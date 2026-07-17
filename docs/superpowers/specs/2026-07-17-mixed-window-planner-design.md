# Mixed-Window Planner And Lowering Design

## Problem

CodeStr currently compiles one AST into one nested Polars expression. This is
incorrect when a time-series window contains a cross-section window, or the
reverse.

For example:

```text
ts_mean(cs_moderate(x), 60)
```

is compiled into the equivalent of:

```python
(
    x - x.mean().over(partition_by=["datetime"], order_by=["asset"])
).abs().rolling_mean(60).over(
    partition_by=["asset"],
    order_by=["datetime"],
)
```

Polars evaluates the child expression inside the outer asset window context.
The inner date partition then contains only the current asset, so
`x - mean(x)` becomes zero. The expression is valid to Polars and therefore
returns silently incorrect results instead of raising an error.

The same issue exists in the inverse direction:

```text
cs_mean(ts_mean(x, 60))
```

Same-domain nesting, such as `ts_mean(ts_mean(x, 5), 20)` or
`cs_rank(cs_moderate(x))`, matches explicit staged execution and remains
supported.

## Goals

- Make mixed TS/CS expressions safe by construction.
- Automatically lower mixed-window expressions passed to `CodeStr.sql()` into
  sequential lazy projection stages.
- Reject mixed-window expressions passed to the pure `CodeStr.compile()` API,
  because one `pl.Expr` cannot represent a required projection boundary.
- Detect both `ts -> cs` and `cs -> ts`, including window calls hidden under
  ordinary math calls or keyword arguments.
- Preserve same-domain window nesting.
- Preserve every existing operator signature and numerical definition.
- Preserve stateful expression-cache behavior and output aliases.

## Non-Goals

- Do not change any TS, CS, or math operator implementation.
- Do not infer window behavior by function-name prefixes.
- Do not eagerly collect intermediate frames.
- Do not change `CodeStr.compile()` to return a frame or a new public type.
- Do not add a public planner API in the first implementation.
- Do not expose internal intermediate columns in `sql()` results.
- Do not inspect arbitrary `category="user"` function bodies for hidden Polars
  window expressions.
- Do not optimize stage grouping beyond what is required for correctness.

## Window-Domain Analysis

Window identity comes from `UDFMeta.category`:

| Category | Domain |
|---|---|
| `ts` | time-series |
| `cs` | cross-section |
| `math` | transparent/non-window |
| `user` | opaque/non-window |

The analyzer walks positional arguments and `KeywordArg.value` nodes. Math
calls are transparent, so the nearest window ancestor remains active while
their children are inspected.

A mixed-window boundary exists when a window call has a nearest window
ancestor with a different domain.

Examples:

| Expression | Mixed boundary |
|---|---|
| `ts_mean(cs_moderate(x), 60)` | `ts -> cs` |
| `ts_mean(abs(cs_moderate(x)), 60)` | `ts -> cs` |
| `cs_mean(ts_mean(x, 60))` | `cs -> ts` |
| `ts_mean(ts_mean(x, 5), 20)` | none |
| `cs_rank(cs_moderate(x))` | none |
| `cs_rank(x) + ts_mean(y, 20)` | none; windows are siblings |

The analyzer reports the outer and inner call names so errors can identify the
unsafe path.

## Planner Model

Add an internal planner module with immutable plan records:

```python
@dataclass(frozen=True)
class PlanStep:
    node: ExprNode
    output_name: str
    internal: bool


@dataclass(frozen=True)
class ExecutionPlan:
    steps: tuple[PlanStep, ...]
    output_name: str
```

`PlanStep.node` is a lowered AST that can safely compile to one Polars
expression. Steps are stored in topological order. The final step preserves the
root expression's public alias; earlier steps use internal column names.

The existing AST remains the source language. The plan is an execution IR, not
a replacement parser tree.

## Lowering Algorithm

Lowering recursively tracks the nearest active window domain.

When a window call has a different domain from its nearest window ancestor:

1. Recursively lower any mixed boundaries inside that child call.
2. Add the rewritten child call as an internal `PlanStep`.
3. Replace the child call in its parent AST with `Column(step.output_name)`.
4. Continue lowering the parent expression.

For:

```text
ts_mean(abs(cs_moderate(x)), 60)
```

the result is:

```text
stage 1: __codestr_<fingerprint> = cs_moderate(x)
stage 2: result = ts_mean(abs(__codestr_<fingerprint>), 60)
```

For alternating domains such as `ts(cs(ts(x)))`, each boundary becomes a
separate topological stage.

Independent extracted subexpressions may initially use separate stages.
Grouping independent projections into one `with_columns` call is an optional
future optimization.

## Internal Column Identity

Internal names must be deterministic, collision-resistant, and independent of
Python's randomized `hash()`.

Create a canonical structural fingerprint from:

- node type;
- call function name;
- positional argument order;
- keyword argument names and values;
- column names;
- literal Python type and value;
- aliases excluded from identity.

Use a truncated SHA-256 digest:

```text
__codestr_<16 hex characters>
```

Before using an internal name, check the input/lazy schema. If a user column
already has that name and is not mapped to the same AST, append a deterministic
numeric suffix.

Repeated structural subexpressions in one plan share one step. Existing
`ExprNode` equality and hashing provide in-memory deduplication; the stable
fingerprint provides the column name.

## Compilation Contracts

### Pure Compiler

The module-level compiler and `CodeStr.compile()` continue returning one
`pl.Expr`.

Before recursive compilation, they run mixed-window analysis. If a boundary is
present, compilation raises:

```text
Mixed window domains require a multi-stage plan: ts_mean -> cs_moderate.
Use CodeStr.sql() for automatic lowering.
```

Same-domain nested windows continue compiling normally.

### Stateful SQL

`CodeStr.sql()` parses the expression, checks the root cache, lowers an
uncached expression to an `ExecutionPlan`, and executes each step with a
separate lazy `with_columns` call:

```python
for step in plan.steps:
    lazy_frame = lazy_frame.with_columns(
        compile_single_stage(step.node).alias(step.output_name)
    )
```

Sequential `with_columns` nodes create the required logical projection
boundary. They do not force eager collection, and Polars retains both stages in
the optimized lazy plan because the outer stage depends on the inner column.

Only the requested public aliases are selected into the returned frame.

## Engine And Cache Integration

The persistent cache continues to key the final result by the original,
unlowered AST. A repeated expression with a new public alias can therefore
reuse the final cached column without replanning.

Within one plan:

- extracted subexpressions are deduplicated by AST identity;
- the final cache entry maps the original root AST to its public alias;
- internal columns remain available in the accumulated lazy graph but do not
  appear in the returned result;
- `cover=True` recomputes all plan steps;
- a failure restores the expression's pre-plan lazy graph and pending cache
  state, so a partially executed plan cannot leak into later expressions.

Internal cross-query cache entries are optional for the first implementation.
Correct final-expression caching and within-plan deduplication are required.

The `lazy=True` path rolls back the engine graph after returning a LazyFrame.
It must not persist cache entries that refer to internal columns absent from
the retained engine graph.

## Validation And Error Handling

Unknown functions and invalid arguments continue to use existing parser and
compiler error boundaries.

Planner-specific errors include:

- a mixed expression sent to the pure compiler;
- an internal column-name collision that cannot be resolved;
- an invalid plan dependency;
- a stage that still contains mixed window domains after lowering.

The last check is an invariant assertion before stage compilation and protects
against incomplete lowering.

`CodeStr.check_expr()` treats a mixed-window expression as structurally valid,
because `CodeStr.sql()` supports it. Pure-compilation limitations are reported
only by `compile()`.

## Tests

Add focused coverage for:

- `ts_mean(cs_moderate(x), 60)` matching explicit two-stage execution;
- `cs_mean(ts_mean(x, 2, min_samples=1))` matching explicit two-stage execution;
- mixed windows hidden under math calls;
- alternating domain chains producing topologically ordered stages;
- same-domain TS nesting remaining a single expression;
- same-domain CS nesting remaining a single expression;
- sibling TS and CS windows not being classified as nested;
- pure `compile()` rejecting both mixed directions with the unsafe path;
- explicit two-expression SQL remaining compatible;
- eager and lazy SQL results matching;
- custom `partition_by` and `order_by` configurations;
- root cache reuse with a different output alias;
- repeated mixed subexpressions being lowered once per plan;
- `cover=True` recomputation;
- rollback after a failing stage;
- internal columns not appearing in returned results;
- stable internal names distinguishing integer and floating-point literals;
- full regression coverage for existing non-window and same-window expressions.

Verification includes:

```text
uv lock --check
uv sync --locked --extra test
uv run --locked pytest tests/ -q
uvx ruff check src/ tests/
uvx ruff format --check src/ tests/
uv build
```

## Rollout

Implementation proceeds in correctness-first checkpoints:

1. Add mixed-window analysis and pure-compiler fail-fast tests.
2. Add planner IR and lowering unit tests.
3. Integrate staged execution into `CodeStr.sql()`.
4. Integrate cache, rollback, eager, and lazy behavior.
5. Add documentation and full verification.

No operator release or specification change is required. The behavior change
belongs entirely to expression planning, lowering, and execution.
