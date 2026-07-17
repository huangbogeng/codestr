# Mixed-Window Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `CodeStr.sql()` automatically lower mixed TS/CS window expressions into correct sequential lazy projection stages while making pure `compile()` reject expressions that cannot be represented by one `pl.Expr`.

**Architecture:** Add a focused `planner.py` module that analyzes window-domain ancestry, produces an immutable `ExecutionPlan`, and rewrites cross-domain child windows to deterministic internal columns. Keep the existing recursive compiler as the single-stage compiler; `CodeStr.sql()` invokes the planner and executes multi-stage plans, while `CodeStr.compile()` remains a pure `pl.Expr` API and fails fast for mixed domains.

**Tech Stack:** Python 3.10+, Polars 1.42.1, immutable CodeStr AST nodes, pytest, Ruff, uv

---

## File Map

| File | Responsibility |
|---|---|
| `src/codestr/planner.py` | Window-domain analysis, structural fingerprints, plan IR, and AST lowering |
| `src/codestr/compiler.py` | Single-stage AST-to-Polars compiler and mixed-window fail-fast boundary |
| `src/codestr/engine.py` | Stateful plan execution, query ordering, cache integration, and rollback |
| `tests/test_planner.py` | Analyzer, lowering, naming, deduplication, and plan invariant tests |
| `tests/test_compiler.py` | Pure compiler rejection and same-domain compatibility |
| `tests/test_engine.py` | Eager/lazy execution, same-query aliases, cache, cover, and rollback |
| `README.md` | User-facing mixed-window SQL example and pure compile limitation |
| `docs/operators.md` | Planner semantics and explicit staging behavior |

The parser, syntax node definitions, registry metadata, and TS/CS operator files
remain unchanged.

### Task 1: Add Window-Domain Analysis

**Files:**
- Create: `src/codestr/planner.py`
- Create: `tests/test_planner.py`

- [ ] **Step 1: Write failing analyzer tests**

Create `tests/test_planner.py`:

```python
"""Tests for mixed-window analysis and lowering."""

import pytest

from codestr.parser import parse
from codestr.planner import (
    MixedWindowBoundary,
    find_mixed_window_boundary,
)
from codestr.udf.registry import UDFRegistry


@pytest.fixture
def registry():
    return UDFRegistry.get_instance()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "ts_mean(cs_moderate(x), 60)",
            MixedWindowBoundary("ts_mean", "ts", "cs_moderate", "cs"),
        ),
        (
            "cs_mean(ts_mean(x, 60))",
            MixedWindowBoundary("cs_mean", "cs", "ts_mean", "ts"),
        ),
        (
            "ts_mean(abs(cs_moderate(x)), 60)",
            MixedWindowBoundary("ts_mean", "ts", "cs_moderate", "cs"),
        ),
        (
            "ts_mean(add(left=cs_moderate(x), right=y), 60)",
            MixedWindowBoundary("ts_mean", "ts", "cs_moderate", "cs"),
        ),
    ],
)
def test_find_mixed_window_boundary(source, expected, registry):
    assert find_mixed_window_boundary(parse(source), registry) == expected


@pytest.mark.parametrize(
    "source",
    [
        "ts_mean(ts_mean(x, 5), 20)",
        "cs_rank(cs_moderate(x))",
        "cs_rank(x) + ts_mean(y, 20)",
        "abs(x) + log(y)",
    ],
)
def test_same_domain_and_sibling_windows_are_not_mixed(source, registry):
    assert find_mixed_window_boundary(parse(source), registry) is None


def test_unknown_call_is_opaque_to_window_analysis(registry):
    node = parse("ts_mean(not_registered(cs_moderate(x)), 60)")

    assert find_mixed_window_boundary(node, registry) is None
```

- [ ] **Step 2: Run analyzer tests to verify RED**

Run:

```bash
uv run --locked pytest tests/test_planner.py -q
```

Expected: collection fails because `codestr.planner` does not exist.

- [ ] **Step 3: Implement immutable boundary analysis**

Create `src/codestr/planner.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from codestr.syntax import Call, ExprNode, KeywordArg
from codestr.udf.registry import UDFRegistry

WindowDomain = Literal["ts", "cs"]


@dataclass(frozen=True)
class WindowRef:
    call_name: str
    domain: WindowDomain


@dataclass(frozen=True)
class MixedWindowBoundary:
    outer_call: str
    outer_domain: WindowDomain
    inner_call: str
    inner_domain: WindowDomain


def _arg_value(arg):
    return arg.value if isinstance(arg, KeywordArg) else arg


def _call_domain(call: Call, registry: UDFRegistry) -> WindowDomain | None:
    meta = registry.get(call.fn_name)
    if meta is None or meta.category not in {"ts", "cs"}:
        return None
    return meta.category


def find_mixed_window_boundary(
    node: ExprNode,
    registry: UDFRegistry,
) -> MixedWindowBoundary | None:
    def visit(
        current: ExprNode,
        active_window: WindowRef | None,
    ) -> MixedWindowBoundary | None:
        if not isinstance(current, Call):
            return None

        # Preserve the compiler's existing Unknown function error boundary.
        if registry.get(current.fn_name) is None:
            return None

        domain = _call_domain(current, registry)
        if (
            domain is not None
            and active_window is not None
            and domain != active_window.domain
        ):
            return MixedWindowBoundary(
                outer_call=active_window.call_name,
                outer_domain=active_window.domain,
                inner_call=current.fn_name,
                inner_domain=domain,
            )

        next_window = (
            WindowRef(current.fn_name, domain)
            if domain is not None
            else active_window
        )
        for arg in current.args:
            boundary = visit(_arg_value(arg), next_window)
            if boundary is not None:
                return boundary
        return None

    return visit(node, None)
```

- [ ] **Step 4: Run analyzer tests to verify GREEN**

Run:

```bash
uv run --locked pytest tests/test_planner.py -q
```

Expected: `9 passed`.

- [ ] **Step 5: Run planner lint and format**

Run:

```bash
uvx ruff format src/codestr/planner.py tests/test_planner.py
uvx ruff check src/codestr/planner.py tests/test_planner.py
git diff --check
```

Expected: Ruff passes and `git diff --check` has no output.

- [ ] **Step 6: Commit analyzer**

```bash
git add src/codestr/planner.py tests/test_planner.py
git commit -m "Analyze mixed window domains"
```

### Task 2: Make Pure Compilation Fail Fast

**Files:**
- Modify: `src/codestr/compiler.py:19-44`
- Modify: `tests/test_compiler.py`
- Modify: `tests/test_engine.py`

- [ ] **Step 1: Write failing pure-compiler tests**

Append to `tests/test_compiler.py`:

```python
class TestMixedWindowCompile:
    @pytest.mark.parametrize(
        ("source", "path"),
        [
            ("ts_mean(cs_moderate(x), 60)", "ts_mean -> cs_moderate"),
            ("cs_mean(ts_mean(x, 60))", "cs_mean -> ts_mean"),
        ],
    )
    def test_pure_compile_rejects_mixed_window_domains(self, source, path):
        with pytest.raises(CompileError, match=path):
            ast_compile(parse(source))

    @pytest.mark.parametrize(
        "source",
        [
            "ts_mean(ts_mean(x, 2, min_samples=1), 2, min_samples=1)",
            "cs_rank(cs_moderate(x))",
        ],
    )
    def test_pure_compile_keeps_same_domain_nesting(self, source):
        assert isinstance(ast_compile(parse(source)), pl.Expr)

    def test_unknown_function_remains_the_primary_error(self):
        source = "ts_mean(not_registered(cs_moderate(x)), 60)"

        with pytest.raises(CompileError, match="Unknown function: not_registered"):
            ast_compile(parse(source))
```

Move the existing local `CompileError` import in `TestCompileErrors` to the
module imports:

```python
from codestr.errors import CompileError
```

Add to `TestCheckExpr` in `tests/test_engine.py`:

```python
    def test_mixed_window_is_structurally_valid(self, sample_df):
        cs = CodeStr(sample_df)

        result = cs.check_expr("ts_mean(cs_moderate(close), 60)")

        assert result["valid"] is True
        assert result["reasons"] == []
```

- [ ] **Step 2: Run compiler tests to verify RED**

Run:

```bash
uv run --locked pytest \
  tests/test_compiler.py::TestMixedWindowCompile \
  tests/test_engine.py::TestCheckExpr::test_mixed_window_is_structurally_valid \
  -q
```

Expected: the two mixed-window compile cases do not raise; the unknown-function
and structural-validation cases already pass.

- [ ] **Step 3: Add the single-expression guard**

Modify `src/codestr/compiler.py` imports:

```python
from codestr.planner import find_mixed_window_boundary
```

Add this block in `compile()` after registry initialization and before
`_compile()`:

```python
    boundary = find_mixed_window_boundary(node, registry)
    if boundary is not None:
        raise CompileError(
            "Mixed window domains require a multi-stage plan: "
            f"{boundary.outer_call} -> {boundary.inner_call}. "
            "Use CodeStr.sql() for automatic lowering."
        )
```

Do not add the guard to `_compile()`. `_compile()` remains the recursive
single-stage implementation used below the public boundary.

- [ ] **Step 4: Run compiler tests to verify GREEN**

Run:

```bash
uv run --locked pytest \
  tests/test_compiler.py \
  tests/test_engine.py::TestCheckExpr::test_mixed_window_is_structurally_valid \
  -q
```

Expected: all compiler tests pass.

- [ ] **Step 5: Format and lint compiler changes**

Run:

```bash
uvx ruff format src/codestr/compiler.py tests/test_compiler.py tests/test_engine.py
uvx ruff check src/codestr/compiler.py tests/test_compiler.py tests/test_engine.py
git diff --check
```

Expected: Ruff passes and `git diff --check` has no output.

- [ ] **Step 6: Commit pure compile guard**

```bash
git add src/codestr/compiler.py tests/test_compiler.py tests/test_engine.py
git commit -m "Reject mixed windows in pure compilation"
```

### Task 3: Add Plan IR, Fingerprints, And Lowering

**Files:**
- Modify: `src/codestr/planner.py`
- Modify: `tests/test_planner.py`

- [ ] **Step 1: Write failing lowering tests**

Replace the CodeStr imports in `tests/test_planner.py` with:

```python
from codestr.errors import CompileError
from codestr.parser import parse
from codestr.planner import (
    ExecutionPlan,
    MixedWindowBoundary,
    PlanStep,
    _validate_step_dependencies,
    build_execution_plan,
    find_mixed_window_boundary,
)
from codestr.syntax import Call, Column
from codestr.udf.registry import UDFRegistry
```

Append:

```python
class TestExecutionPlan:
    def test_lowers_ts_over_cs_to_two_stages(self, registry):
        node = parse("ts_mean(cs_moderate(x), 60)")

        plan = build_execution_plan(node, registry)

        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 2
        internal, final = plan.steps
        assert internal.internal is True
        assert str(internal.node) == "cs_moderate(x)"
        assert internal.output_name.startswith("__codestr_")
        assert final.internal is False
        assert final.output_name == node.alias
        assert isinstance(final.node, Call)
        assert final.node.args[0] == Column(internal.output_name)

    def test_lowers_alternating_domains_in_topological_order(self, registry):
        node = parse(
            "ts_mean("
            "cs_mean(ts_mean(x, 2, min_samples=1)), "
            "2, min_samples=1"
            ")"
        )

        plan = build_execution_plan(node, registry)

        assert len(plan.steps) == 3
        assert [step.internal for step in plan.steps] == [True, True, False]
        for step in plan.steps:
            assert find_mixed_window_boundary(step.node, registry) is None

    def test_deduplicates_repeated_mixed_subexpression(self, registry):
        node = parse(
            "ts_mean("
            "cs_moderate(x) + cs_moderate(x), "
            "2, min_samples=1"
            ")"
        )

        plan = build_execution_plan(node, registry)

        assert len(plan.steps) == 2
        internal_name = plan.steps[0].output_name
        final = plan.steps[-1].node
        add_call = final.args[0]
        assert isinstance(add_call, Call)
        assert add_call.args == (Column(internal_name), Column(internal_name))

    def test_single_domain_plan_has_one_public_step(self, registry):
        node = parse("ts_mean(ts_mean(x, 2), 3)")

        plan = build_execution_plan(node, registry)

        assert len(plan.steps) == 1
        assert plan.is_single_stage
        assert plan.steps[0].node == node
        assert plan.steps[0].output_name == node.alias
        assert plan.steps[0].internal is False

    def test_internal_name_avoids_existing_column(self, registry):
        node = parse("ts_mean(cs_moderate(x), 2)")
        first = build_execution_plan(node, registry)
        occupied = first.steps[0].output_name

        second = build_execution_plan(
            node,
            registry,
            existing_columns={occupied},
        )

        assert second.steps[0].output_name == f"{occupied}_1"

    def test_internal_names_are_stable_and_preserve_literal_type(self, registry):
        int_node = parse(
            "ts_mean(cs_moderate(x + 1), 2, min_samples=1)"
        )
        float_node = parse(
            "ts_mean(cs_moderate(x + 1.0), 2, min_samples=1)"
        )

        first = build_execution_plan(int_node, registry)
        repeated = build_execution_plan(int_node, registry)
        float_plan = build_execution_plan(float_node, registry)

        assert first.steps[0].output_name == repeated.steps[0].output_name
        assert first.steps[0].output_name != float_plan.steps[0].output_name

    def test_unknown_call_is_not_partially_lowered(self, registry):
        node = parse("ts_mean(not_registered(cs_moderate(x)), 60)")

        plan = build_execution_plan(node, registry)

        assert plan.is_single_stage
        assert plan.steps[0].node == node

    def test_rejects_forward_internal_dependency(self):
        steps = (
            PlanStep(
                node=Column("__codestr_later"),
                output_name="result",
                internal=False,
            ),
            PlanStep(
                node=Column("x"),
                output_name="__codestr_later",
                internal=True,
            ),
        )

        with pytest.raises(CompileError, match="Invalid plan dependency"):
            _validate_step_dependencies(steps)
```

- [ ] **Step 2: Run lowering tests to verify RED**

Run:

```bash
uv run --locked pytest tests/test_planner.py::TestExecutionPlan -q
```

Expected: import errors for missing plan types/functions.

- [ ] **Step 3: Add plan records and structural fingerprinting**

Replace the import block in `src/codestr/planner.py` with:

```python
from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

from codestr.errors import CompileError
from codestr.syntax import Call, Column, ExprNode, KeywordArg, Literal as LiteralNode
from codestr.udf.registry import UDFRegistry
```

Add:

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

    @property
    def is_single_stage(self) -> bool:
        return len(self.steps) == 1


def _canonical_node(node: ExprNode):
    if isinstance(node, Column):
        return ["column", node.name]
    if isinstance(node, LiteralNode):
        value_type = f"{type(node.value).__module__}.{type(node.value).__qualname__}"
        return ["literal", value_type, repr(node.value)]
    if isinstance(node, Call):
        args = []
        for arg in node.args:
            if isinstance(arg, KeywordArg):
                args.append(["keyword", arg.name, _canonical_node(arg.value)])
            else:
                args.append(["positional", _canonical_node(arg)])
        return ["call", node.fn_name, args]
    raise TypeError(f"Unknown node type: {type(node)}")


def node_fingerprint(node: ExprNode) -> str:
    payload = json.dumps(
        _canonical_node(node),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()
```

- [ ] **Step 4: Implement recursive lowering**

Add below the analyzer in `src/codestr/planner.py`:

```python
class _LoweringContext:
    def __init__(
        self,
        registry: UDFRegistry,
        existing_columns: Collection[str],
    ):
        self.registry = registry
        self.used_names = set(existing_columns)
        self.steps: list[PlanStep] = []
        self.extracted: dict[ExprNode, str] = {}

    def _reserve_internal_name(self, node: ExprNode) -> str:
        base = f"__codestr_{node_fingerprint(node)[:16]}"
        candidate = base
        suffix = 1
        while candidate in self.used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self.used_names.add(candidate)
        return candidate

    def _rewrite_call(
        self,
        node: Call,
        active_domain: WindowDomain | None,
    ) -> Call:
        args = []
        for arg in node.args:
            if isinstance(arg, KeywordArg):
                args.append(
                    KeywordArg(
                        arg.name,
                        self.lower(arg.value, active_domain),
                    )
                )
            else:
                args.append(self.lower(arg, active_domain))
        return Call(node.fn_name, tuple(args), _alias=node.alias)

    def lower(
        self,
        node: ExprNode,
        active_domain: WindowDomain | None = None,
    ) -> ExprNode:
        if not isinstance(node, Call):
            return node

        # Unknown calls stay intact so compilation reports Unknown function
        # before planner-specific mixed-window diagnostics.
        if self.registry.get(node.fn_name) is None:
            return node

        domain = _call_domain(node, self.registry)
        if (
            domain is not None
            and active_domain is not None
            and domain != active_domain
        ):
            existing = self.extracted.get(node)
            if existing is not None:
                return Column(existing)

            rewritten = self._rewrite_call(node, domain)
            output_name = self._reserve_internal_name(node)
            self.steps.append(
                PlanStep(
                    node=rewritten,
                    output_name=output_name,
                    internal=True,
                )
            )
            self.extracted[node] = output_name
            return Column(output_name)

        next_domain = domain if domain is not None else active_domain
        return self._rewrite_call(node, next_domain)


def _column_names(node: ExprNode) -> set[str]:
    if isinstance(node, Column):
        return {node.name}
    if not isinstance(node, Call):
        return set()

    names = set()
    for arg in node.args:
        names.update(_column_names(_arg_value(arg)))
    return names


def _validate_step_dependencies(steps: tuple[PlanStep, ...]) -> None:
    internal_names = {
        step.output_name
        for step in steps
        if step.internal
    }
    available_internal = set()

    for step in steps:
        missing = (
            _column_names(step.node) & internal_names
        ) - available_internal
        if missing:
            raise CompileError(
                f"Invalid plan dependency for {step.output_name}: "
                f"{sorted(missing)}"
            )
        if step.internal:
            available_internal.add(step.output_name)


def build_execution_plan(
    node: ExprNode,
    registry: UDFRegistry,
    existing_columns: Collection[str] = (),
) -> ExecutionPlan:
    context = _LoweringContext(registry, existing_columns)
    lowered = context.lower(node)
    steps = [
        *context.steps,
        PlanStep(
            node=lowered,
            output_name=node.alias,
            internal=False,
        ),
    ]

    frozen_steps = tuple(steps)
    _validate_step_dependencies(frozen_steps)

    for step in frozen_steps:
        boundary = find_mixed_window_boundary(step.node, registry)
        if boundary is not None:
            raise CompileError(
                "Planner invariant failed; stage still contains mixed windows: "
                f"{boundary.outer_call} -> {boundary.inner_call}"
            )

    return ExecutionPlan(
        steps=frozen_steps,
        output_name=node.alias,
    )
```

- [ ] **Step 5: Run all planner tests to verify GREEN**

Run:

```bash
uv run --locked pytest tests/test_planner.py -q
```

Expected: all analyzer and lowering tests pass.

- [ ] **Step 6: Run syntax/compiler/planner regression**

Run:

```bash
uv run --locked pytest \
  tests/test_syntax.py \
  tests/test_parser.py \
  tests/test_compiler.py \
  tests/test_planner.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Format and lint lowering changes**

Run:

```bash
uvx ruff format src/codestr/planner.py tests/test_planner.py
uvx ruff check src/codestr/planner.py tests/test_planner.py
git diff --check
```

Expected: Ruff passes and `git diff --check` has no output.

- [ ] **Step 8: Commit lowering**

```bash
git add src/codestr/planner.py tests/test_planner.py
git commit -m "Lower mixed windows into execution plans"
```

### Task 4: Execute Mixed Plans In Stateful SQL

**Files:**
- Modify: `src/codestr/engine.py:247-280`
- Modify: `tests/conftest.py`
- Modify: `tests/test_engine.py`

- [ ] **Step 1: Add a reusable mixed-window fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def mixed_window_df() -> pl.DataFrame:
    rows = [
        (day, asset, day * scale)
        for day in range(1, 66)
        for asset, scale in (("A", 1.0), ("B", 2.0), ("C", 3.0))
    ]
    return pl.DataFrame(
        rows,
        schema=("datetime", "asset", "x"),
        orient="row",
    )
```

- [ ] **Step 2: Write failing TS-over-CS integration test**

Add to `TestSQLInteractiveMode` in `tests/test_engine.py`:

```python
    def test_sql_lowers_ts_over_cs(self, mixed_window_df):
        actual_engine = CodeStr(mixed_window_df, align=False)
        actual = actual_engine.sql(
            "ts_mean(cs_moderate(x), 60) as factor"
        ).sort(["asset", "datetime"])

        expected_engine = CodeStr(mixed_window_df, align=False)
        expected_engine.sql("cs_moderate(x) as moderate")
        expected = expected_engine.sql(
            "ts_mean(moderate, 60) as factor"
        ).sort(["asset", "datetime"])

        assert actual_engine.failed == []
        assert actual["factor"].equals(expected["factor"])
        assert actual.filter(pl.col("asset") == "A")["factor"].tail(3).to_list() == [
            33.5,
            34.5,
            35.5,
        ]
        assert actual.columns == ["datetime", "asset", "factor"]
```

- [ ] **Step 3: Run TS-over-CS test to verify RED**

Run:

```bash
uv run --locked pytest \
  tests/test_engine.py::TestSQLInteractiveMode::test_sql_lowers_ts_over_cs \
  -q
```

Expected: result is all zero or SQL records the pure compiler mixed-window
error.

- [ ] **Step 4: Integrate plan execution into `_compile_expr`**

Modify `src/codestr/engine.py` imports:

```python
from codestr.planner import build_execution_plan
```

Replace `_compile_expr()` with:

```python
    def _compile_expr(self, expr: str, cover: bool):
        """Parse, plan, and append one expression to the current lazy graph."""
        if self._data_ is None:
            self._data_ = self.data.lazy()

        data_saved = self._data_
        cache_saved = dict(self._cur_expr_cache)

        try:
            node = _parse(expr)
            alias = node.alias
            current_cols = set(self._data_.collect_schema().names())

            if alias in current_cols and not cover:
                return pl.col(alias), alias
            if node in self._expr_cache and not cover:
                cached_alias = self._expr_cache[node]
                if cached_alias in current_cols:
                    expr_pl = pl.col(cached_alias).alias(alias)
                    self._data_ = self._data_.with_columns(expr_pl)
                    return pl.col(alias), alias
            if node in self._cur_expr_cache and not cover:
                cached_alias = self._cur_expr_cache[node]
                if cached_alias in current_cols:
                    expr_pl = pl.col(cached_alias).alias(alias)
                    self._data_ = self._data_.with_columns(expr_pl)
                    return pl.col(alias), alias

            registry = UDFRegistry.get_instance()
            plan = build_execution_plan(
                node,
                registry,
                existing_columns=current_cols,
            )

            if plan.is_single_stage:
                expr_pl = _pure_compile(
                    node,
                    registry=registry,
                    dims=getattr(self, "dims", None),
                    ts_over=self._ts_over,
                    cs_over=self._cs_over,
                )
                self._data_ = self._data_.with_columns(expr_pl.alias(alias))
            else:
                for step in plan.steps:
                    expr_pl = _pure_compile(
                        step.node,
                        registry=registry,
                        dims=getattr(self, "dims", None),
                        ts_over=self._ts_over,
                        cs_over=self._cs_over,
                    )
                    self._data_ = self._data_.with_columns(
                        expr_pl.alias(step.output_name)
                    )

            self._cur_expr_cache[node] = alias
            return pl.col(alias), alias

        except Exception as e:
            self._data_ = data_saved
            self._cur_expr_cache = cache_saved
            raise CompileError(
                message=f"[表达式]: {expr}\n[编译器外层]\n{e}"
            ) from e
```

This preserves the old single-stage compiler path and only enters the step
loop for mixed plans.

- [ ] **Step 5: Run TS-over-CS test to verify GREEN**

Run:

```bash
uv run --locked pytest \
  tests/test_engine.py::TestSQLInteractiveMode::test_sql_lowers_ts_over_cs \
  -q
```

Expected: test passes with non-zero A/C results.

- [ ] **Step 6: Add inverse mixed-window integration test**

Add to `TestSQLInteractiveMode`:

```python
    def test_sql_lowers_cs_over_ts(self, mixed_window_df):
        actual_engine = CodeStr(mixed_window_df, align=False)
        actual = actual_engine.sql(
            "cs_mean(ts_mean(x, 2, min_samples=1)) as factor"
        ).sort(["asset", "datetime"])

        expected_engine = CodeStr(mixed_window_df, align=False)
        expected_engine.sql(
            "ts_mean(x, 2, min_samples=1) as rolling"
        )
        expected = expected_engine.sql(
            "cs_mean(rolling) as factor"
        ).sort(["asset", "datetime"])

        assert actual_engine.failed == []
        assert actual["factor"].equals(expected["factor"])
```

- [ ] **Step 7: Run related engine tests**

Run:

```bash
uv run --locked pytest \
  tests/test_engine.py::TestSQLInteractiveMode::test_sql_lowers_ts_over_cs \
  tests/test_engine.py::TestSQLInteractiveMode::test_sql_lowers_cs_over_ts \
  tests/test_engine.py::TestSQLInteractiveMode::test_sql_keyword_argument_reuses_cache_with_new_alias \
  tests/test_engine.py::TestSQLInteractiveMode::test_sql_min_samples_keyword_reuses_cache \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Format and lint SQL execution changes**

Run:

```bash
uvx ruff format src/codestr/engine.py tests/conftest.py tests/test_engine.py
uvx ruff check src/codestr/engine.py tests/conftest.py tests/test_engine.py
git diff --check
```

Expected: Ruff passes and `git diff --check` has no output.

- [ ] **Step 9: Commit basic SQL execution**

```bash
git add src/codestr/engine.py tests/conftest.py tests/test_engine.py
git commit -m "Execute mixed window plans in SQL"
```

### Task 5: Preserve Same-Query Alias Ordering And Lazy Semantics

**Files:**
- Modify: `src/codestr/engine.py:304-359`
- Modify: `tests/test_engine.py`

- [ ] **Step 1: Write the requested same-query alias regression test**

Add to `TestSQLInteractiveMode`:

```python
    def test_mixed_window_uses_preceding_alias_in_same_sql(
        self,
        mixed_window_df,
    ):
        actual_engine = CodeStr(mixed_window_df, align=False)
        actual = actual_engine.sql(
            "x * 2 as scaled",
            "ts_mean(cs_moderate(scaled), 2, min_samples=1) as factor",
        ).sort(["asset", "datetime"])

        expected_engine = CodeStr(mixed_window_df, align=False)
        expected = expected_engine.sql(
            "x * 2 as scaled",
            "cs_moderate(scaled) as moderate",
            "ts_mean(moderate, 2, min_samples=1) as factor",
        ).sort(["asset", "datetime"])

        assert actual_engine.failed == []
        assert actual["factor"].equals(expected["factor"])
        assert actual.filter(pl.col("asset") == "A")["factor"].head(4).to_list() == [
            2.0,
            3.0,
            5.0,
            7.0,
        ]
```

- [ ] **Step 2: Write eager/lazy equivalence test**

Add:

```python
    def test_mixed_window_lazy_matches_eager(self, mixed_window_df):
        eager_engine = CodeStr(mixed_window_df, align=False)
        eager = eager_engine.sql(
            "ts_mean(cs_moderate(x), 2, min_samples=1) as factor"
        ).sort(["asset", "datetime"])

        lazy_engine = CodeStr(mixed_window_df, align=False)
        lazy = (
            lazy_engine.sql(
                "ts_mean(cs_moderate(x), 2, min_samples=1) as factor",
                lazy=True,
            )
            .collect()
            .sort(["asset", "datetime"])
        )

        assert lazy["factor"].equals(eager["factor"])
        assert lazy_engine._expr_cache == {}
```

- [ ] **Step 3: Run ordering/lazy tests to expose stale lazy cache**

Run:

```bash
uv run --locked pytest \
  tests/test_engine.py::TestSQLInteractiveMode::test_mixed_window_uses_preceding_alias_in_same_sql \
  tests/test_engine.py::TestSQLInteractiveMode::test_mixed_window_lazy_matches_eager \
  -q
```

Expected: same-query alias passes after Task 4; lazy cache assertion fails because
the current `lazy=True` path persists `_cur_expr_cache` before rolling back the
graph.

- [ ] **Step 4: Stop persisting cache entries for rolled-back lazy graphs**

Replace the `lazy` branch in `CodeStr.sql()`:

```python
        if lazy:
            result = self._data_.select(*self.index, *exprs_select)
            self._data_ = _data_saved
            self._cur_expr_cache = {}
            return result
```

Do not update `_expr_cache` in this branch. The returned LazyFrame owns the
planned graph, while the engine returns to its pre-query graph.

- [ ] **Step 5: Add custom window configuration test**

Add:

```python
    def test_mixed_window_respects_custom_over_columns(self):
        df = pl.DataFrame(
            {
                "trade_date": [1, 1, 2, 2, 3, 3],
                "symbol": ["A", "B", "A", "B", "A", "B"],
                "x": [1.0, 3.0, 2.0, 6.0, 4.0, 8.0],
            }
        )
        actual_engine = CodeStr(
            df,
            index=("trade_date", "symbol"),
            partition_by=["symbol"],
            order_by=["trade_date"],
            align=False,
        )
        actual = actual_engine.sql(
            "ts_mean(cs_moderate(x), 2, min_samples=1) as factor"
        ).sort(["symbol", "trade_date"])

        expected = (
            df.lazy()
            .with_columns(
                (
                    pl.col("x")
                    - pl.col("x")
                    .mean()
                    .over(
                        partition_by=["trade_date"],
                        order_by=["symbol"],
                    )
                )
                .abs()
                .alias("moderate")
            )
            .with_columns(
                pl.col("moderate")
                .rolling_mean(2, min_samples=1)
                .over(
                    partition_by=["symbol"],
                    order_by=["trade_date"],
                )
                .alias("factor")
            )
            .select("trade_date", "symbol", "factor")
            .collect()
            .sort(["symbol", "trade_date"])
        )

        assert actual["factor"].equals(expected["factor"])
```

- [ ] **Step 6: Run new engine tests to verify GREEN**

Run:

```bash
uv run --locked pytest \
  tests/test_engine.py::TestSQLInteractiveMode::test_mixed_window_uses_preceding_alias_in_same_sql \
  tests/test_engine.py::TestSQLInteractiveMode::test_mixed_window_lazy_matches_eager \
  tests/test_engine.py::TestSQLInteractiveMode::test_mixed_window_respects_custom_over_columns \
  -q
```

Expected: all three tests pass.

- [ ] **Step 7: Format and lint ordering changes**

Run:

```bash
uvx ruff format src/codestr/engine.py tests/test_engine.py
uvx ruff check src/codestr/engine.py tests/test_engine.py
git diff --check
```

Expected: Ruff passes and `git diff --check` has no output.

- [ ] **Step 8: Commit ordering and lazy behavior**

```bash
git add src/codestr/engine.py tests/test_engine.py
git commit -m "Preserve query ordering in mixed window plans"
```

### Task 6: Harden Cache, Cover, Deduplication, And Rollback

**Files:**
- Modify: `src/codestr/engine.py`
- Modify: `tests/test_engine.py`
- Modify: `tests/test_planner.py`

- [ ] **Step 1: Add final-root cache reuse test**

Add to `TestSQLInteractiveMode`:

```python
    def test_mixed_window_reuses_root_cache_with_new_alias(
        self,
        mixed_window_df,
    ):
        cs = CodeStr(mixed_window_df, align=False)

        first = cs.sql(
            "ts_mean(cs_moderate(x), 2, min_samples=1) as first"
        )
        cache_size = len(cs._expr_cache)
        second = cs.sql(
            "ts_mean(cs_moderate(x), 2, min_samples=1) as second"
        )

        assert cs.failed == []
        assert second["second"].to_list() == first["first"].to_list()
        assert len(cs._expr_cache) == cache_size
```

- [ ] **Step 2: Add `cover=True` full-plan recomputation test**

Add:

```python
    def test_mixed_window_cover_recomputes_internal_stages(
        self,
        mixed_window_df,
    ):
        cs = CodeStr(mixed_window_df, align=False)
        first = cs.sql(
            "ts_mean(cs_moderate(x), 2, min_samples=1) as factor"
        )

        cs._data_ = cs._data_.with_columns((pl.col("x") * 2).alias("x"))
        second = cs.sql(
            "ts_mean(cs_moderate(x), 2, min_samples=1) as factor",
            cover=True,
        )

        assert second["factor"].to_list() != first["factor"].to_list()
        assert second["factor"].fill_null(0).sum() == pytest.approx(
            2 * first["factor"].fill_null(0).sum()
        )
```

- [ ] **Step 3: Add partial-plan rollback test**

Add these module imports to `tests/test_engine.py`:

```python
from codestr.udf.registry import UDFMeta, UDFRegistry
```

Add:

```python
    def test_failed_mixed_plan_does_not_leak_internal_columns(
        self,
        mixed_window_df,
    ):
        cs = CodeStr(mixed_window_df, align=False)
        before = set(cs.data.columns)

        def fail_after_child(
            expr,
            partition_by=None,
            order_by=None,
        ):
            raise ValueError("planned failure")

        UDFRegistry.get_instance().register(
            UDFMeta(
                name="ts_fail_after_child",
                fn=fail_after_child,
                category="ts",
            )
        )
        result = cs.sql(
            "ts_fail_after_child(cs_moderate(x)) as invalid"
        )

        assert len(cs.failed) == 1
        assert "planned failure" in str(cs.failed[0])
        assert set(cs._data_.collect_schema().names()) == before
        assert not any(
            name.startswith("__codestr_")
            for name in cs._data_.collect_schema().names()
        )
        assert "invalid" not in result.columns
```

- [ ] **Step 4: Run hardening tests**

Run:

```bash
uv run --locked pytest \
  tests/test_engine.py::TestSQLInteractiveMode::test_mixed_window_reuses_root_cache_with_new_alias \
  tests/test_engine.py::TestSQLInteractiveMode::test_mixed_window_cover_recomputes_internal_stages \
  tests/test_engine.py::TestSQLInteractiveMode::test_failed_mixed_plan_does_not_leak_internal_columns \
  -q
```

Expected: all three tests pass. The rollback test fails deterministically while
compiling the final TS stage, after the internal CS stage has been appended.

- [ ] **Step 5: Run complete engine and planner suites**

Run:

```bash
uv run --locked pytest tests/test_planner.py tests/test_engine.py -q
```

Expected: all planner and engine tests pass.

- [ ] **Step 6: Format and lint hardening changes**

Run:

```bash
uvx ruff format src/codestr/engine.py tests/test_engine.py tests/test_planner.py
uvx ruff check src/codestr/engine.py tests/test_engine.py tests/test_planner.py
git diff --check
```

Expected: Ruff passes and `git diff --check` has no output.

- [ ] **Step 7: Commit cache and rollback hardening**

```bash
git add src/codestr/engine.py tests/test_engine.py tests/test_planner.py
git commit -m "Harden mixed window plan state handling"
```

### Task 7: Document Planner Semantics

**Files:**
- Modify: `README.md`
- Modify: `docs/operators.md`

- [ ] **Step 1: Add README mixed-window example**

Add after the window configuration section in `README.md`:

````markdown
### 混合窗口

`CodeStr.sql()` 会自动把 TS/CS 混合窗口拆成连续的 lazy projection：

```python
cs.sql(
    "close * 2 as scaled",
    "ts_mean(cs_moderate(scaled), 60) as factor",
)
```

其语义等价于先生成 `cs_moderate(scaled)` 中间列，再沿资产时间轴
计算 `ts_mean`。中间列不会出现在返回结果中。

纯 `cs.compile()` 只能返回一个 `pl.Expr`，因此会明确拒绝需要多阶段
执行的 TS/CS 混合窗口。TS→TS 和 CS→CS 同域嵌套不受影响。
````

- [ ] **Step 2: Add operator reference planner section**

Add to the window semantics section in `docs/operators.md`:

````markdown
### 混合窗口规划

TS 与 CS 使用不同分组轴，不能安全地嵌套成一个 Polars window
expression。`CodeStr.sql()` 会检测窗口祖先链上的域变化，并生成多个
连续 `with_columns` 阶段：

```text
ts_mean(cs_moderate(x), 60)

stage 1: internal = cs_moderate(x)
stage 2: result = ts_mean(internal, 60)
```

该过程保持 lazy，不会在阶段间调用 `collect()`。同一次 `sql()` 中，
后续表达式也可以引用前一条表达式创建的别名。

`CodeStr.compile()` 仍返回单个 `pl.Expr`，对混合窗口抛出
`CompileError`。同域嵌套和并列的 TS/CS 表达式继续正常编译。
````

- [ ] **Step 3: Check documentation diff**

Run:

```bash
git diff --check
git diff -- README.md docs/operators.md
```

Expected: no whitespace errors; documentation states SQL auto-lowering and
pure compile rejection without changing operator signatures.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/operators.md
git commit -m "Document mixed window planning"
```

### Task 8: Full Verification And Review

**Files:**
- Verify all changed files

- [ ] **Step 1: Verify dependency lock**

Run:

```bash
uv lock --check
uv sync --locked --extra test
```

Expected: lock is current and sync succeeds without changing `uv.lock`.

- [ ] **Step 2: Run full tests**

Run:

```bash
uv run --locked pytest tests/ -q
```

Expected: all tests pass, including the original 214 tests and all new planner
tests.

- [ ] **Step 3: Run lint and formatting checks**

Run:

```bash
uvx ruff check src/ tests/
uvx ruff format --check src/ tests/
```

Expected: Ruff reports no errors and all files are formatted.

- [ ] **Step 4: Build release artifacts outside the repository**

Run:

```bash
BUILD_DIR=$(mktemp -d /tmp/codestr-mixed-window.XXXXXX)
uv build --out-dir "$BUILD_DIR"
```

Expected: `codestr-0.2.1.tar.gz` and
`codestr-0.2.1-py3-none-any.whl` build successfully. This task does not change
the project version.

- [ ] **Step 5: Inspect final repository state**

Run:

```bash
git diff --check master..HEAD
git status --short --branch
git log --oneline master..HEAD
```

Expected: no uncommitted changes and only planner-related commits appear above
`master`.

- [ ] **Step 6: Request independent code review**

Review `master..HEAD` with focus on:

- mixed-domain classification through math and keyword nodes;
- same-domain and sibling-window false positives;
- topological lowering for alternating domains;
- deterministic internal names and collisions;
- same-query alias dependencies;
- eager/lazy cache lifetime;
- `cover=True` and partial-plan rollback;
- preservation of the old single-stage compiler path.

Fix every critical or important finding and rerun Steps 1-5.
