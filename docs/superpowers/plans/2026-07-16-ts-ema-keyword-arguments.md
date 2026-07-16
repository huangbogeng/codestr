# EMA And Keyword Arguments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ts_ema(expr, span)`, make keyword arguments immutable and cache-safe, and require Polars 1.42.1 or newer.

**Architecture:** Replace mutable keyword-argument dictionaries in the AST with a frozen `KeywordArg` value that is transparent to syntax analysis and explicitly resolved by the compiler. Add `ts_ema` as a normal `ts` UDF using the per-instance window configuration and the recursive EMA semantics verified in the `lidb` reference. Keep compatibility aliases and all data integrations out of scope.

**Tech Stack:** Python 3.10+, Polars 1.42.1, Lark, uv, pytest, Ruff, Hatchling

---

### Task 1: Make keyword arguments immutable and cache-safe

**Files:**
- Modify: `src/codestr/syntax.py`
- Modify: `src/codestr/parser.py`
- Modify: `src/codestr/compiler.py`
- Modify: `src/codestr/engine.py`
- Test: `tests/test_parser.py`
- Test: `tests/test_syntax.py`
- Test: `tests/test_compiler.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Add failing parser and syntax tests**

Add tests that require keyword arguments to be immutable, hashable, correctly rendered, and
transparent to analysis:

```python
from codestr.parser import parse
from codestr.syntax import Call, Column, KeywordArg, Literal, depth, node_count, to_rpn


def test_keyword_argument_is_immutable_and_hashable():
    node = parse("clip(close, lower_bound=0)")
    keyword = node.args[1]

    assert keyword == KeywordArg("lower_bound", Literal(0))
    assert node.alias == "clip(close, lower_bound=0)"
    assert hash(node) == hash(parse("clip(close, lower_bound=0)"))
    with pytest.raises(Exception):  # noqa: B017
        keyword.name = "upper_bound"


def test_keyword_argument_is_transparent_to_analysis():
    node = parse("clip(abs(close), lower_bound=0)")

    assert depth(node) == 3
    assert node_count(node) == 4
    assert [token.name for token in to_rpn(node)] == ["close", "abs", "0", "clip"]


def test_call_rejects_mutable_argument_dicts():
    with pytest.raises(TypeError, match="Call arguments"):
        Call("clip", (Column("close"), {"lower_bound": Literal(0)}))
```

- [ ] **Step 2: Add failing compiler and stateful-cache regression tests**

Add pure and stateful execution coverage:

```python
def test_compile_keyword_argument():
    df = pl.DataFrame({"close": [-1.0, 2.0]})
    result = df.select(ast_compile(parse("clip(close, lower_bound=0)")))
    assert result["clip(close, lower_bound=0)"].to_list() == [0.0, 2.0]


def test_sql_keyword_argument_reuses_cache_with_new_alias(sample_df):
    cs = CodeStr(sample_df)

    first = cs.sql("clip(close, lower_bound=101) as clipped")
    cache_size = len(cs._expr_cache)
    second = cs.sql("clip(close, lower_bound=101) as clipped_again")

    assert cs.failed == []
    assert first["clipped"].to_list() == [101.0, 200.0, 101.0, 198.0, 102.0, 202.0, 103.0, 204.0]
    assert second["clipped_again"].to_list() == first["clipped"].to_list()
    assert len(cs._expr_cache) == cache_size
```

- [ ] **Step 3: Run focused tests and verify the current representation fails**

Run:

```bash
uv run pytest tests/test_parser.py tests/test_syntax.py tests/test_compiler.py tests/test_engine.py -q
```

Expected: new tests fail because parsing returns `dict`, rendering is not canonical, analysis sees an
unknown node type, and `CodeStr.sql()` reports `unhashable type: 'dict'`.

- [ ] **Step 4: Implement the immutable keyword-argument AST**

In `src/codestr/syntax.py`, add a frozen value and update call argument handling:

```python
@dataclass(frozen=True)
class KeywordArg:
    """An immutable keyword argument inside a function call."""

    name: str
    value: ExprNode

    def __post_init__(self) -> None:
        if not isinstance(self.value, ExprNode):
            raise TypeError("Keyword argument values must be expression nodes")

    def __str__(self) -> str:
        return f"{self.name}={self.value}"


CallArg = ExprNode | KeywordArg


def _arg_value(arg: CallArg) -> ExprNode:
    return arg.value if isinstance(arg, KeywordArg) else arg
```

Change `Call.args` and `_render_call` to use `CallArg`. In `Call.__post_init__`, reject any argument
that is not an `ExprNode` or `KeywordArg` before rendering. Update `depth`, `node_count`, `to_rpn`,
and `descendants` to recurse into `_arg_value(arg)`. Keep `Call.__hash__` and `Call.__eq__` structural
and alias-independent.

In `src/codestr/parser.py`, emit the immutable value:

```python
def keyword_arg(self, item) -> KeywordArg:
    key, value = item
    name = key.name if isinstance(key, Column) else str(key)
    return KeywordArg(name, value)
```

In `src/codestr/compiler.py`, resolve it explicitly:

```python
if isinstance(arg, KeywordArg):
    kwargs[arg.name] = _resolve(arg.value, registry, dims, ts_over, cs_over)
else:
    args.append(_resolve(arg, registry, dims, ts_over, cs_over))
```

In `src/codestr/engine.py`, unwrap `KeywordArg.value` while recursively checking redundant nested
expressions.

- [ ] **Step 5: Run focused tests and verify they pass**

Run:

```bash
uv run pytest tests/test_parser.py tests/test_syntax.py tests/test_compiler.py tests/test_engine.py -q
```

Expected: all focused tests pass and the stateful keyword-argument query has no failures.

- [ ] **Step 6: Commit the keyword-argument fix**

```bash
git add src/codestr/syntax.py src/codestr/parser.py src/codestr/compiler.py src/codestr/engine.py \
  tests/test_parser.py tests/test_syntax.py tests/test_compiler.py tests/test_engine.py
git commit -m "Fix keyword argument expression caching"
```

### Task 2: Add the `ts_ema` operator

**Files:**
- Modify: `src/codestr/udf/ts_udf.py`
- Test: `tests/test_ts_udf.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Add failing EMA unit tests**

Add tests for Polars parity, grouping, ordering, nulls, and validation:

```python
class TestTSEma:
    def test_matches_recursive_polars_ema(self):
        df = pl.DataFrame(
            {
                "datetime": [3, 1, 2, 1, 3, 2],
                "asset": ["A", "A", "A", "B", "B", "B"],
                "value": [3.0, 1.0, None, 10.0, 14.0, 12.0],
            }
        )
        node = Call("ts_ema", (Column("value"), Literal(2)))
        actual = df.select(ast_compile(node))[node.alias]
        expected = df.select(
            pl.col("value")
            .ewm_mean(span=2, adjust=False, min_samples=1)
            .over(partition_by=["asset"], order_by=["datetime"])
            .alias("expected")
        )["expected"]
        assert actual.equals(expected)

    def test_span_one_returns_input(self, ts_df):
        node = Call("ts_ema", (Column("close"), Literal(1)))
        assert ts_df.select(ast_compile(node))[node.alias].to_list() == ts_df["close"].to_list()

    @pytest.mark.parametrize("span", [0, -1, 2.5])
    def test_rejects_invalid_span(self, span):
        node = Call("ts_ema", (Column("close"), Literal(span)))
        with pytest.raises(CompileError, match="positive integer"):
            ast_compile(node)
```

- [ ] **Step 2: Add a failing same-call intermediate-expression test**

```python
def test_ts_ema_can_feed_later_expression_in_same_sql(sample_df):
    cs = CodeStr(sample_df)
    result = cs.sql(
        "ts_ema(close, 2) as ema2",
        "ts_delta(ema2, 1) as ema_delta",
    )

    assert cs.failed == []
    assert {"ema2", "ema_delta"} <= set(result.columns)
    assert result["ema_delta"].null_count() == 2
```

- [ ] **Step 3: Run the focused tests and verify the operator is missing**

Run:

```bash
uv run pytest tests/test_ts_udf.py tests/test_engine.py -q
```

Expected: new tests fail with `Unknown function: ts_ema`.

- [ ] **Step 4: Implement `ts_ema`**

Add `ts_ema` to `__all__` and implement it next to the rolling mean operators:

```python
@udf(category="ts")
def ts_ema(expr: pl.Expr, span: int, partition_by=None, order_by=None):
    """Recursive exponential moving average over each ordered entity series."""
    if type(span) is not int or span < 1:
        raise ValueError("span must be a positive integer")
    return expr.ewm_mean(span=span, adjust=False, min_samples=1).over(
        partition_by=partition_by,
        order_by=order_by,
    )
```

- [ ] **Step 5: Run focused tests and verify they pass**

Run:

```bash
uv run pytest tests/test_ts_udf.py tests/test_engine.py -q
```

Expected: all EMA and engine tests pass.

- [ ] **Step 6: Commit the EMA operator**

```bash
git add src/codestr/udf/ts_udf.py tests/test_ts_udf.py tests/test_engine.py
git commit -m "Add time-series EMA operator"
```

### Task 3: Upgrade Polars, document the operator, and verify the package

**Files:**
- Modify: `pyproject.toml:26`
- Create: `uv.lock`
- Modify: `README.md:109`
- Modify: `docs/operators.md:281`

- [ ] **Step 1: Raise the dependency floor and refresh the lock**

Change the dependency entry to:

```toml
"polars>=1.42.1",
```

Then run:

```bash
uv lock --upgrade-package polars
uv sync --extra test
uv run python -c "import polars as pl; print(pl.__version__)"
```

Expected: the environment reports Polars 1.42.1 or newer and `uv.lock` records Polars 1.42.1.

- [ ] **Step 2: Document `ts_ema`**

Add `ts_ema` to the README time-series operator list. Add this row and explanation to the operator
reference:

```markdown
| `ts_ema` | 递归指数移动平均 | `ewm_mean(span=span, adjust=False, min_samples=1)` |
```

Document that `span` is a positive integer and that CodeStr independently applies the EMA inside each
entity partition in configured time order.

- [ ] **Step 3: Run the complete test suite**

Run:

```bash
uv run pytest tests/ -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 4: Run lint and formatting checks**

Run:

```bash
uvx ruff check src/ tests/
uvx ruff format --check src/ tests/
```

Expected: both commands exit successfully with no findings.

- [ ] **Step 5: Build and inspect package metadata**

Run:

```bash
uv build
unzip -p dist/codestr-0.1.0-py3-none-any.whl '*/METADATA' | rg 'Requires-Dist: polars'
```

Expected: source and wheel distributions build successfully and wheel metadata contains
`Requires-Dist: polars>=1.42.1`.

- [ ] **Step 6: Review and commit the final change**

Run:

```bash
git diff --check
git status --short
git add pyproject.toml uv.lock README.md docs/operators.md
git commit -m "Require Polars 1.42.1"
```

Expected: only approved implementation, tests, documentation, dependency metadata, and plan/spec
files are committed; `.coverage` remains outside the change.
