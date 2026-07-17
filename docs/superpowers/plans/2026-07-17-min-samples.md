# Time-Series `min_samples` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose `min_samples` on every rolling and EMA time-series operator while preserving each underlying Polars default.

**Architecture:** Add one private validator in `ts_udf.py`, then pass the validated keyword to each supported Polars window method. Rolling operators use `None`, EMA uses `1`, and MAD applies one threshold to both median stages; the existing immutable keyword-argument AST and compiler require no changes.

**Tech Stack:** Python 3.10+, Polars 1.42.1, Lark, uv, pytest, Ruff, Hatchling

---

### Task 1: Add `min_samples` to the standard rolling operators

**Files:**
- Modify: `src/codestr/udf/ts_udf.py:31-85`
- Modify: `tests/test_ts_udf.py`
- Modify: `tests/test_engine.py:53-126`

- [ ] **Step 1: Write failing rolling, validation, and cache tests**

Add these imports and fixtures to `tests/test_ts_udf.py`:

```python
from polars.testing import assert_series_equal

from codestr.parser import parse
from codestr.udf.ts_udf import ts_mean


@pytest.fixture
def unordered_ts_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [3, 1, 2, 4, 3, 1, 2, 4],
            "asset": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "value": [4.0, 1.0, None, 8.0, 30.0, 10.0, 20.0, None],
        }
    )
```

Add parameterized coverage for all standard rolling operators:

```python
@pytest.mark.parametrize(
    ("fn_name", "method_name"),
    [
        ("ts_max", "rolling_max"),
        ("ts_min", "rolling_min"),
        ("ts_mean", "rolling_mean"),
        ("ts_std", "rolling_std"),
        ("ts_skew", "rolling_skew"),
        ("ts_kurt", "rolling_kurtosis"),
        ("ts_sum", "rolling_sum"),
        ("ts_var", "rolling_var"),
        ("ts_mid", "rolling_median"),
    ],
)
@pytest.mark.parametrize("min_samples", [None, 1])
def test_rolling_min_samples_matches_polars(
    unordered_ts_df,
    fn_name,
    method_name,
    min_samples,
):
    if min_samples is None:
        source = f"{fn_name}(value, 3)"
        rolling_kwargs = {}
    else:
        source = f"{fn_name}(value, 3, min_samples={min_samples})"
        rolling_kwargs = {"min_samples": min_samples}

    node = parse(source)
    actual = unordered_ts_df.select(ast_compile(node))[node.alias]
    expected = unordered_ts_df.select(
        getattr(pl.col("value"), method_name)(3, **rolling_kwargs)
        .over(partition_by=["asset"], order_by=["datetime"])
        .alias("expected")
    )["expected"]

    assert_series_equal(actual.rename("expected"), expected)


@pytest.mark.parametrize("min_samples", [0, -1, 1.5, True])
def test_rolling_rejects_invalid_min_samples(min_samples):
    with pytest.raises(ValueError, match="min_samples must be a positive integer"):
        ts_mean(pl.col("value"), 3, min_samples=min_samples)
```

Add the stateful cache regression to `TestSQLInteractiveMode` in
`tests/test_engine.py`:

```python
def test_sql_min_samples_keyword_reuses_cache(self, sample_df):
    cs = CodeStr(sample_df)

    first = cs.sql("ts_mean(close, 3, min_samples=1) as mean_fast")
    cache_size = len(cs._expr_cache)
    second = cs.sql("ts_mean(close, 3, min_samples=1) as mean_fast_again")

    assert cs.failed == []
    assert first["mean_fast"].null_count() == 0
    assert second["mean_fast_again"].to_list() == first["mean_fast"].to_list()
    assert len(cs._expr_cache) == cache_size
```

- [ ] **Step 2: Run focused tests and verify the feature is missing**

Run:

```bash
uv run --locked pytest \
  tests/test_ts_udf.py::test_rolling_min_samples_matches_polars \
  tests/test_ts_udf.py::test_rolling_rejects_invalid_min_samples \
  tests/test_engine.py::TestSQLInteractiveMode::test_sql_min_samples_keyword_reuses_cache \
  -q
```

Expected: explicit threshold cases fail because the current UDF signatures do
not accept `min_samples`; direct validation cases raise `TypeError` instead of
the required `ValueError`.

- [ ] **Step 3: Implement shared validation and standard rolling support**

Add the validator after `__all__` in `src/codestr/udf/ts_udf.py`:

```python
def _validate_min_samples(min_samples: int | None) -> int | None:
    if min_samples is not None and (
        type(min_samples) is not int or min_samples < 1
    ):
        raise ValueError("min_samples must be a positive integer")
    return min_samples
```

Replace the nine standard rolling functions with:

```python
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
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run:

```bash
uv run --locked pytest \
  tests/test_ts_udf.py::test_rolling_min_samples_matches_polars \
  tests/test_ts_udf.py::test_rolling_rejects_invalid_min_samples \
  tests/test_engine.py::TestSQLInteractiveMode::test_sql_min_samples_keyword_reuses_cache \
  -q
```

Expected: all selected tests pass. The cases where `min_samples` is omitted
confirm that rolling defaults remain unchanged.

- [ ] **Step 5: Run formatting and commit Task 1**

Run:

```bash
uvx ruff format src/codestr/udf/ts_udf.py tests/test_ts_udf.py tests/test_engine.py
uvx ruff check src/codestr/udf/ts_udf.py tests/test_ts_udf.py tests/test_engine.py
git diff --check
git add src/codestr/udf/ts_udf.py tests/test_ts_udf.py tests/test_engine.py
git commit -m "Expose min_samples on rolling operators"
```

Expected: Ruff and `git diff --check` succeed, then the commit is created.

### Task 2: Add `min_samples` to EMA and MAD

**Files:**
- Modify: `src/codestr/udf/ts_udf.py:47-98`
- Modify: `tests/test_ts_udf.py`

- [ ] **Step 1: Write failing EMA and MAD tests**

Extend `TestTSEma` in `tests/test_ts_udf.py`:

```python
from codestr.udf.ts_udf import ts_ema, ts_mean


def test_explicit_min_samples(self):
    df = pl.DataFrame(
        {
            "datetime": [1, 2, 3, 4],
            "asset": ["A", "A", "A", "A"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    node = parse("ts_ema(value, 2, min_samples=3)")

    actual = df.select(ast_compile(node))[node.alias]
    expected = df.select(
        pl.col("value")
        .ewm_mean(span=2, adjust=False, min_samples=3)
        .over(partition_by=["asset"], order_by=["datetime"])
        .alias("expected")
    )["expected"]

    assert_series_equal(actual.rename("expected"), expected)
    assert actual.to_list()[:2] == [None, None]


@pytest.mark.parametrize("min_samples", [0, -1, 1.5, True])
def test_rejects_invalid_min_samples(self, min_samples):
    with pytest.raises(ValueError, match="min_samples must be a positive integer"):
        ts_ema(pl.col("value"), 2, min_samples=min_samples)
```

Add MAD coverage:

```python
class TestTSMad:
    @pytest.mark.parametrize("min_samples", [None, 1])
    def test_min_samples_applies_to_both_stages(
        self,
        unordered_ts_df,
        min_samples,
    ):
        if min_samples is None:
            source = "ts_mad(value, 3)"
            rolling_kwargs = {}
        else:
            source = f"ts_mad(value, 3, min_samples={min_samples})"
            rolling_kwargs = {"min_samples": min_samples}

        node = parse(source)
        actual = unordered_ts_df.select(ast_compile(node))[node.alias]
        inner = pl.col("value").rolling_median(3, **rolling_kwargs).over(
            partition_by=["asset"],
            order_by=["datetime"],
        )
        expected = unordered_ts_df.select(
            (
                1.4826
                * (pl.col("value") - inner)
                .abs()
                .rolling_median(3, **rolling_kwargs)
                .over(partition_by=["asset"], order_by=["datetime"])
            ).alias("expected")
        )["expected"]

        assert_series_equal(actual.rename("expected"), expected)
```

- [ ] **Step 2: Run EMA and MAD tests and verify explicit thresholds fail**

Run:

```bash
uv run --locked pytest \
  tests/test_ts_udf.py::TestTSEma::test_explicit_min_samples \
  tests/test_ts_udf.py::TestTSEma::test_rejects_invalid_min_samples \
  tests/test_ts_udf.py::TestTSMad::test_min_samples_applies_to_both_stages \
  -q
```

Expected: explicit `min_samples` cases fail because `ts_ema` and `ts_mad` do
not yet accept the keyword. Omitted MAD behavior may already pass.

- [ ] **Step 3: Implement EMA and MAD support**

Replace `ts_ema` with:

```python
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
    return expr.ewm_mean(
        span=span,
        adjust=False,
        min_samples=min_samples,
    ).over(
        partition_by=partition_by,
        order_by=order_by,
    )
```

Replace `ts_mad` with:

```python
@udf(category="ts")
def ts_mad(
    expr: pl.Expr,
    windows,
    partition_by=None,
    order_by=None,
    *,
    min_samples=None,
):
    """Median Absolute Deviation over a rolling window."""
    min_samples = _validate_min_samples(min_samples)
    rolling_median = expr.rolling_median(
        windows,
        min_samples=min_samples,
    ).over(
        partition_by=partition_by,
        order_by=order_by,
    )
    return (
        1.4826
        * (expr - rolling_median)
        .abs()
        .rolling_median(
            windows,
            min_samples=min_samples,
        )
        .over(
            partition_by=partition_by,
            order_by=order_by,
        )
    )
```

- [ ] **Step 4: Run all time-series and engine tests**

Run:

```bash
uv run --locked pytest tests/test_ts_udf.py tests/test_engine.py -q
```

Expected: all time-series and engine tests pass. Existing EMA tests confirm
that omitted `min_samples` remains equivalent to Polars' default `1`.

- [ ] **Step 5: Format and commit Task 2**

Run:

```bash
uvx ruff format src/codestr/udf/ts_udf.py tests/test_ts_udf.py
uvx ruff check src/codestr/udf/ts_udf.py tests/test_ts_udf.py
git diff --check
git add src/codestr/udf/ts_udf.py tests/test_ts_udf.py
git commit -m "Expose min_samples on EMA and MAD"
```

Expected: Ruff and `git diff --check` succeed, then the commit is created.

### Task 3: Document the Polars-aligned defaults

**Files:**
- Modify: `README.md:60-86`
- Modify: `docs/operators.md:271-304`

- [ ] **Step 1: Update the README examples**

Add this paragraph and example after the window-rules table in `README.md`:

````markdown
Rolling and EMA operators expose Polars' `min_samples` option. Defaults follow
Polars: rolling operators use `None` (a full window), while `ts_ema` uses `1`.

```python
cs.sql("ts_mean(close, 20, min_samples=5) as ma20_partial")
cs.sql("ts_ema(close, 10, min_samples=3) as ema10_partial")
```
````

- [ ] **Step 2: Update the operator reference**

Replace the signature paragraph in `docs/operators.md` with:

```markdown
滚动算子的签名为 `(expr, windows, min_samples=None)`，其中 `windows`
是回溯窗口大小。`ts_ema` 的签名为
`(expr, span, min_samples=1)`，`span` 必须是正整数。`min_samples`
省略时沿用 Polars 默认值，因此不会改变现有表达式结果。
```

Update the implementation column for every covered row so it includes the
parameter. Use these exact forms:

```markdown
| `ts_mean` | 滚动均值 | `rolling_mean(windows, min_samples=min_samples)` |
| `ts_ema` | 递归指数移动平均 | `ewm_mean(span=span, adjust=False, min_samples=min_samples)` |
| `ts_sum` | 滚动求和 | `rolling_sum(windows, min_samples=min_samples)` |
| `ts_std` | 滚动标准差 | `rolling_std(windows, min_samples=min_samples)` |
| `ts_var` | 滚动方差 | `rolling_var(windows, min_samples=min_samples)` |
| `ts_skew` | 滚动偏度 | `rolling_skew(windows, min_samples=min_samples)` |
| `ts_kurt` | 滚动峰度 | `rolling_kurtosis(windows, min_samples=min_samples)` |
| `ts_max` | 滚动最大值 | `rolling_max(windows, min_samples=min_samples)` |
| `ts_min` | 滚动最小值 | `rolling_min(windows, min_samples=min_samples)` |
| `ts_mid` | 滚动中位数 | `rolling_median(windows, min_samples=min_samples)` |
| `ts_mad` | 滚动 MAD | 内外两层 `rolling_median` 使用同一 `min_samples` |
```

Add explicit examples:

```python
cs.sql("ts_mean(close, 20, min_samples=5) as ma20_partial")
cs.sql("ts_ema(close, 10, min_samples=3) as ema10_partial")
```

- [ ] **Step 3: Check docs and commit Task 3**

Run:

```bash
rg -n "min_samples" README.md docs/operators.md
git diff --check
git add README.md docs/operators.md
git commit -m "Document time-series min_samples"
```

Expected: both documents describe rolling default `None`, EMA default `1`,
and explicit threshold examples; `git diff --check` succeeds.

### Task 4: Complete repository verification

**Files:**
- Verify: `src/`
- Verify: `tests/`
- Verify: `pyproject.toml`
- Verify: `uv.lock`

- [ ] **Step 1: Verify the uv lock and complete test suite**

Run:

```bash
uv lock --check
uv sync --locked --extra test
uv run --locked pytest tests/ -q
```

Expected: lock verification and sync succeed, and the full test suite reports
zero failures.

- [ ] **Step 2: Verify lint and formatting**

Run:

```bash
uvx ruff check src/ tests/
uvx ruff format --check src/ tests/
```

Expected: Ruff reports no lint errors and no files requiring formatting.

- [ ] **Step 3: Build and inspect package metadata**

Run:

```bash
build_dir=$(mktemp -d)
uv build --out-dir "$build_dir"
.venv/bin/python - "$build_dir" <<'PY'
import pathlib
import sys
import zipfile

build_dir = pathlib.Path(sys.argv[1])
wheel = next(build_dir.glob("codestr-*.whl"))
with zipfile.ZipFile(wheel) as archive:
    metadata_name = next(
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    )
    metadata = archive.read(metadata_name).decode()

assert "Requires-Dist: polars>=1.42.1\n" in metadata
print(wheel.name)
print("metadata_ok=yes")
PY
```

Expected: `uv build` creates an sdist and wheel, and the metadata check prints
`metadata_ok=yes`.

- [ ] **Step 4: Inspect the final diff**

Run:

```bash
git diff --check origin/master..HEAD
git status --short --branch
git log --oneline origin/master..HEAD
```

Expected: the worktree is clean, the diff check succeeds, and the branch
contains only the design, implementation, test, and documentation commits for
this feature.
