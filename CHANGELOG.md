# Changelog

## [Unreleased]

### Added
- **Per-instance over-window config**: `partition_by` / `order_by` params on `CodeStr.__init__`
  replacing the global `over` dict and `configure_over()`.
- Multi-column support for `partition_by` and `order_by`.
- Compiler auto-injects `partition_by`/`order_by` based on `UDFMeta.category`.
- 163 tests covering parser, compiler, engine, syntax, registry, and all UDF operators.

### Changed
- `CodeStr.__init__` signature: `index` is now a 2-tuple `(time_col, entity_col)`.
- TS/CS UDF functions accept `partition_by`/`order_by` keyword arguments (injected by compiler).
- `udf/__init__.py` uses explicit module imports instead of `import *`.
- Stricter ruff rules: added B, SIM, RUF, TCH.

### Removed
- **BREAKING**: Module-level `over` dict and `configure_over()` from `ts_udf` and `cs_udf`.

### Fixed
- `cs_midby`/`cs_meanby`: `[*over, *by]` was unpacking dict keys as column names —
  now uses `partition_by + list(by)`.

## [0.1.0] — Initial Release

### Added
- DSL parser (Lark LALR grammar): binary/unary/ternary operators, function calls, implicit multiplication
- AST compiler that translates expressions to Polars expressions
- `CodeStr` engine with two API modes: pure `compile()` and interactive `sql()`
- Expression-level caching with hash-based deduplication (alias-independent)
- Built-in UDF operators:
  - **base_udf**: arithmetic, logical, trigonometric, and horizontal operations
  - **cs_udf**: cross-section operators (rank, z-score, IC, quantile cut, etc.)
  - **ts_udf**: time-series operators (rolling mean, sum, delay, delta, etc.)
- UDF registry with `@udf` decorator for custom operators
- CI workflow with lint and format checks (ruff)
