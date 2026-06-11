# Changelog

## [0.1.0] — 2025-06-10

### Added
- Initial release of CodeStr expression compute engine
- DSL parser (Lark LALR grammar) with support for binary/unary/ternary operators, function calls, and implicit multiplication
- AST compiler that translates expressions to Polars expressions
- `CodeStr` engine with two API modes: pure `compile()` and interactive `sql()`
- Expression-level caching with hash-based deduplication (alias-independent)
- Built-in UDF operators:
  - **base_udf**: arithmetic, logical, trigonometric, and horizontal operations
  - **cs_udf**: cross-section operators (rank, z-score, IC, quantile cut, etc.)
  - **ts_udf**: time-series operators (rolling mean, sum, delay, delta, etc.)
- UDF registry with `@udf` decorator for custom operators
- CI workflow with lint and format checks (ruff)
