# Contributing to CodeStr

CodeStr is an expression compute engine for quantitative factor mining — contributions are welcome.

## Getting started

```bash
git clone https://github.com/huangbogeng/codestr.git
cd codestr
uv sync --extra dev
```

## Development workflow

```bash
# 1. Create a feature branch from master
git checkout -b feature/my-change

# 2. Make your changes

# 3. Run checks locally (same as CI)
uvx ruff check src/ tests/
uvx ruff format --check src/ tests/
uv run pytest tests/ -v

# 4. Commit and push
git commit -m "feat: describe your change"
git push -u origin feature/my-change
```

## Code style

- **Formatter**: [Ruff](https://docs.astral.sh/ruff/) — 100 char line length, double quotes
- **Type checker**: [mypy](https://www.mypy-lang.org/) — Python 3.10 target
- **Linter**: Ruff with E, F, I, N, W, UP, B, SIM, RUF, TCH rules
- **Tests**: pytest in `tests/` directory
- **Build**: [hatchling](https://hatch.pypa.io/)

CI will fail if any check doesn't pass.

## Testing

- Test files mirror the source structure: `tests/test_engine.py` tests `src/codestr/engine.py`
- Use `conftest.py` fixtures for shared test data
- The `reset_registry` fixture (autouse) resets the UDF registry between tests — register new UDFs in test functions, not at module level

```bash
uv run pytest tests/                     # all tests
uv run pytest tests/test_parser.py -v    # single file
uv run pytest tests/ -k "test_ts" -v     # filter by name
```

## Project architecture

See [CLAUDE.md](./CLAUDE.md) for a detailed walkthrough of the data flow, caching, and window configuration.

## Reporting issues

Please include:
- A minimal reproducible example
- Expected vs actual behavior
- Python version (`python --version`) and package versions (`uv pip list | grep -E "polars|codestr|lark"`)
