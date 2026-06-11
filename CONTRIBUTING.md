# Contributing to CodeStr

Thanks for your interest in contributing! CodeStr is an expression compute engine for quantitative factor mining.

## Getting started

```bash
git clone https://github.com/huangbogeng/codestr.git
cd codestr
uv venv
uv pip install -e ".[dev]"
```

## Development workflow

1. Create a feature branch from `master`.
2. Make your changes.
3. Run lint and tests before committing:

```bash
uvx ruff check src/       # lint
uvx ruff format --check src/  # format check
uv run pytest tests/ -v      # tests
```

4. Commit and open a pull request.

## Code style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. CI will fail if your code doesn't pass both checks.

- Line length: 100 characters
- Quote style: double quotes
- Python target: 3.10+

## Testing

Write tests for new features or bug fixes. Tests live in `tests/` and use `pytest`.

## Reporting issues

Please include:
- A minimal reproducible example
- Expected vs actual behavior
- Your Python and package versions
