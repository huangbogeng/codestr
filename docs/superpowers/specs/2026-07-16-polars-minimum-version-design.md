# Polars Minimum Version Upgrade

## Scope

Raise CodeStr's Polars dependency minimum from `1.0` to `1.42.1` without an upper bound.

## Changes

- Set the project dependency to `polars>=1.42.1` in `pyproject.toml`.
- Refresh `uv.lock` so the development and CI environment resolves Polars 1.42.1.
- Do not change CodeStr behavior or add compatibility code in this change.

## Verification

- Run the complete pytest suite.
- Run Ruff lint and formatting checks.
- Build the source distribution and wheel.
- Confirm the installed Polars version is at least 1.42.1.
