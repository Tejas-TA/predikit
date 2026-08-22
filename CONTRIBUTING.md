# Contributing to predikit

Thanks for your interest in contributing.

## Prerequisites

- Python 3.10+
- [pre-commit](https://pre-commit.com/) (`pip install pre-commit`)

## Development setup

```bash
git clone https://github.com/Tejas-TA/predikit
cd predikit
pip install -e ".[dev]"
pre-commit install
```

`pre-commit install` wires up ruff and mypy to run automatically before every commit.

## Running tests

```bash
pytest --cov=src/predikit --cov-report=term-missing
```

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [mypy](https://mypy.readthedocs.io/) for type checking. All three are configured in `pyproject.toml`.

```bash
# Check and auto-fix
ruff check --fix src/ tests/ examples/
ruff format src/ tests/ examples/

# Type check
mypy src/predikit
```

Or run everything at once via pre-commit:

```bash
pre-commit run --all-files
```

CI enforces all three checks on every push and PR (see `.github/workflows/test.yml`).

## Pull request guidelines

- Target `main`.
- Keep changes focused — one feature or fix per PR.
- Add or update tests for any changed behaviour.
- Run `pre-commit run --all-files` before pushing; CI will catch failures regardless.

## Adding a new exporter or loader

- Exporters live in `src/predikit/exporters/` and should expose a single function consumed by `ModelTool`.
- Loaders live in `src/predikit/loaders/` and should return a `ModelTool` instance.
- Any new optional dependency must be added to both the relevant optional group and `dev` in `pyproject.toml`, and imported inside the function body (not at module level).

## Release process (maintainers)

1. Bump `version` in `pyproject.toml` and `__version__` in `src/predikit/__init__.py`.
2. Add a section to `CHANGELOG.md`.
3. Merge to `main`, create a GitHub Release tagged `vX.Y.Z`.
4. `publish.yml` automatically publishes to PyPI via OIDC trusted publishing.
