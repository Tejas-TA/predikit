# predikit — Claude Code context

## What this is

predikit wraps fitted scikit-learn / XGBoost models as LLM-callable tools with auto-generated JSON schemas, typed I/O via Pydantic, and first-class support for OpenAI function calling and LangChain.

## Project layout

```
src/predikit/          Core package (installed as `predikit`)
  tool.py              ModelTool — main public class
  registry.py          ToolRegistry — groups multiple tools for bulk export
  ensemble.py          ModelEnsemble — multi-model reconciliation strategies
  exceptions.py        LowConfidenceError
  coerce.py            LLM-string → Python type coercion (bool, int, float, str)
  introspect.py        Extract sklearn model metadata (feature names, task type)
  cli.py               `predikit inspect` CLI (click + joblib)
  exporters/
    openai.py          .to_openai() — OpenAI function-calling schema
    langchain.py       .to_langchain() — LangChain StructuredTool
  loaders/
    mlflow.py          from_mlflow() — load from MLflow Model Registry
    snowflake.py       from_snowflake() — load from Snowflake Model Registry
tests/                 pytest suite (mirrors src/ layout)
examples/              Standalone usage scripts (01–07)
```

## Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run tests
pytest --cov=src/predikit --cov-report=term-missing

# Lint (check only)
ruff check src/ tests/ examples/

# Format (check only)
ruff format --check src/ tests/ examples/

# Lint + format (auto-fix)
ruff check --fix src/ tests/ examples/
ruff format src/ tests/ examples/

# Type check
mypy src/predikit

# All checks via pre-commit
pre-commit run --all-files

# Install pre-commit hooks
pre-commit install
```

## Key conventions

- **Public API** — `ModelTool`, `ToolRegistry`, `ModelEnsemble`, `LowConfidenceError` are all re-exported from `predikit/__init__.py`. Everything else is internal.
- **Optional imports** — heavy optional deps (`langchain`, `xgboost`, `mlflow`, `snowflake`) are imported inside functions/methods, never at module level, so missing extras don't break core imports.
- **Feature name matching** — inputs map to model features by column name (using `feature_names_in_` from sklearn), not by position. Mismatches raise a `ValueError` with a clear diff.
- **Bool coercion** — `coerce_value()` converts LLM strings (`"yes"`, `"true"`, `"1"`, `"on"`) to Python bools before Pydantic validation.
- **Async** — `ainvoke()` is a thin wrapper over `invoke()` via `loop.run_in_executor`. No async-native I/O involved.
- **Ensemble strategies** — `"collect"`, `"mean"`, `"vote"`, `"weighted_mean"`, `"weighted_vote"`. Weighted strategies take a `weights` list parallel to `tools`.

## Release process

1. Bump `version` in `pyproject.toml` and `__version__` in `src/predikit/__init__.py`.
2. Add a section to `CHANGELOG.md`.
3. Commit, push to `main`, create a GitHub Release with the version tag (e.g. `v0.4.1`).
4. `publish.yml` auto-triggers a PyPI upload via OIDC trusted publishing.
