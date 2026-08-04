<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/Tejas-TA/predikit/main/docs/logo.gif">
    <img src="https://raw.githubusercontent.com/Tejas-TA/predikit/main/docs/logo.png" alt="predikit" width="500"/>
  </picture>
</p>

<p align="center">
  <strong>The bridge between your ML models and LLM agents.</strong><br/>
  Wrap any trained scikit-learn or XGBoost model as an LLM-callable tool —<br/>
  auto-generated JSON schemas, typed I/O, zero boilerplate.
</p>

<p align="center">
  <a href="https://pypi.org/project/predikit/"><img src="https://img.shields.io/pypi/v/predikit.svg" alt="PyPI version"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"/></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"/></a>
</p>

<p align="center">
  <a href="https://github.com/Tejas-TA/predikit/stargazers"><img src="https://img.shields.io/github/stars/Tejas-TA/predikit?style=social" alt="GitHub Stars"/></a>
  <a href="https://github.com/Tejas-TA/predikit/network/members"><img src="https://img.shields.io/github/forks/Tejas-TA/predikit?style=social" alt="GitHub Forks"/></a>
  <a href="https://pepy.tech/project/predikit"><img src="https://static.pepy.tech/personalized-badge/predikit?period=week&units=international_system&left_color=grey&right_color=blue&left_text=weekly+downloads" alt="Weekly Downloads"/></a>
  <a href="https://pepy.tech/project/predikit"><img src="https://static.pepy.tech/personalized-badge/predikit?period=month&units=international_system&left_color=grey&right_color=blue&left_text=monthly+downloads" alt="Monthly Downloads"/></a>
  <a href="https://pepy.tech/project/predikit"><img src="https://static.pepy.tech/personalized-badge/predikit?period=total&units=international_system&left_color=grey&right_color=blue&left_text=total+downloads" alt="Total Downloads"/></a>
</p>

```python
tool = ModelTool(model=clf, name="classify_iris", ...)
tool.to_openai()              # OpenAI function schema, ready to pass to the API
tool.invoke({"sqft": 2200})   # → {"price_usd": 370730}
```

---

## Table of Contents
- [Why predikit?](#why-predikit)
- [Works with](#works-with)
- [Install](#install)
- [Quick start](#quick-start)
- [Core API](#core-api)
- [Field naming rule](#field-naming-rule)
- [Cookbook](#cookbook)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why predikit?

Most ML pipelines stop at `.predict()`. Getting that prediction callable by an LLM agent — with validated inputs, correct types, and a schema the model can reason about — requires glue code that's tedious to write and easy to get wrong.

predikit handles that layer for you:

|  | Without predikit | With predikit |
|--|------------------|---------------|
| **Schema** | Hand-write JSON Schema for every model | Auto-generated from your Pydantic `BaseModel` |
| **Type safety** | Manual casting, silent failures | Pydantic v2 validation with clear error messages |
| **LLM integration** | OpenAI / LangChain boilerplate per model | `.to_openai()` / `.to_langchain()` in one line |
| **Ensemble routing** | Custom aggregation logic per project | `ModelEnsemble` with 5 built-in strategies |
| **Confidence handling** | Write your own threshold checks | `confidence_threshold` + `on_low_confidence` |
| **Model registries** | Manual MLflow / Snowflake registry calls | `from_mlflow()` / `from_snowflake()` loaders |
| **Async** | `asyncio.get_event_loop().run_in_executor(...)` | `await tool.ainvoke(inputs)` |

---

## Works with

predikit is designed to plug into the tools your team already uses:

**Models** — scikit-learn · XGBoost

**LLM frameworks** — OpenAI function calling · LangChain · any tool-calling API that accepts JSON Schema

**Model registries** — MLflow Model Registry · Snowflake Model Registry

**Validation** — Pydantic v2

**Async** — `asyncio`-compatible via `ainvoke()`

---

## Install

```bash
pip install predikit

# Optional extras
pip install predikit[xgboost]    # XGBoost support
pip install predikit[langchain]  # LangChain StructuredTool export
pip install predikit[mlflow]     # MLflow Model Registry loader
pip install predikit[snowflake]  # Snowflake Model Registry loader
```

---

## Quick start

```python
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from predikit import ModelTool

# Train
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=200).fit(X, y)

# Define what the LLM will pass in
class IrisInput(BaseModel):
    sepal_length: float = Field(description="Sepal length in cm")
    sepal_width:  float = Field(description="Sepal width in cm")
    petal_length: float = Field(description="Petal length in cm")
    petal_width:  float = Field(description="Petal width in cm")

# Wrap the model
tool = ModelTool(
    model=clf,
    name="classify_iris",
    description="Classify an iris flower: 0=setosa, 1=versicolor, 2=virginica.",
    input_schema=IrisInput,
    output_name="species",
    output_description="Predicted species index",
)

# Get an OpenAI-ready schema
import json
print(json.dumps(tool.to_openai(), indent=2))

# Call it directly
tool.invoke({
    "sepal_length": 5.1, "sepal_width": 3.5,
    "petal_length": 1.4, "petal_width": 0.2,
})
# → {"species": 0}
```

---

## Core API

### `ModelTool`

```python
ModelTool(
    model,               # fitted sklearn-compatible estimator
    name: str,           # tool name the LLM sees
    description: str,    # tool description the LLM sees
    input_schema,        # Pydantic BaseModel describing inputs
    output_name: str,    # key for the prediction in the returned dict
    output_description: str,
)
```

| Method | Returns | What it does |
|--------|---------|--------------|
| `.invoke(input_dict)` | `dict` | Validates → predicts → returns `{output_name: value}` |
| `.ainvoke(input_dict)` | `dict` | Async version of `.invoke()` |
| `.to_openai()` | `dict` | OpenAI function-calling schema |
| `.to_langchain()` | `StructuredTool` | LangChain tool |
| `.to_callable()` | `Callable` | Plain Python function |

### `ToolRegistry`

Group multiple tools for bulk export:

```python
registry = ToolRegistry([price_tool, risk_tool])
registry.to_openai()     # → list[dict], pass directly to OpenAI
registry.to_langchain()  # → list[StructuredTool]
registry.get("name")     # → ModelTool
```

### `ModelEnsemble`

Call multiple models and reconcile their outputs in one step:

```python
ModelEnsemble(
    tools: list[ModelTool],   # models to run in parallel
    name: str,                # ensemble tool name the LLM sees
    description: str,
    strategy: str,            # "collect" | "mean" | "vote" | "weighted_mean" | "weighted_vote"
    weights: list[float],     # optional, for weighted strategies
)
```

| Strategy | Behaviour |
|----------|-----------|
| `"collect"` | Merges all outputs into one dict (tools can have different `output_name`) |
| `"mean"` | Averages numeric outputs (all tools must share `output_name`) |
| `"vote"` | Majority class vote (all tools must share `output_name`) |
| `"weighted_mean"` | Weighted average — provide a `weights` list |
| `"weighted_vote"` | Weighted majority vote — provide a `weights` list |

`ModelEnsemble` exposes the same `.invoke()`, `.ainvoke()`, `.to_openai()`, and `.to_langchain()` interface as `ModelTool`.

---

## Field naming rule

**Your Pydantic schema field names must exactly match the column names the model was trained on.**

predikit maps inputs to features by name, not position. If you trained on a DataFrame with columns `["sqft", "bedrooms"]`, your schema fields must be `sqft` and `bedrooms` — not `sq_ft`, not `Sqft`.

```python
# ✓ Columns match: sqft, bedrooms, bathrooms
class GoodInput(BaseModel):
    sqft:      float
    bedrooms:  float
    bathrooms: float

# ✗ Name mismatch — raises ValueError at runtime
class BadInput(BaseModel):
    square_footage: float  # model expects "sqft"
    beds:           float  # model expects "bedrooms"
    baths:          float  # model expects "bathrooms"
```

When there's a mismatch, predikit tells you exactly which names are wrong:

```
ValueError: Input schema is missing model features: ['sqft', 'bedrooms'].
Schema has: ['square_footage', 'beds', 'bathrooms'], model expects: ['sqft', 'bedrooms', 'bathrooms']
```

> **Tip:** If you trained with a numpy array (no DataFrame), predikit has no feature names to check — it uses your schema's field definition order instead.

---

## Cookbook

### XGBoost regression

```python
from xgboost import XGBRegressor
from predikit import ModelTool

reg = XGBRegressor().fit(X_train, y_train)

class HouseInput(BaseModel):
    sqft:       float
    bedrooms:   float
    year_built: float

tool = ModelTool(
    model=reg,
    name="price_estimate",
    description="Predict home price in USD.",
    input_schema=HouseInput,
    output_name="price_usd",
    output_description="Predicted sale price in USD",
)
```

### Multiple tools in one registry

```python
registry = ToolRegistry([price_tool, risk_tool, demand_tool])

# OpenAI
response = client.chat.completions.create(
    model="gpt-4o",
    tools=registry.to_openai(),
    ...
)

# LangChain
agent = initialize_agent(tools=registry.to_langchain(), ...)
```

### MCP server

Expose the same registry to MCP-compatible clients with the optional MCP extra:

```bash
pip install predikit[mcp]
predikit serve my_tools:registry
```

`my_tools:registry` must point to a `ToolRegistry`, or to a zero-argument function
that returns one. The default transport is `stdio`; use `--transport streamable-http`
when serving over HTTP. MCP tools use the same Pydantic input validation and model
execution as direct `invoke()` calls.

### Bool inputs from an LLM

LLMs sometimes return `"yes"`, `"true"`, or `"1"` for boolean fields. predikit coerces these automatically before Pydantic validation:

```python
class Input(BaseModel):
    has_pool: bool

tool.invoke({"has_pool": "yes"})   # → coerced to True
tool.invoke({"has_pool": "false"}) # → coerced to False
tool.invoke({"has_pool": "maybe"}) # → raises ValueError with clear message
```

Supported strings: `true/false`, `yes/no`, `1/0`, `on/off`.

### Confidence-aware routing

Route uncertain predictions to a fallback tool, or raise an error the agent can catch:

```python
from predikit import ModelTool, LowConfidenceError

tool = ModelTool(
    model=clf,
    name="churn_risk",
    description="Predict member churn risk.",
    input_schema=MemberInput,
    output_name="churn_probability",
    output_description="Probability of churn (0–1)",
    confidence_threshold=0.80,       # classifiers with predict_proba only
    on_low_confidence="warn",        # "warn" | "raise" | "fallback"
    fallback_tool=rule_based_tool,   # used when mode="fallback"
)

result = tool.invoke(inputs)
if result.get("_low_confidence"):
    print(f"Uncertain ({result['_confidence']:.2f}) — consider routing to a human")
```

| mode | behaviour |
|------|-----------|
| `"warn"` | returns prediction + `_confidence` + `_low_confidence: True` |
| `"raise"` | raises `LowConfidenceError` |
| `"fallback"` | invokes `fallback_tool` and returns its result |

Only applies to classifiers that implement `predict_proba`. Regressors are unaffected.

### Multi-model ensemble

```python
from predikit import ModelEnsemble, ToolRegistry

ensemble = ModelEnsemble(
    tools=[price_tool_a, price_tool_b],
    name="averaged_price",
    description="Ensemble price: mean of two XGBoost models.",
    strategy="mean",              # "collect" | "mean" | "vote"
)

result  = ensemble.invoke(inputs)  # → {"price_usd": 370112}
schema  = ensemble.to_openai()     # works exactly like ModelTool
```

Register ensembles alongside individual tools:

```python
registry = ToolRegistry(tools=[price_tool], ensembles=[ensemble])
registry.to_openai()  # includes both tools and ensembles
```

### MLflow Model Registry loader

Load a registered MLflow model directly — no manual `.load_model()` call:

```python
from predikit.loaders import from_mlflow

tool = from_mlflow(
    model_uri="models:/churn-classifier/Production",
    name="churn_risk",
    description="Predict member churn probability.",
    input_schema=MemberInput,
    output_name="churn_probability",
    output_description="Churn probability 0–1",
)

tool.invoke({"tenure_months": 24, "trips_last_year": 2, "avg_spend": 500})
# → {"churn_probability": 0.73}
```

The loader auto-detects `classes_` and `feature_names_in_` from the underlying sklearn model, so confidence routing and ensemble work unchanged. Requires `pip install predikit[mlflow]`.

### Snowflake Model Registry loader

Load a model registered in the Snowflake Model Registry via the Snowpark ML Python library:

```python
from predikit.loaders import from_snowflake

tool = from_snowflake(
    session=snowpark_session,
    model_name="VACATION_CHURN",
    model_version="V3",
    name="churn_risk",
    description="Churn classifier.",
    input_schema=MemberInput,
    output_name="churn_probability",
    output_description="Churn probability 0–1",
    output_method="predict",   # method to call on the Snowflake model object
)
```

Pass `output_method="predict_proba"` or any other method your Snowflake model exposes. The returned `ModelTool` is identical to one built directly — all exporters, confidence routing, and ensemble strategies work as-is. Requires `pip install predikit[snowflake]`.

### End-to-end demo

See [`examples/03_orlando_real_estate.py`](examples/03_orlando_real_estate.py) for a full walkthrough: synthetic dataset → XGBoost training → `ModelTool` → registry → OpenAI schema → prediction.

---

## Roadmap

### Shipped

- [x] OpenAI function-calling schema export
- [x] LangChain `StructuredTool` export
- [x] Pydantic v2 typed I/O validation
- [x] Multi-model `ModelEnsemble` — collect / mean / vote / weighted variants
- [x] Confidence routing — `warn` / `raise` / `fallback`
- [x] Async `ainvoke()` via thread pool executor
- [x] MLflow Model Registry loader
- [x] Snowflake Model Registry loader
- [x] `predikit inspect` CLI

### Planned

- [ ] HuggingFace / PyTorch / TensorFlow model support
- [ ] Streaming inference support
- [ ] OpenAI Assistants API integration
- [x] MCP server mode (stdio and Streamable HTTP)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines. The [CHANGELOG](CHANGELOG.md) tracks notable changes per release.

Issues and PRs are welcome — if you're wrapping a model type or registry that predikit doesn't support yet, open a discussion.

## License

MIT © Tejas Tumakuru Ashok
