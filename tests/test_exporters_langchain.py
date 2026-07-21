import asyncio

import pytest
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from predikit import ModelTool


class IrisInput(BaseModel):
    sepal_length: float = Field(description="Sepal length in cm")
    sepal_width: float = Field(description="Sepal width in cm")
    petal_length: float = Field(description="Petal length in cm")
    petal_width: float = Field(description="Petal width in cm")


SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


def _make_tool() -> ModelTool:
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    return ModelTool(
        model=clf,
        name="iris_classifier",
        description="Classify iris species from petal/sepal measurements",
        input_schema=IrisInput,
        output_name="species",
        output_description="Predicted species index",
    )


def test_to_langchain_returns_structured_tool():
    from langchain_core.tools import StructuredTool

    lc_tool = _make_tool().to_langchain()
    assert isinstance(lc_tool, StructuredTool)
    assert lc_tool.name == "iris_classifier"
    assert lc_tool.description == "Classify iris species from petal/sepal measurements"
    assert lc_tool.args_schema is IrisInput


def test_to_langchain_sync_invoke_runs_prediction():
    lc_tool = _make_tool().to_langchain()
    result = lc_tool.invoke(SAMPLE)
    assert "species" in result


def test_to_langchain_async_invoke_runs_prediction():
    lc_tool = _make_tool().to_langchain()
    result = asyncio.run(lc_tool.ainvoke(SAMPLE))
    assert "species" in result


def test_to_langchain_missing_dependency_raises(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_core.tools":
            raise ImportError("no langchain_core")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="langchain-core is required"):
        _make_tool().to_langchain()
