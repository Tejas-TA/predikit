import asyncio

import pytest
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from predikit import LowConfidenceError, ModelEnsemble, ModelTool


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


SAMPLES = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.0, "sepal_width": 2.2, "petal_length": 4.0, "petal_width": 1.0},
]


@pytest.fixture
def tool():
    X, y = load_iris(return_X_y=True)
    return ModelTool(
        model=LogisticRegression(max_iter=200).fit(X, y),
        name="iris_classifier",
        description="Classify iris species.",
        input_schema=IrisInput,
        output_name="species",
        output_description="Species index.",
    )


def test_invoke_many_matches_individual_predictions(tool):
    assert tool.invoke_many(SAMPLES) == [tool.invoke(sample) for sample in SAMPLES]


def test_invoke_many_calls_predict_once(tool, monkeypatch):
    calls = 0
    original_predict = tool.model.predict

    def count_predict(X):
        nonlocal calls
        calls += 1
        return original_predict(X)

    monkeypatch.setattr(tool.model, "predict", count_predict)
    result = tool.invoke_many(SAMPLES)
    assert calls == 1
    assert len(result) == len(SAMPLES)


def test_invoke_many_empty_batch_skips_model(tool, monkeypatch):
    monkeypatch.setattr(tool.model, "predict", lambda X: pytest.fail("predict should not run"))
    assert tool.invoke_many([]) == []


def test_invoke_many_reports_invalid_input_index(tool):
    with pytest.raises(ValueError, match="at index 1"):
        tool.invoke_many([SAMPLES[0], {"sepal_length": "not-a-number"}])


def test_invoke_many_rejects_non_dictionary_item(tool):
    with pytest.raises(TypeError, match="at index 1"):
        tool.invoke_many([SAMPLES[0], "invalid"])


def test_ainvoke_many_matches_invoke_many(tool):
    assert asyncio.run(tool.ainvoke_many(SAMPLES)) == tool.invoke_many(SAMPLES)


def test_invoke_many_preserves_low_confidence_behavior():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    warn_tool = ModelTool(
        model=model,
        name="uncertain",
        description="",
        input_schema=IrisInput,
        output_name="species",
        output_description="",
        confidence_threshold=1.0,
        on_low_confidence="warn",
    )
    results = warn_tool.invoke_many(SAMPLES)
    assert all(result["_low_confidence"] is True for result in results)

    raising_tool = ModelTool(
        model=model,
        name="raising",
        description="",
        input_schema=IrisInput,
        output_name="species",
        output_description="",
        confidence_threshold=1.0,
        on_low_confidence="raise",
    )
    with pytest.raises(LowConfidenceError):
        raising_tool.invoke_many(SAMPLES)


def test_ensemble_invoke_many_reconciles_each_row(tool):
    second_tool = ModelTool(
        model=tool.model,
        name="iris_classifier_second",
        description="",
        input_schema=IrisInput,
        output_name="species",
        output_description="Species index.",
    )
    ensemble = ModelEnsemble(
        tools=[tool, second_tool], name="iris_vote", description="", strategy="vote"
    )
    assert ensemble.invoke_many(SAMPLES) == [ensemble.invoke(sample) for sample in SAMPLES]
    assert asyncio.run(ensemble.ainvoke_many(SAMPLES)) == ensemble.invoke_many(SAMPLES)
