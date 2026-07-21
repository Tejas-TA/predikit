import pytest
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

from predikit import LowConfidenceError, ModelTool

X_iris, y_iris = load_iris(return_X_y=True)
SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@pytest.fixture
def clf():
    return LogisticRegression(max_iter=200).fit(X_iris, y_iris)


def _make(clf, threshold, mode, fallback=None):
    return ModelTool(
        model=clf,
        name="iris",
        description="test",
        input_schema=IrisInput,
        output_name="species",
        output_description="species",
        confidence_threshold=threshold,
        on_low_confidence=mode,
        fallback_tool=fallback,
    )


def test_above_threshold_returns_clean_result(clf):
    result = _make(clf, threshold=0.0, mode="warn").invoke(SAMPLE)
    assert "species" in result
    assert "_low_confidence" not in result


def test_warn_mode_returns_flags(clf):
    # threshold=1.0 is impossible to exceed for real probabilities, so always fires
    result = _make(clf, threshold=1.0, mode="warn").invoke(SAMPLE)
    assert result["_low_confidence"] is True
    assert 0.0 <= result["_confidence"] <= 1.0
    assert "species" in result


def test_raise_mode_raises_low_confidence_error(clf):
    with pytest.raises(LowConfidenceError, match="below threshold"):
        _make(clf, threshold=1.0, mode="raise").invoke(SAMPLE)


def test_fallback_mode_invokes_fallback_tool(clf):
    fallback_clf = LogisticRegression(max_iter=200).fit(X_iris, y_iris)
    fallback = ModelTool(
        model=fallback_clf,
        name="fallback",
        description="fallback",
        input_schema=IrisInput,
        output_name="species",
        output_description="species",
    )
    result = _make(clf, threshold=1.0, mode="fallback", fallback=fallback).invoke(SAMPLE)
    assert "species" in result
    assert "_low_confidence" not in result


def test_fallback_without_fallback_tool_raises_at_init(clf):
    with pytest.raises(ValueError, match="fallback_tool"):
        _make(clf, threshold=1.0, mode="fallback", fallback=None)


def test_no_threshold_skips_confidence_check(clf):
    result = _make(clf, threshold=None, mode="raise").invoke(SAMPLE)
    assert "species" in result  # raise mode never triggered


def test_confidence_not_applied_to_regressor():
    reg = LinearRegression().fit(X_iris, y_iris.astype(float))
    tool = ModelTool(
        model=reg,
        name="reg",
        description="test",
        input_schema=IrisInput,
        output_name="value",
        output_description="value",
        confidence_threshold=1.0,
        on_low_confidence="raise",  # would fire for a classifier
    )
    result = tool.invoke(SAMPLE)  # must NOT raise
    assert "value" in result


def test_invalid_on_low_confidence_raises_at_construction(clf):
    with pytest.raises(ValueError, match="on_low_confidence"):
        _make(clf, threshold=0.9, mode="explode")


def test_invalid_confidence_threshold_raises_at_construction(clf):
    with pytest.raises(ValueError, match="confidence_threshold"):
        _make(clf, threshold=1.1, mode="warn")


def test_confidence_threshold_on_regressor_warns():
    reg = LinearRegression().fit(X_iris, y_iris.astype(float))
    with pytest.warns(UserWarning, match="regressor"):
        ModelTool(
            model=reg,
            name="reg",
            description="test",
            input_schema=IrisInput,
            output_name="value",
            output_description="value",
            confidence_threshold=0.9,
        )


def test_confidence_threshold_without_predict_proba_warns():
    clf_no_proba = SVC(probability=False).fit(X_iris, y_iris)
    with pytest.warns(UserWarning, match="predict_proba"):
        ModelTool(
            model=clf_no_proba,
            name="svc",
            description="test",
            input_schema=IrisInput,
            output_name="species",
            output_description="species",
            confidence_threshold=0.9,
        )
