import pytest
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from predikit import ModelEnsemble, ModelTool, ToolRegistry


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@pytest.fixture
def registry():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    tool = ModelTool(
        model=clf,
        name="iris_classifier",
        description="Classify iris species",
        input_schema=IrisInput,
        output_name="species",
        output_description="Predicted species",
    )
    return ToolRegistry([tool])


def test_get_tool(registry):
    tool = registry.get("iris_classifier")
    assert tool.name == "iris_classifier"


def test_get_missing_raises(registry):
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_to_openai_returns_list(registry):
    schemas = registry.to_openai()
    assert isinstance(schemas, list)
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"


def test_get_ensemble():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    tool = ModelTool(
        model=clf,
        name="iris_classifier",
        description="Classify iris species",
        input_schema=IrisInput,
        output_name="species",
        output_description="Predicted species",
    )
    ensemble = ModelEnsemble(
        tools=[tool],
        name="iris_ensemble",
        description="Classify iris species with an ensemble",
        strategy="vote",
    )
    registry = ToolRegistry([tool], ensembles=[ensemble])
    assert registry.get("iris_ensemble") is ensemble


def test_duplicate_tool_names_raise():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    tool_a = ModelTool(
        model=clf,
        name="iris_classifier",
        description="Classify iris species",
        input_schema=IrisInput,
        output_name="species",
        output_description="Predicted species",
    )
    tool_b = ModelTool(
        model=clf,
        name="iris_classifier",
        description="Classify iris species again",
        input_schema=IrisInput,
        output_name="species",
        output_description="Predicted species",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry([tool_a, tool_b])


def test_duplicate_tool_and_ensemble_names_raise(registry):
    tool = registry.get("iris_classifier")
    ensemble = ModelEnsemble(
        tools=[tool],
        name=tool.name,
        description="Duplicate name",
        strategy="vote",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry([tool], ensembles=[ensemble])
