import pytest
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

from predikit import ModelEnsemble, ModelTool

X, y = load_iris(return_X_y=True)
SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@pytest.fixture
def reg_tools():
    y_f = y.astype(float)
    return [
        ModelTool(
            model=LinearRegression().fit(X, y_f),
            name=f"reg_{i}",
            description="",
            input_schema=IrisInput,
            output_name="value",
            output_description="",
        )
        for i in range(2)
    ]


@pytest.fixture
def clf_tools():
    return [
        ModelTool(
            model=LogisticRegression(max_iter=200, C=c).fit(X, y),
            name=f"clf_{c}",
            description="",
            input_schema=IrisInput,
            output_name="species",
            output_description="",
        )
        for c in [0.5, 1.0]
    ]


# --- weighted_mean ---


def test_weighted_mean_identical_models_same_as_mean(reg_tools):
    equal = ModelEnsemble(
        tools=reg_tools, name="e", description="", strategy="weighted_mean", weights=[1.0, 1.0]
    )
    mean = ModelEnsemble(tools=reg_tools, name="e", description="", strategy="mean")
    assert abs(equal.invoke(SAMPLE)["value"] - mean.invoke(SAMPLE)["value"]) < 1e-9


def test_weighted_mean_full_weight_on_first_tool(reg_tools):
    ens = ModelEnsemble(
        tools=reg_tools, name="e", description="", strategy="weighted_mean", weights=[1.0, 0.0]
    )
    expected = reg_tools[0].invoke(SAMPLE)["value"]
    assert abs(ens.invoke(SAMPLE)["value"] - expected) < 1e-9


def test_weighted_mean_full_weight_on_second_tool(reg_tools):
    ens = ModelEnsemble(
        tools=reg_tools, name="e", description="", strategy="weighted_mean", weights=[0.0, 1.0]
    )
    expected = reg_tools[1].invoke(SAMPLE)["value"]
    assert abs(ens.invoke(SAMPLE)["value"] - expected) < 1e-9


def test_weighted_mean_known_ratio(reg_tools):
    # weights [3, 1] → result = (3*a + 1*b) / 4
    a = reg_tools[0].invoke(SAMPLE)["value"]
    b = reg_tools[1].invoke(SAMPLE)["value"]
    expected = (3 * a + 1 * b) / 4
    ens = ModelEnsemble(
        tools=reg_tools, name="e", description="", strategy="weighted_mean", weights=[3.0, 1.0]
    )
    assert abs(ens.invoke(SAMPLE)["value"] - expected) < 1e-9


# --- weighted_vote ---


def test_weighted_vote_returns_valid_class(clf_tools):
    ens = ModelEnsemble(
        tools=clf_tools, name="e", description="", strategy="weighted_vote", weights=[0.7, 0.3]
    )
    result = ens.invoke(SAMPLE)
    assert result["species"] in [0, 1, 2]


def test_weighted_vote_dominant_weight_wins():
    """A single high-weight tool should dominate over many low-weight ones."""
    # Build two tools: tool_a always predicts class 0, tool_b always predicts class 1.
    # We fake this by training on degenerate data.
    from sklearn.dummy import DummyClassifier

    tool_a = ModelTool(
        model=DummyClassifier(strategy="constant", constant=0).fit(X, y),
        name="a",
        description="",
        input_schema=IrisInput,
        output_name="species",
        output_description="",
    )
    tool_b = ModelTool(
        model=DummyClassifier(strategy="constant", constant=1).fit(X, y),
        name="b",
        description="",
        input_schema=IrisInput,
        output_name="species",
        output_description="",
    )
    # tool_a gets 0.9 weight, tool_b gets 0.1 → class 0 wins
    ens = ModelEnsemble(
        tools=[tool_a, tool_b],
        name="e",
        description="",
        strategy="weighted_vote",
        weights=[0.9, 0.1],
    )
    assert ens.invoke(SAMPLE)["species"] == 0

    # Flip weights → class 1 wins
    ens2 = ModelEnsemble(
        tools=[tool_a, tool_b],
        name="e",
        description="",
        strategy="weighted_vote",
        weights=[0.1, 0.9],
    )
    assert ens2.invoke(SAMPLE)["species"] == 1


# --- validation ---


def test_weights_wrong_length_raises(reg_tools):
    with pytest.raises(ValueError, match="weights length"):
        ModelEnsemble(
            tools=reg_tools, name="e", description="", strategy="weighted_mean", weights=[1.0]
        )


def test_negative_weight_raises(reg_tools):
    with pytest.raises(ValueError, match="non-negative"):
        ModelEnsemble(
            tools=reg_tools,
            name="e",
            description="",
            strategy="weighted_mean",
            weights=[1.0, -0.5],
        )


def test_zero_total_weight_raises(reg_tools):
    with pytest.raises(ValueError, match="positive"):
        ModelEnsemble(
            tools=reg_tools,
            name="e",
            description="",
            strategy="weighted_mean",
            weights=[0.0, 0.0],
        )


def test_weights_require_a_weighted_strategy(reg_tools):
    with pytest.raises(ValueError, match="only valid"):
        ModelEnsemble(
            tools=reg_tools, name="e", description="", strategy="mean", weights=[1.0, 1.0]
        )


def test_invalid_strategy_raises_at_init(reg_tools):
    with pytest.raises(ValueError, match="Unknown strategy"):
        ModelEnsemble(tools=reg_tools, name="e", description="", strategy="bad")
