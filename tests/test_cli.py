import json

import pytest

joblib = pytest.importorskip("joblib")
click_testing = pytest.importorskip("click.testing")

from click.testing import CliRunner  # noqa: E402
from sklearn.datasets import load_iris  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402

from predikit.cli import cli  # noqa: E402


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def reg_pkl(tmp_path):
    """Regression model fitted with named features, so it has feature_names_in_."""
    X, y = load_iris(return_X_y=True, as_frame=True)
    model = LinearRegression().fit(X, y.astype(float))
    path = tmp_path / "reg.pkl"
    joblib.dump(model, path)
    return str(path)


@pytest.fixture
def clf_pkl(tmp_path):
    """Classifier fitted with named features."""
    X, y = load_iris(return_X_y=True, as_frame=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    path = tmp_path / "clf.pkl"
    joblib.dump(model, path)
    return str(path)


@pytest.fixture
def unnamed_pkl(tmp_path):
    """Model fitted without named features (numpy array)."""
    X, y = load_iris(return_X_y=True)
    model = LinearRegression().fit(X, y.astype(float))
    path = tmp_path / "unnamed.pkl"
    joblib.dump(model, path)
    return str(path)


def test_inspect_regression_shows_model_type(runner, reg_pkl):
    result = runner.invoke(cli, ["inspect", reg_pkl])
    assert result.exit_code == 0, result.output
    assert "LinearRegression" in result.output
    assert "regression" in result.output


def test_inspect_classifier_shows_classes(runner, clf_pkl):
    result = runner.invoke(cli, ["inspect", clf_pkl])
    assert result.exit_code == 0, result.output
    assert "classification" in result.output
    assert "Classes" in result.output


def test_inspect_shows_feature_names(runner, reg_pkl):
    result = runner.invoke(cli, ["inspect", reg_pkl])
    assert "sepal length (cm)" in result.output


def test_inspect_emits_valid_openai_schema(runner, reg_pkl):
    result = runner.invoke(cli, ["inspect", reg_pkl])
    assert result.exit_code == 0
    # Extract JSON from output (everything after "OpenAI schema:\n")
    json_part = result.output.split("OpenAI schema:\n", 1)[1].strip()
    schema = json.loads(json_part)
    assert schema["type"] == "function"
    assert "parameters" in schema["function"]


def test_inspect_custom_name_and_description(runner, reg_pkl):
    result = runner.invoke(
        cli, ["inspect", reg_pkl, "--name", "my_tool", "--description", "my desc"]
    )
    assert result.exit_code == 0
    json_part = result.output.split("OpenAI schema:\n", 1)[1].strip()
    schema = json.loads(json_part)
    assert schema["function"]["name"] == "my_tool"
    assert schema["function"]["description"] == "my desc"


def test_inspect_unnamed_model_skips_schema(runner, unnamed_pkl):
    result = runner.invoke(cli, ["inspect", unnamed_pkl])
    assert result.exit_code == 0
    assert "unavailable" in result.output


def test_inspect_missing_file(runner):
    result = runner.invoke(cli, ["inspect", "does_not_exist.pkl"])
    assert result.exit_code != 0


def test_serve_requires_module_target(runner):
    result = runner.invoke(cli, ["serve", "not-a-module-target"])
    assert result.exit_code != 0
    assert "MODULE:ATTRIBUTE" in result.output
