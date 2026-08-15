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


@pytest.fixture
def registry_module(monkeypatch):
    """A module exposing both a ToolRegistry instance and a factory returning one."""
    import sys
    import types

    from pydantic import create_model

    from predikit import ModelTool, ToolRegistry

    class EchoModel:
        def predict(self, X):
            return [float(X[0][0])]

    def build():
        tool = ModelTool(
            model=EchoModel(),
            name="echo_value",
            description="Echo a numeric value.",
            input_schema=create_model("Input", value=(float, ...)),
            output_name="value",
            output_description="The value.",
        )
        return ToolRegistry([tool])

    module = types.ModuleType("predikit_serve_fixture")
    module.registry = build()
    module.make_registry = build
    monkeypatch.setitem(sys.modules, "predikit_serve_fixture", module)
    return module


@pytest.fixture
def captured_run(monkeypatch):
    """Stop short of actually serving; record how the server was configured."""
    calls = {}

    def fake_run(self, transport):
        calls["transport"] = transport
        calls["host"] = self.settings.host
        calls["port"] = self.settings.port
        calls["tools"] = sorted(self._tool_manager._tools)

    pytest.importorskip("mcp")
    from mcp.server.fastmcp import FastMCP

    monkeypatch.setattr(FastMCP, "run", fake_run)
    return calls


def test_serve_loads_a_registry_instance(runner, registry_module, captured_run):
    result = runner.invoke(cli, ["serve", "predikit_serve_fixture:registry"])
    assert result.exit_code == 0, result.output
    assert captured_run["transport"] == "stdio"
    assert captured_run["tools"] == ["echo_value"]


def test_serve_calls_a_registry_factory(runner, registry_module, captured_run):
    result = runner.invoke(cli, ["serve", "predikit_serve_fixture:make_registry"])
    assert result.exit_code == 0, result.output
    assert captured_run["tools"] == ["echo_value"]


def test_serve_passes_host_and_port(runner, registry_module, captured_run):
    result = runner.invoke(
        cli,
        [
            "serve",
            "predikit_serve_fixture:registry",
            "--transport",
            "streamable-http",
            "--host",
            "0.0.0.0",
            "--port",
            "9123",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (captured_run["transport"], captured_run["host"], captured_run["port"]) == (
        "streamable-http",
        "0.0.0.0",
        9123,
    )


def test_serve_reports_an_unimportable_module(runner):
    result = runner.invoke(cli, ["serve", "no_such_module_xyz:registry"])
    assert result.exit_code != 0
    assert "Could not import module 'no_such_module_xyz'" in result.output


def test_serve_reports_a_missing_attribute(runner, registry_module):
    result = runner.invoke(cli, ["serve", "predikit_serve_fixture:nope"])
    assert result.exit_code != 0
    assert "has no attribute 'nope'" in result.output


def test_serve_does_not_disguise_errors_from_the_registry_factory(runner, monkeypatch):
    """An ImportError raised *inside* the target must not read as a bad target."""
    import sys
    import types

    def broken():
        raise ImportError("optional backend missing")

    module = types.ModuleType("predikit_broken_fixture")
    module.make_registry = broken
    monkeypatch.setitem(sys.modules, "predikit_broken_fixture", module)

    result = runner.invoke(cli, ["serve", "predikit_broken_fixture:make_registry"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ImportError)
    assert "Could not import module" not in result.output
