"""Tests for ModelTool(verbose=True) diagnostic output."""

from unittest.mock import MagicMock

from pydantic import BaseModel

from predikit import ModelTool


class MockInput(BaseModel):
    feature1: float
    feature2: float


def _verbose_tool():
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    return ModelTool(
        model=mock_model,
        name="test_tool",
        description="testing logs",
        input_schema=MockInput,
        output_name="prediction",
        output_description="The result of the mock test",
        verbose=True,
    )


def test_verbose_prints_invocation_prediction_and_latency(capsys):
    _verbose_tool().invoke({"feature1": 10.0, "feature2": 20.0})

    captured = capsys.readouterr()
    assert "[predikit] Invoking tool: test_tool" in captured.out
    assert "[predikit] Prediction: 1" in captured.out
    assert "Latency" in captured.out


def test_quiet_by_default(capsys):
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    tool = ModelTool(
        model=mock_model,
        name="test_tool",
        description="testing logs",
        input_schema=MockInput,
        output_name="prediction",
        output_description="The result of the mock test",
    )

    tool.invoke({"feature1": 10.0, "feature2": 20.0})

    assert capsys.readouterr().out == ""
