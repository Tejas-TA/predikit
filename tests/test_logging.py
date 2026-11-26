"""Tests for ModelTool(verbose=True) diagnostic logging."""

import logging
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


def test_verbose_logs_invocation_prediction_and_latency(caplog):
    caplog.set_level(logging.INFO, logger="predikit.tool")
    _verbose_tool().invoke({"feature1": 10.0, "feature2": 20.0})

    messages = caplog.messages
    assert "Invoking tool test_tool with input {'feature1': 10.0, 'feature2': 20.0}" in messages
    assert any("Tool test_tool predicted 1 in" in message for message in messages)


def test_quiet_by_default(caplog):
    caplog.set_level(logging.INFO, logger="predikit.tool")
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

    assert caplog.messages == []
