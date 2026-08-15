"""MCP integration tests.

These run against the real MCP SDK (a dev dependency) rather than a stand-in for
FastMCP, so that schema-generation regressions are actually caught.
"""

import asyncio
import sys

import pytest
from pydantic import create_model

from predikit import ModelEnsemble, ModelTool, ToolRegistry, create_mcp_server

pytest.importorskip("mcp", reason="the mcp extra is required for MCP integration tests")


class EchoModel:
    """Returns the first feature, doubled."""

    def predict(self, X):
        return [float(X[0][0]) * 2]


class LowConfidenceClassifier:
    classes_ = ["a", "b"]

    def predict(self, X):
        return ["a"]

    def predict_proba(self, X):
        return [[0.51, 0.49]]


def make_tool(name="echo_value", output_name="value", model=None):
    schema = create_model("Input", value=(float, ...), flag=(bool, False))
    return ModelTool(
        model=model if model is not None else EchoModel(),
        name=name,
        description="Echo a numeric value.",
        input_schema=schema,
        output_name=output_name,
        output_description="The doubled value.",
    )


@pytest.fixture
def registry():
    return ToolRegistry([make_tool()])


def call(server, name, arguments):
    return asyncio.run(server.call_tool(name, arguments))


def list_tools(server):
    return asyncio.run(server.list_tools())


def test_registry_tools_are_exposed(registry):
    server = create_mcp_server(registry, name="Test Server")
    tools = list_tools(server)

    assert server.name == "Test Server"
    assert [t.name for t in tools] == ["echo_value"]
    assert tools[0].description == "Echo a numeric value."


def test_input_schema_mirrors_the_pydantic_model(registry):
    schema = list_tools(create_mcp_server(registry))[0].inputSchema

    assert schema["properties"]["value"]["type"] == "number"
    assert schema["properties"]["flag"]["type"] == "boolean"
    assert schema["properties"]["flag"]["default"] is False
    assert schema["required"] == ["value"]


def test_tool_advertises_an_output_schema(registry):
    # Regression: a bare `dict` return annotation left outputSchema null, so MCP
    # clients saw no output contract at all.
    schema = list_tools(create_mcp_server(registry))[0].outputSchema

    assert schema is not None
    assert schema["properties"]["value"]["description"] == "The doubled value."
    assert schema["required"] == ["value"]


def test_call_returns_structured_content(registry):
    _, structured = call(create_mcp_server(registry), "echo_value", {"value": 3})

    assert structured == {"value": 6.0}


def test_llm_string_booleans_are_accepted(registry):
    server = create_mcp_server(registry)

    for truthy in ("yes", "true", "on", "1"):
        _, structured = call(server, "echo_value", {"value": 2, "flag": truthy})
        assert structured == {"value": 4.0}


def test_low_confidence_keys_survive_the_output_schema():
    tool = ModelTool(
        model=LowConfidenceClassifier(),
        name="classify",
        description="Classify.",
        input_schema=create_model("Input", value=(float, ...)),
        output_name="label",
        output_description="The label.",
        confidence_threshold=0.9,
    )
    _, structured = call(create_mcp_server(ToolRegistry([tool])), "classify", {"value": 1})

    # An output model without extra="allow" would validate these away.
    assert structured == {"label": "a", "_confidence": 0.51, "_low_confidence": True}


def test_collect_ensemble_declares_every_member_output():
    ensemble = ModelEnsemble(
        tools=[make_tool("a", output_name="first"), make_tool("b", output_name="second")],
        name="both",
        description="Both.",
        strategy="collect",
    )
    server = create_mcp_server(ToolRegistry([], ensembles=[ensemble]))
    schema = list_tools(server)[0].outputSchema

    assert sorted(schema["properties"]) == ["first", "second"]
    _, structured = call(server, "both", {"value": 3})
    assert structured == {"first": 6.0, "second": 6.0}


def test_non_identifier_output_name_falls_back_to_a_mapping():
    tool = make_tool(output_name="not an identifier")
    schema = list_tools(create_mcp_server(ToolRegistry([tool])))[0].outputSchema

    assert schema["type"] == "object"
    _, structured = call(create_mcp_server(ToolRegistry([tool])), "echo_value", {"value": 3})
    assert structured == {"not an identifier": 6.0}


def test_host_and_port_configure_http_transport(registry):
    server = create_mcp_server(registry, host="0.0.0.0", port=9123)

    assert (server.settings.host, server.settings.port) == ("0.0.0.0", 9123)


def test_host_and_port_default_to_the_sdk_settings(registry):
    server = create_mcp_server(registry)

    assert (server.settings.host, server.settings.port) == ("127.0.0.1", 8000)


def test_missing_mcp_dependency_has_actionable_error(monkeypatch, registry):
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    with pytest.raises(ImportError, match=r"pip install predikit\[mcp\]"):
        create_mcp_server(registry)
