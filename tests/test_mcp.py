import asyncio
import sys
import types

import pytest
from pydantic import create_model

from predikit import ModelTool, ToolRegistry, create_mcp_server


class EchoModel:
    def predict(self, X):
        return [float(X[0][0])]


@pytest.fixture
def registry():
    schema = create_model("Input", value=(float, ...))
    tool = ModelTool(
        model=EchoModel(),
        name="echo_value",
        description="Echo a numeric value.",
        input_schema=schema,
        output_name="value",
        output_description="The value.",
    )
    return ToolRegistry([tool])


def test_create_mcp_server_registers_registry_tools(monkeypatch, registry):
    registered = []

    class FakeFastMCP:
        def __init__(self, name, json_response):
            self.name = name
            self.json_response = json_response

        def add_tool(self, function, name, description):
            registered.append((function, name, description))

    mcp_module = types.ModuleType("mcp")
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", types.ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)

    server = create_mcp_server(registry, name="Test Server")

    assert server.name == "Test Server"
    assert server.json_response is True
    function, name, description = registered[0]
    assert (name, description) == ("echo_value", "Echo a numeric value.")
    assert function.__signature__.parameters["value"].annotation is float


def test_mcp_callable_invokes_tool(monkeypatch, registry):
    registered = []

    class FakeFastMCP:
        def __init__(self, name, json_response):
            pass

        def add_tool(self, function, **kwargs):
            registered.append(function)

    mcp_module = types.ModuleType("mcp")
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", types.ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)

    create_mcp_server(registry)
    result = asyncio.run(registered[0](value=3))
    assert result == {"value": 3.0}


def test_missing_mcp_dependency_has_actionable_error(monkeypatch, registry):
    monkeypatch.setitem(sys.modules, "mcp", None)
    with pytest.raises(ImportError, match=r"pip install predikit\[mcp\]"):
        create_mcp_server(registry)
