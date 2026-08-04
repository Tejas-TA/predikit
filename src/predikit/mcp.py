"""MCP server integration for predikit tools."""

from __future__ import annotations

import inspect
from typing import Any, cast

from .registry import ToolRegistry


def create_mcp_server(registry: ToolRegistry, name: str = "predikit") -> Any:
    """Create an MCP server exposing every tool in ``registry``.

    The MCP dependency is optional. Install it with ``pip install predikit[mcp]``.
    The returned server uses the SDK's standard transports, including ``stdio``.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as err:
        raise ImportError(
            "The MCP integration requires the optional dependency. "
            "Install it with: pip install predikit[mcp]"
        ) from err

    server = FastMCP(name, json_response=True)
    for tool in _registry_items(registry):
        server.add_tool(
            _make_mcp_callable(tool),
            name=tool.name,
            description=tool.description,
        )
    return server


def _registry_items(registry: ToolRegistry) -> list[Any]:
    """Return registry members in the same order as the existing exporters."""
    return list(registry._tools.values()) + list(registry._ensembles.values())


def _make_mcp_callable(tool: Any) -> Any:
    """Build a callable whose signature mirrors a tool's Pydantic input model."""

    async def invoke(**kwargs: Any) -> dict:
        return cast(dict, await tool.ainvoke(kwargs))

    parameters = []
    annotations: dict[str, Any] = {}
    for name, field in tool.input_schema.model_fields.items():
        annotation = field.annotation if field.annotation is not None else Any
        annotations[name] = annotation
        default = (
            inspect.Parameter.empty
            if field.is_required()
            else field.get_default(call_default_factory=True)
        )
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=annotation,
                default=default,
            )
        )
    invoke.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    invoke.__annotations__ = annotations
    invoke.__name__ = tool.name
    invoke.__doc__ = tool.description
    return invoke
