"""MCP server integration for predikit tools."""

from __future__ import annotations

import inspect
from typing import Any, cast

from pydantic import ConfigDict, Field, create_model

from .registry import ToolRegistry


def create_mcp_server(
    registry: ToolRegistry,
    name: str = "predikit",
    host: str | None = None,
    port: int | None = None,
) -> Any:
    """Create an MCP server exposing every tool in ``registry``.

    The MCP dependency is optional. Install it with ``pip install predikit[mcp]``.
    The returned server uses the SDK's standard transports, including ``stdio``.

    ``host`` and ``port`` configure the HTTP transports; leave them unset to keep
    the MCP SDK defaults (``127.0.0.1:8000``).
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as err:
        raise ImportError(
            "The MCP integration requires the optional dependency. "
            "Install it with: pip install predikit[mcp]"
        ) from err

    settings: dict[str, Any] = {"json_response": True}
    if host is not None:
        settings["host"] = host
    if port is not None:
        settings["port"] = port

    server = FastMCP(name, **settings)
    for tool in registry.items():
        server.add_tool(
            _make_mcp_callable(tool),
            name=tool.name,
            description=tool.description,
        )
    return server


def _output_fields(tool: Any) -> list[tuple[str, str]]:
    """Return the (name, description) pairs a tool's result dict is keyed by."""
    tools = getattr(tool, "tools", None)
    if tools is not None and getattr(tool, "strategy", None) == "collect":
        # A collect ensemble merges every member's output into one dict.
        return [(t.output_name, t.output_description) for t in tools]
    return [(tool.output_name, tool.output_description)]


def _make_output_model(tool: Any) -> Any:
    """Build the return annotation used to advertise a tool's output schema.

    ``extra="allow"`` matters: it keeps the ``_confidence`` / ``_low_confidence``
    keys that a low-confidence result carries. A model without it validates them
    away, so MCP clients would silently lose them.
    """
    fields = _output_fields(tool)
    if not all(name.isidentifier() for name, _ in fields):
        # Output names are user-supplied; fall back to a permissive mapping.
        return dict[str, Any]
    definitions: dict[str, Any] = {
        name: (Any, Field(..., description=desc)) for name, desc in fields
    }
    return create_model(
        f"{tool.name}_output",
        __config__=ConfigDict(extra="allow"),
        **definitions,
    )


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
    # FastMCP derives outputSchema from the return annotation. A bare ``dict`` is
    # not serializable for structured output, so it must be a concrete model.
    output_model = _make_output_model(tool)
    annotations["return"] = output_model
    invoke.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters, return_annotation=output_model
    )
    invoke.__annotations__ = annotations
    invoke.__name__ = tool.name
    invoke.__doc__ = tool.description
    return invoke
