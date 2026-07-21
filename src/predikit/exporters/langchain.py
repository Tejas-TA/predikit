from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..ensemble import ModelEnsemble
    from ..tool import ModelTool


def to_langchain_tool(tool: ModelTool | ModelEnsemble) -> Any:
    """Convert a ModelTool to a LangChain StructuredTool."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as err:
        raise ImportError(
            "langchain-core is required. Install with: pip install predikit[langchain]"
        ) from err

    def _run(**kwargs) -> dict:
        return tool.invoke(kwargs)

    async def _arun(**kwargs) -> dict:
        return await tool.ainvoke(kwargs)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name=tool.name,
        description=tool.description,
        args_schema=tool.input_schema,
    )
