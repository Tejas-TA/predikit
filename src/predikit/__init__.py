"""Wrap fitted scikit-learn / XGBoost models as LLM-callable tools."""

from .ensemble import ModelEnsemble
from .exceptions import LowConfidenceError
from .mcp import create_mcp_server
from .registry import ToolRegistry
from .tool import ModelTool

__all__ = [
    "LowConfidenceError",
    "ModelEnsemble",
    "ModelTool",
    "ToolRegistry",
    "create_mcp_server",
]
__version__ = "0.6.2"
