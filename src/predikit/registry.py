from __future__ import annotations

from collections.abc import Iterator

from .ensemble import ModelEnsemble
from .tool import ModelTool

RegistryItem = ModelTool | ModelEnsemble


class ToolRegistry:
    """Bundles multiple ModelTools (and optional ModelEnsembles) for bulk export."""

    def __init__(
        self,
        tools: list[ModelTool],
        ensembles: list[ModelEnsemble] | None = None,
    ) -> None:
        names = [t.name for t in tools] + [e.name for e in (ensembles or [])]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate tool or ensemble names are not allowed: {duplicates}.")
        self._tools: dict[str, ModelTool] = {t.name: t for t in tools}
        self._ensembles: dict[str, ModelEnsemble] = {e.name: e for e in (ensembles or [])}

    def get(self, name: str) -> RegistryItem:
        if name in self._tools:
            return self._tools[name]
        if name in self._ensembles:
            return self._ensembles[name]
        available = self.names()
        raise KeyError(f"No tool or ensemble named '{name}'. Available: {available}")

    def items(self) -> list[RegistryItem]:
        """Return every tool followed by every ensemble, in registration order."""
        return list(self._tools.values()) + list(self._ensembles.values())

    def names(self) -> list[str]:
        """Return the name of every registered tool and ensemble."""
        return [item.name for item in self.items()]

    def __iter__(self) -> Iterator[RegistryItem]:
        return iter(self.items())

    def __len__(self) -> int:
        return len(self._tools) + len(self._ensembles)

    def to_openai(self) -> list[dict]:
        return [item.to_openai() for item in self.items()]

    def to_langchain(self) -> list:
        return [item.to_langchain() for item in self.items()]
