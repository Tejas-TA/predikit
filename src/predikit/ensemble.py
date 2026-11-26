from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from .exporters.langchain import to_langchain_tool
from .exporters.openai import to_openai_schema
from .tool import ModelTool

_VALID_STRATEGIES = {"collect", "mean", "weighted_mean", "vote", "weighted_vote"}
_WEIGHTED_STRATEGIES = {"weighted_mean", "weighted_vote"}


class ModelEnsemble:
    """Calls multiple ModelTools and reconciles their outputs into one dict."""

    def __init__(
        self,
        tools: list[ModelTool],
        name: str,
        description: str,
        strategy: str = "collect",
        weights: list[float] | None = None,
    ) -> None:
        if not tools:
            raise ValueError("ModelEnsemble requires at least one tool.")
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Use one of: {sorted(_VALID_STRATEGIES)}."
            )
        if weights is not None and len(weights) != len(tools):
            raise ValueError(
                f"weights length ({len(weights)}) must match number of tools ({len(tools)})."
            )
        if weights is not None:
            if strategy not in _WEIGHTED_STRATEGIES:
                raise ValueError(f"weights are only valid with {sorted(_WEIGHTED_STRATEGIES)}.")
            if any(w < 0 for w in weights):
                raise ValueError("weights must be non-negative.")
            if strategy in _WEIGHTED_STRATEGIES and sum(weights) <= 0:
                raise ValueError("weights must sum to a positive value.")
        output_names = [tool.output_name for tool in tools]
        input_fields = set(tools[0].input_schema.model_fields)
        mismatched_inputs = [
            tool.name for tool in tools[1:] if set(tool.input_schema.model_fields) != input_fields
        ]
        if mismatched_inputs:
            raise ValueError(
                "All ensemble tools must use the same input fields; "
                f"'{tools[0].name}' differs from {mismatched_inputs}."
            )
        if strategy == "collect":
            duplicates = sorted({name for name in output_names if output_names.count(name) > 1})
            if duplicates:
                raise ValueError(
                    "collect strategy requires unique output_name values; "
                    f"duplicates: {duplicates}."
                )
        elif len(set(output_names)) != 1:
            raise ValueError(
                f"{strategy} strategy requires all tools to share the same output_name; "
                f"got {output_names}."
            )
        self.tools = tools
        self.name = name
        self.description = description
        self.strategy = strategy
        self.weights = weights
        self.input_schema = tools[0].input_schema
        self.output_description = tools[0].output_description

    @property
    def output_name(self) -> str:
        return self.tools[0].output_name

    def invoke(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Call all tools and reconcile outputs using the configured strategy."""
        return self._reconcile([t.invoke(input_dict) for t in self.tools])

    def invoke_many(self, input_dicts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run each member over a batch and reconcile each input's results."""
        member_results = [tool.invoke_many(input_dicts) for tool in self.tools]
        return [self._reconcile(list(results)) for results in zip(*member_results, strict=True)]

    async def ainvoke(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Call all tools concurrently and reconcile outputs using the configured strategy."""
        results = await asyncio.gather(*[t.ainvoke(input_dict) for t in self.tools])
        return self._reconcile(list(results))

    async def ainvoke_many(self, input_dicts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Async batch inference, running ensemble members concurrently."""
        member_results = await asyncio.gather(
            *[tool.ainvoke_many(input_dicts) for tool in self.tools]
        )
        return [self._reconcile(list(results)) for results in zip(*member_results, strict=True)]

    def _reconcile(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        if self.strategy == "collect":
            merged: dict[str, Any] = {}
            for r in results:
                merged.update(r)
            return merged

        output_name = self.tools[0].output_name
        values = [r[output_name] for r in results]
        weights = self.weights if self.weights is not None else [1.0] * len(self.tools)

        if self.strategy == "mean":
            # Equal-weight average regardless of self.weights; use weighted_mean for non-uniform weights.
            numeric = [float(v) for v in values]
            return {output_name: sum(numeric) / len(numeric)}

        if self.strategy == "weighted_mean":
            numeric = [float(v) for v in values]
            total = sum(weights)
            return {output_name: sum(w * v for w, v in zip(weights, numeric, strict=True)) / total}

        if self.strategy == "vote":
            return {output_name: Counter(values).most_common(1)[0][0]}

        if self.strategy == "weighted_vote":
            tally: dict[Any, float] = {}
            for w, v in zip(weights, values, strict=True):
                tally[v] = tally.get(v, 0.0) + w
            return {output_name: max(tally, key=tally.__getitem__)}

        # unreachable — __init__ validates strategy — but satisfies type checkers
        raise ValueError(f"Unknown strategy '{self.strategy}'.")

    def to_openai(self) -> dict[str, Any]:
        return to_openai_schema(self)

    def to_langchain(self) -> Any:
        return to_langchain_tool(self)

    def to_callable(self) -> Callable[..., dict[str, Any]]:
        def _fn(**kwargs: Any) -> dict[str, Any]:
            return self.invoke(kwargs)

        _fn.__name__ = self.name
        _fn.__doc__ = self.description
        return _fn
