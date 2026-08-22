from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel

from ..tool import ModelTool


class _SnowflakeShim:
    """Wrap a Snowflake Model Registry model as an sklearn estimator."""

    def __init__(
        self, sf_model: Any, output_method: str = "predict", classes: list | None = None
    ) -> None:
        if not callable(getattr(sf_model, output_method, None)):
            raise AttributeError(
                f"Snowflake model object has no callable method {output_method!r}."
            )
        self._model = sf_model
        self._method = output_method
        underlying = getattr(sf_model, "_model_impl", None)
        if underlying is not None:
            underlying = getattr(underlying, "sklearn_model", None)

        if classes is None and underlying is not None and hasattr(underlying, "classes_"):
            classes = list(underlying.classes_)

        if classes is not None:
            self.classes_ = classes

        if underlying is not None:
            if hasattr(underlying, "feature_names_in_"):
                self.feature_names_in_ = underlying.feature_names_in_
            if hasattr(underlying, "n_features_in_"):
                self.n_features_in_ = underlying.n_features_in_

    def predict(self, X: Any) -> np.ndarray:
        result = getattr(self._model, self._method)(X)
        if hasattr(result, "to_numpy"):
            return np.asarray(result.to_numpy()).flatten()
        return np.asarray(result).flatten()

    def predict_proba(self, X: Any) -> np.ndarray:
        if callable(getattr(self._model, "predict_proba", None)):
            return np.asarray(self._model.predict_proba(X))
        raise NotImplementedError(
            "predict_proba is not available for this Snowflake model object. "
            "The underlying model may not expose it."
        )


def from_snowflake(
    session: Any,
    model_name: str,
    model_version: str,
    name: str,
    description: str,
    input_schema: type[BaseModel],
    output_name: str,
    output_description: str,
    output_method: str = "predict",
    classes: list | None = None,
    **model_tool_kwargs,
) -> ModelTool:
    """Load a registered Snowflake model and return it as a ModelTool.

    Args:
        session: An active ``snowflake.snowpark.Session``.
        model_name: Name of the model in the Snowflake Model Registry.
        model_version: Version string, e.g. ``"V3"``.
        name: Tool name the LLM sees.
        description: Tool description the LLM sees.
        input_schema: Pydantic BaseModel describing the model's inputs.
        output_name: Key for the prediction in the returned dict.
        output_description: Human-readable description of the output.
        output_method: Method to call on the registry model object (default ``"predict"``).
        classes: Optional list of class labels; enables confidence routing for classifiers.
        **model_tool_kwargs: Forwarded to ModelTool (e.g. ``confidence_threshold``).

    Returns:
        A fully configured :class:`~predikit.ModelTool`.
    """
    try:
        from snowflake.ml.registry import Registry
    except ImportError as exc:
        raise ImportError(
            "snowflake-ml-python is required for from_snowflake(). "
            "Install it with: pip install predikit[snowflake]"
        ) from exc

    registry = Registry(session=session)
    sf_model = registry.get_model(model_name).version(model_version)
    shim = _SnowflakeShim(sf_model, output_method=output_method, classes=classes)
    return ModelTool(
        model=shim,
        name=name,
        description=description,
        input_schema=input_schema,
        output_name=output_name,
        output_description=output_description,
        **model_tool_kwargs,
    )
