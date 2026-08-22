"""Unit tests for from_snowflake: fully mocked — no Snowflake connection needed."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import BaseModel, Field

from predikit.loaders.snowflake import _SnowflakeShim, from_snowflake

# ---------------------------------------------------------------------------
# Shared schema
# ---------------------------------------------------------------------------


class MemberInput(BaseModel):
    tenure_months: float = Field(description="Months as a member")
    trips_last_year: float = Field(description="Trips taken in past 12 months")
    avg_spend: float = Field(description="Average spend per trip in USD")


# ---------------------------------------------------------------------------
# _SnowflakeShim unit tests
# ---------------------------------------------------------------------------


class TestSnowflakeShim:
    def test_invalid_output_method_raises(self):
        with pytest.raises(AttributeError, match="no callable method"):
            _SnowflakeShim(MagicMock(spec=["predict"]), output_method="score")

    def test_predict_calls_correct_method(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        shim = _SnowflakeShim(mock_model)
        result = shim.predict(np.array([[24.0, 2.0, 500.0]]))
        mock_model.predict.assert_called_once()
        assert result[0] == 1

    def test_custom_output_method(self):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.27, 0.73]])
        shim = _SnowflakeShim(mock_model, output_method="predict_proba")
        result = shim.predict(np.array([[24.0, 2.0, 500.0]]))
        mock_model.predict_proba.assert_called_once()
        assert result[0] == pytest.approx(0.27, abs=1e-6)

    def test_classes_optional(self):
        shim_no_classes = _SnowflakeShim(MagicMock(spec=["predict"]))
        assert not hasattr(shim_no_classes, "classes_")

        shim_with_classes = _SnowflakeShim(MagicMock(spec=["predict"]), classes=[0, 1])
        assert shim_with_classes.classes_ == [0, 1]

    def test_predict_flattens_dataframe_result(self):
        import pandas as pd

        mock_model = MagicMock()
        mock_model.predict.return_value = pd.DataFrame({"output": [1]})
        shim = _SnowflakeShim(mock_model)
        result = shim.predict(np.array([[1.0, 2.0, 3.0]]))
        assert result.ndim == 1
        assert result[0] == 1

    def test_predict_flattens_numpy_result(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.73]])
        shim = _SnowflakeShim(mock_model)
        result = shim.predict(np.array([[1.0]]))
        assert result.ndim == 1
        assert result[0] == pytest.approx(0.73)

    def test_predict_proba_delegates_to_underlying_model(self):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        shim = _SnowflakeShim(mock_model)

        result = shim.predict_proba(np.array([[1.0, 2.0, 3.0]]))

        mock_model.predict_proba.assert_called_once()
        assert result.shape == (1, 2)
        assert result[0, 1] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# from_snowflake integration tests (mocked session)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_registry():
    """Yield a stand-in ``Registry`` class importable as snowflake.ml.registry.

    snowflake-ml-python is not a test dependency, so the module tree does not
    exist at all. from_snowflake() imports Registry lazily inside its own body,
    which makes sys.modules the seam rather than a module attribute.
    """
    registry_cls = MagicMock(name="Registry")
    registry_mod = ModuleType("snowflake.ml.registry")
    registry_mod.Registry = registry_cls
    ml_mod = ModuleType("snowflake.ml")
    ml_mod.registry = registry_mod
    root_mod = ModuleType("snowflake")
    root_mod.ml = ml_mod

    with patch.dict(
        sys.modules,
        {
            "snowflake": root_mod,
            "snowflake.ml": ml_mod,
            "snowflake.ml.registry": registry_mod,
        },
    ):
        yield registry_cls


def _mock_sf_model(return_value):
    """Return a mock registry model whose predict() yields return_value."""
    mock_sf_model = MagicMock()
    mock_sf_model.predict.return_value = return_value
    return mock_sf_model


class TestFromSnowflake:
    def test_roundtrip_invoke(self, mock_registry):
        sf_model = _mock_sf_model(np.array([1]))
        mock_registry.return_value.get_model.return_value.version.return_value = sf_model

        tool = from_snowflake(
            session=MagicMock(),
            model_name="VACATION_CHURN",
            model_version="V3",
            name="churn_risk",
            description="Predict member churn.",
            input_schema=MemberInput,
            output_name="churn_class",
            output_description="Predicted churn class",
        )

        result = tool.invoke({"tenure_months": 24.0, "trips_last_year": 2.0, "avg_spend": 500.0})
        assert result == {"churn_class": 1}

    def test_registry_called_with_correct_args(self, mock_registry):
        mock_session = MagicMock()
        sf_model = _mock_sf_model(np.array([0]))
        registry = mock_registry.return_value
        registry.get_model.return_value.version.return_value = sf_model

        from_snowflake(
            session=mock_session,
            model_name="VACATION_CHURN",
            model_version="V3",
            name="churn_risk",
            description="Predict churn.",
            input_schema=MemberInput,
            output_name="churn_class",
            output_description="Churn class",
        )

        mock_registry.assert_called_once_with(session=mock_session)
        registry.get_model.assert_called_once_with("VACATION_CHURN")
        registry.get_model.return_value.version.assert_called_once_with("V3")

    def test_custom_output_method_forwarded(self, mock_registry):
        sf_model = MagicMock()
        sf_model.score.return_value = np.array([0.87])
        mock_registry.return_value.get_model.return_value.version.return_value = sf_model

        tool = from_snowflake(
            session=MagicMock(),
            model_name="SCORE_MODEL",
            model_version="V1",
            name="scorer",
            description="Score members.",
            input_schema=MemberInput,
            output_name="score",
            output_description="Member score",
            output_method="score",
        )

        result = tool.invoke({"tenure_months": 12.0, "trips_last_year": 5.0, "avg_spend": 300.0})
        assert result["score"] == pytest.approx(0.87, abs=1e-6)
        sf_model.score.assert_called_once()

    def test_confidence_routing_uses_predict_proba(self, mock_registry):
        sf_model = MagicMock()
        sf_model.predict.return_value = np.array([1])
        sf_model.predict_proba.return_value = np.array([[0.6, 0.4]])
        sf_model.classes_ = [0, 1]
        mock_registry.return_value.get_model.return_value.version.return_value = sf_model

        tool = from_snowflake(
            session=MagicMock(),
            model_name="CHURN",
            model_version="V1",
            name="churn",
            description="Churn.",
            input_schema=MemberInput,
            output_name="churn_class",
            output_description="Churn class",
            confidence_threshold=0.9,
            on_low_confidence="warn",
        )

        result = tool.invoke({"tenure_months": 12.0, "trips_last_year": 5.0, "avg_spend": 300.0})
        assert result["churn_class"] == 1
        assert result["_low_confidence"] is True
        assert result["_confidence"] == pytest.approx(0.6)
        sf_model.predict_proba.assert_called_once()

    def test_model_tool_kwargs_forwarded(self, mock_registry):
        sf_model = _mock_sf_model(np.array([1]))
        mock_registry.return_value.get_model.return_value.version.return_value = sf_model

        tool = from_snowflake(
            session=MagicMock(),
            model_name="CHURN",
            model_version="V1",
            name="churn",
            description="Churn.",
            input_schema=MemberInput,
            output_name="churn_class",
            output_description="Churn class",
            classes=[0, 1],
            confidence_threshold=0.85,
            on_low_confidence="warn",
        )

        assert tool.confidence_threshold == 0.85
        assert tool.on_low_confidence == "warn"

    def test_import_error_without_snowflake_ml(self):
        # A None entry in sys.modules makes the import raise, whether or not
        # snowflake-ml-python happens to be installed in the environment.
        with (
            patch.dict(sys.modules, {"snowflake.ml.registry": None}),
            pytest.raises(ImportError, match="snowflake-ml-python is required"),
        ):
            from_snowflake(
                session=MagicMock(),
                model_name="X",
                model_version="V1",
                name="x",
                description="x",
                input_schema=MemberInput,
                output_name="y",
                output_description="y",
            )
