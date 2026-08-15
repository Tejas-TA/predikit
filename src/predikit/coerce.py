from typing import Any

from pydantic import BaseModel

_BOOL_TRUE = {"true", "1", "yes", "on"}
_BOOL_FALSE = {"false", "0", "no", "off"}


def coerce_value(value: Any, target_type: type) -> Any:
    """Coerce a single value to target_type with LLM-friendly string handling."""
    if target_type is int and isinstance(value, bool):
        # isinstance(True, int) is True, so bools would otherwise pass through unconverted.
        return int(value)

    if isinstance(value, target_type):
        return value

    if target_type is bool:
        if isinstance(value, str):
            low = value.lower()
            if low in _BOOL_TRUE:
                return True
            if low in _BOOL_FALSE:
                return False
            raise ValueError(
                f"Cannot interpret {value!r} as bool. Expected one of: true/false, yes/no, 1/0, on/off"
            )
        return bool(value)

    if target_type is int:
        return int(float(value))

    if target_type is float:
        return float(value)

    if target_type is str:
        return str(value)

    return value


def coerce_inputs(validated: BaseModel, meta: dict) -> list:
    """Return feature values in the order the model expects them."""
    data = validated.model_dump()
    feature_names = meta.get("feature_names")
    if feature_names:
        missing = [f for f in feature_names if f not in data]
        if missing:
            raise ValueError(
                f"Input schema is missing model features: {missing}. "
                f"Field names in your Pydantic schema must exactly match the column names "
                f"used during model training. "
                f"Schema has: {list(data.keys())}, model expects: {feature_names}"
            )
        return [data[f] for f in feature_names]
    return list(data.values())
