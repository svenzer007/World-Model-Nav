"""Structured world-model constructors for baseline V4 and Version C."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

from .structured_dynamics_model import StructuredDynamicsConfig, StructuredDynamicsModel
from .structured_obstacle_rssm import StructuredObstacleRSSMConfig, StructuredObstacleRSSMModel

BASELINE_MODEL_TYPE = "structured_dynamics_v4"
VERSION_C_MODEL_TYPE = "structured_obstacle_rssm_vc"


def resolve_model_type(model_config: Mapping[str, Any] | None) -> str:
    if model_config is None:
        return BASELINE_MODEL_TYPE
    model_type = model_config.get("model_type")
    return str(model_type) if model_type else BASELINE_MODEL_TYPE


def _config_kwargs(config_cls: type[Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    valid_names = {field.name for field in fields(config_cls)}
    return {key: value for key, value in model_config.items() if key in valid_names}


def build_model(model_config: Mapping[str, Any]) -> StructuredDynamicsModel | StructuredObstacleRSSMModel:
    model_type = resolve_model_type(model_config)
    if model_type == BASELINE_MODEL_TYPE:
        return StructuredDynamicsModel(StructuredDynamicsConfig(**_config_kwargs(StructuredDynamicsConfig, model_config)))
    if model_type == VERSION_C_MODEL_TYPE:
        return StructuredObstacleRSSMModel(
            StructuredObstacleRSSMConfig(**_config_kwargs(StructuredObstacleRSSMConfig, model_config))
        )
    raise ValueError(f"Unknown model_type: {model_type}")


__all__ = [
    "BASELINE_MODEL_TYPE",
    "VERSION_C_MODEL_TYPE",
    "StructuredDynamicsConfig",
    "StructuredDynamicsModel",
    "StructuredObstacleRSSMConfig",
    "StructuredObstacleRSSMModel",
    "build_model",
    "resolve_model_type",
]
