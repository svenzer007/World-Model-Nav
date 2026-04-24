"""Weighted sequence sampling for V4 structured dynamics training."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils.data import WeightedRandomSampler

from .sequence_dataset import StructuredSequenceDataset


def compute_sequence_weights(
    dataset: StructuredSequenceDataset,
    config: Mapping[str, object] | None = None,
) -> torch.Tensor:
    """Compute normalized positive weights for all sequence windows without dropping data."""
    cfg = dict(config or {})
    base_weight = float(cfg.get("base_weight", 1.0))
    event_bonus = float(cfg.get("event_bonus", 2.0))
    risk_bonus = float(cfg.get("risk_bonus", 2.0))
    low_clearance_bonus = float(cfg.get("low_clearance_bonus", 1.5))
    collision_bonus = float(cfg.get("collision_bonus", 3.0))
    stop_downweight = float(cfg.get("stop_downweight", 0.7))
    clearance_scale = max(1e-6, float(cfg.get("clearance_scale", 1.0)))
    min_weight = float(cfg.get("min_weight", 0.05))
    max_weight = float(cfg.get("max_weight", 20.0))

    weights = []
    for window in dataset.windows:
        weight = base_weight
        weight += event_bonus * float(window.dynamic_event_fraction)
        weight += risk_bonus * float(window.max_risk)
        if window.min_clearance < clearance_scale:
            weight += low_clearance_bonus * (1.0 - max(0.0, float(window.min_clearance)) / clearance_scale)
        if (
            window.has_dynamic_collision_next
            or window.has_dynamic_collision_within_3
            or window.has_dynamic_collision_within_5
        ):
            weight += collision_bonus
        weight *= max(min_weight, 1.0 - stop_downweight * float(window.stop_fraction))
        weights.append(max(min_weight, min(max_weight, weight)))

    if not weights:
        raise ValueError("Cannot build a sampler for an empty dataset")
    tensor = torch.tensor(weights, dtype=torch.double)
    total = torch.sum(tensor)
    if not torch.isfinite(total) or float(total) <= 0.0:
        tensor = torch.ones_like(tensor, dtype=torch.double)
        total = torch.sum(tensor)
    tensor = tensor / total
    return tensor


def build_weighted_sampler(
    dataset: StructuredSequenceDataset,
    config: Mapping[str, object] | None = None,
) -> WeightedRandomSampler:
    """Return a replacement sampler using normalized sequence weights."""
    weights = compute_sequence_weights(dataset, config)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
