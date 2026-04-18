"""Metric registry. Each entry is a callable mapping a `ScenarioRollout` and
a `MetricCtx` to a list of result dicts (one per metric variant emitted by
the function).

A metric function receives the loaded `ScenarioRollout` and a `MetricCtx`
(map_dir, bandwidths). It returns
`list[{"metric": str, "value": float | None, "n_agents": int, "weight": int}]`.
The driver augments each record with `scenario_id`, `map_name`, `observer`,
`mode`. `weight` is optional (falls back to `n_agents` in aggregation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

from src.starcraft.eval.load_rollout import ScenarioRollout

from . import (
    horizon_ade,
    interaction_nll,
    kinematic_nll,
    map_violation,
    min_ade,
)


@dataclass(frozen=True)
class MetricCtx:
    """Shared context threaded to every metric function.

    `bandwidths` maps metric-feature name → KDE bandwidth (Gaussian std).
    Missing entries fall back to per-metric defaults.
    """
    map_dir: Optional[Path] = None
    bandwidths: Dict[str, float] = field(default_factory=dict)


MetricFn = Callable[[ScenarioRollout, MetricCtx], list]

# The NLL/Bernoulli metrics mix different task semantics (own vs opponent
# modeling) in their "overall" pool — reporting a single mean across modes is
# misleading for them, so the driver suppresses the overall print for these
# metric names while still populating `summary["overall"]` for CSV callers.
MODE_SENSITIVE_METRICS: frozenset = frozenset({
    "linear_speed_nll",
    "linear_accel_nll",
    "angular_speed_nll",
    "angular_accel_nll",
    "distance_to_nearest_nll",
    "collision_nll",
    "collision_rate",
    "map_violation_nll",
    "map_violation_rate",
})

REGISTRY: Dict[str, MetricFn] = {
    "min_ade": min_ade.compute,
    "horizon_ade": horizon_ade.compute,
    "kinematic_nll": kinematic_nll.compute,
    "interaction_nll": interaction_nll.compute,
    "map_violation": map_violation.compute,
}


def get(name: str) -> MetricFn:
    if name not in REGISTRY:
        avail = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(f"Unknown metric {name!r}. Available: {avail}")
    return REGISTRY[name]
