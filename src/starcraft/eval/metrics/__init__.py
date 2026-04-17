"""Metric registry. Each entry is a callable mapping a `ScenarioRollout` to a
list of result dicts (one per metric variant emitted by the function).

A metric function receives the loaded `ScenarioRollout` and returns
`list[{"metric": str, "value": float | None, "n_agents": int}]`. The driver
augments each record with `scenario_id`, `map_name`, `observer`, `mode`.
"""

from __future__ import annotations

from typing import Callable, Dict

from src.starcraft.eval.load_rollout import ScenarioRollout

from . import horizon_ade, min_ade

MetricFn = Callable[[ScenarioRollout], list]

REGISTRY: Dict[str, MetricFn] = {
    "min_ade": min_ade.compute,
    "horizon_ade": horizon_ade.compute,
}


def get(name: str) -> MetricFn:
    if name not in REGISTRY:
        avail = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(f"Unknown metric {name!r}. Available: {avail}")
    return REGISTRY[name]
