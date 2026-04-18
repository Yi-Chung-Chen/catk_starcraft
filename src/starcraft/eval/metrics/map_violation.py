"""Map violation metric — SC2 analogue of Waymo's road-departure metric.

Per rollout, check whether any GT-valid, non-flying timestep places the
unit's center on a blocked pathing-grid cell. Compare the per-agent rollout
violation rate against the GT violation indicator via a Bernoulli NLL.

Diagnostic `map_violation_rate` reports the mean predicted rate across
agents, for quick eyeballing without reading the NLL directly.

Ground-only gating (using GT `is_flying`) is a deliberate simplification:
rollouts don't predict flying state, and over an 8-second window the
flying-state distribution is stable enough that GT is a reasonable proxy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.starcraft.eval.load_rollout import ScenarioRollout
from src.starcraft.eval.log_kde import bernoulli_nll
from src.starcraft.eval.map_grid import load_pathing_grid


def _empty(name: str) -> dict:
    return {"metric": name, "value": None, "n_agents": 0, "weight": 0}


def _check_blocked(
    xy: np.ndarray,                 # [..., 2]
    pathing: np.ndarray,            # [H, W] bool (True = blocked)
) -> np.ndarray:
    """Per-point blocked flag using center-cell lookup."""
    H, W = pathing.shape
    x = xy[..., 0]
    y = xy[..., 1]
    col = np.clip(x.astype(np.int64), 0, W - 1)
    row = np.clip((H - y).astype(np.int64), 0, H - 1)
    return pathing[row, col]


def compute(scenario: ScenarioRollout, ctx: Optional[object] = None) -> list:
    map_dir = None
    if ctx is not None:
        map_dir = getattr(ctx, "map_dir", None)
    if map_dir is None:
        # No pathing grid available — skip cleanly so the metric still
        # appears in the CSV with weight=0.
        return [
            _empty("map_violation_nll"),
            _empty("map_violation_rate"),
        ]

    pathing = load_pathing_grid(str(map_dir), scenario.map_name)

    pred_traj = scenario.pred_traj.astype(np.float32)     # [N, R, T, 2]
    gt_traj = scenario.gt_traj.astype(np.float32)         # [N, T, 2]
    gt_valid = scenario.gt_valid.astype(bool)             # [N, T]
    gt_is_flying = scenario.gt_is_flying
    if gt_is_flying is None:
        # Older replays without is_flying — treat as "none flying" so the
        # metric still runs (errs toward over-counting air units).
        gt_is_flying = np.zeros_like(gt_valid)
    else:
        gt_is_flying = gt_is_flying.astype(bool)

    # Scope: GT-valid AND on the ground, per (agent, timestep).
    map_valid = gt_valid & ~gt_is_flying                   # [N, T]
    if not map_valid.any():
        return [
            _empty("map_violation_nll"),
            _empty("map_violation_rate"),
        ]

    N, R, T, _ = pred_traj.shape

    # Predicted violation per (agent, rollout): ever blocked on a valid step.
    # Broadcasting map_valid[:, None, :] over rollouts keeps each (i, r) gated
    # by agent i's ground-valid timesteps.
    pred_blocked = _check_blocked(pred_traj, pathing)                 # [N, R, T]
    pred_violation = (pred_blocked & map_valid[:, None, :]).any(axis=-1)  # [N, R]

    # GT violation indicator per agent.
    gt_blocked = _check_blocked(gt_traj, pathing)                     # [N, T]
    gt_violation = (gt_blocked & map_valid).any(axis=-1)              # [N] bool

    # Agents eligible for evaluation: at least one valid ground step.
    eligible = map_valid.any(axis=-1)                                 # [N]
    n_eval = int(eligible.sum())
    if n_eval == 0:
        return [
            _empty("map_violation_nll"),
            _empty("map_violation_rate"),
        ]

    p_hat = pred_violation.mean(axis=1).astype(np.float64)            # [N]
    nlls = bernoulli_nll(p_hat, gt_violation.astype(np.float64))      # [N]

    nll_val = float(nlls[eligible].mean())
    rate_val = float(p_hat[eligible].mean())

    return [
        {
            "metric": "map_violation_nll",
            "value": nll_val,
            "n_agents": n_eval,
            "weight": n_eval,
        },
        {
            "metric": "map_violation_rate",
            "value": rate_val,
            "n_agents": n_eval,
            "weight": n_eval,
        },
    ]
