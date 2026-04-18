"""Kinematic NLL metrics (Waymo Sim-Agents style, adapted for SC2).

For each per-timestep kinematic feature (linear speed, linear acceleration,
angular speed, angular acceleration), compute the Gaussian-KDE log-prob of
the GT feature under the empirical distribution of the R rollout samples,
then report `mean_nll = -mean(log_prob)` over valid (agent, timestep) pairs.

Weight per emitted record is the number of valid feature values contributing
to the mean — this is what `aggregate.summarize()` uses for the weighted
mean across scenarios, preserving unbiasedness when scenarios have
differing valid lifetimes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.starcraft.eval.kinematics import (
    compute_angular_accel,
    compute_angular_speed,
    compute_linear_accel,
    compute_speed,
)
from src.starcraft.eval.load_rollout import ScenarioRollout
from src.starcraft.eval.log_kde import log_kde


_DEFAULT_BANDWIDTHS = {
    "linear_speed_nll": 0.5,
    "linear_accel_nll": 1.0,
    "angular_speed_nll": 0.05,
    "angular_accel_nll": 0.1,
}


def _empty(name: str) -> dict:
    return {"metric": name, "value": None, "n_agents": 0, "weight": 0}


def _reduce(
    name: str,
    pred_feat: np.ndarray,       # [N, R, T_f]
    gt_feat: np.ndarray,          # [N, T_f]
    feat_valid: np.ndarray,       # [N, T_f] bool
    bandwidth: float,
) -> dict:
    n_valid = int(feat_valid.sum())
    if n_valid == 0:
        return _empty(name)

    # log_kde expects rollout on the last axis when called with default
    # rollout_axis=-1. Reshape [N, R, T_f] -> [N, T_f, R].
    pred_rt_last = np.moveaxis(pred_feat, 1, -1)  # [N, T_f, R]
    log_prob = log_kde(pred_rt_last, gt_feat, bandwidth=bandwidth)  # [N, T_f]

    total_log_prob = float(np.sum(log_prob * feat_valid))
    mean_nll = -(total_log_prob / n_valid)
    n_agents = int(feat_valid.any(axis=-1).sum())
    return {
        "metric": name,
        "value": mean_nll,
        "n_agents": n_agents,
        "weight": n_valid,
    }


def compute(scenario: ScenarioRollout, ctx: Optional[object] = None) -> list:
    bandwidths = _DEFAULT_BANDWIDTHS.copy()
    if ctx is not None and getattr(ctx, "bandwidths", None):
        bandwidths.update(ctx.bandwidths)

    dt = 1.0 / float(scenario.native_fps)
    pred_traj = scenario.pred_traj.astype(np.float32)    # [N, R, T, 2]
    pred_head = scenario.pred_head.astype(np.float32)    # [N, R, T]
    gt_traj = scenario.gt_traj.astype(np.float32)        # [N, T, 2]
    gt_head = scenario.gt_head.astype(np.float32)        # [N, T]
    valid = scenario.gt_valid.astype(bool)               # [N, T]

    # Kinematic features — shapes have one fewer T dim per derivative.
    pred_speed = compute_speed(pred_traj, dt)            # [N, R, T-1]
    gt_speed = compute_speed(gt_traj, dt)                # [N, T-1]
    speed_valid = valid[:, 1:] & valid[:, :-1]           # [N, T-1]

    pred_ang_speed = compute_angular_speed(pred_head, dt)  # [N, R, T-1]
    gt_ang_speed = compute_angular_speed(gt_head, dt)      # [N, T-1]
    ang_speed_valid = speed_valid                           # same shape/logic

    pred_accel = compute_linear_accel(pred_speed, dt)    # [N, R, T-2]
    gt_accel = compute_linear_accel(gt_speed, dt)        # [N, T-2]
    accel_valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]  # [N, T-2]

    pred_ang_accel = compute_angular_accel(pred_ang_speed, dt)   # [N, R, T-2]
    gt_ang_accel = compute_angular_accel(gt_ang_speed, dt)       # [N, T-2]
    ang_accel_valid = accel_valid

    out = [
        _reduce("linear_speed_nll", pred_speed, gt_speed, speed_valid,
                bandwidths["linear_speed_nll"]),
        _reduce("linear_accel_nll", pred_accel, gt_accel, accel_valid,
                bandwidths["linear_accel_nll"]),
        _reduce("angular_speed_nll", pred_ang_speed, gt_ang_speed,
                ang_speed_valid, bandwidths["angular_speed_nll"]),
        _reduce("angular_accel_nll", pred_ang_accel, gt_ang_accel,
                ang_accel_valid, bandwidths["angular_accel_nll"]),
    ]
    return out
