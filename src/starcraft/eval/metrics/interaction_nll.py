"""Inter-agent interaction NLL metric (distance-to-nearest).

Mixed-scene pairing — the semantically correct comparison for StarCraft
rollouts since each mode's rollout was generated with the other side
teacher-forced to GT:

* Metric-scope agents use their **predicted** positions `pred_traj[i, r, t]`.
* All other ever-alive scene agents use **GT** positions `gt_scene[k, t]`.

For each metric-scope agent i at rollout r, the signed distance to the
nearest other agent at timestep t is the min over:

* scope↔scope: `‖pred[i,r,t] − pred[j,r,t]‖ − r_i − r_j`   (j ≠ i)
* scope↔scene: `‖pred[i,r,t] − gt_scene[k,t]‖ − r_i − r_k`

Same reduction on GT (all positions GT) gives the comparison baseline for
the KDE log-prob.

A "collision" Bernoulli metric was considered and dropped — SC2 armies pack
units inside each other's collision footprints (radius ≈ unit extent, not a
physical exclusion zone), so both GT and predicted "ever touched" rates
saturate near 1 and the likelihood is uninformative. Proximity signal lives
in `distance_to_nearest_nll` instead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.starcraft.eval.feature_stats import default_bandwidth
from src.starcraft.eval.kinematics import pairwise_signed_distance
from src.starcraft.eval.load_rollout import ScenarioRollout
from src.starcraft.eval.log_kde import log_kde


_DIST_METRIC = "distance_to_nearest_nll"


def _empty(name: str) -> dict:
    return {"metric": name, "value": None, "n_agents": 0, "weight": 0}


def compute(scenario: ScenarioRollout, ctx: Optional[object] = None) -> list:
    # Derive R from the data, not the file attr — the save path opens in
    # append mode and only stamps file-level attrs on first creation, so
    # a rewrite with a different rollout count leaves the attr stale.
    R = int(scenario.pred_traj.shape[1])
    if R < 2:
        return [_empty(_DIST_METRIC)]
    # Bandwidth precedence: explicit ctx override > Silverman from committed σ.
    if ctx is not None and getattr(ctx, "bandwidths", None) and _DIST_METRIC in ctx.bandwidths:
        bandwidth = float(ctx.bandwidths[_DIST_METRIC])
    else:
        bandwidth = default_bandwidth(_DIST_METRIC, R)

    pred_traj = scenario.pred_traj.astype(np.float32)     # [N, R, T, 2]
    gt_traj = scenario.gt_traj.astype(np.float32)         # [N, T, 2]
    gt_valid = scenario.gt_valid.astype(bool)             # [N, T]
    gt_radius = scenario.gt_radius.astype(np.float32)     # [N]

    scene_traj = scenario.gt_scene_traj                    # [M, T, 2] or None
    scene_valid = scenario.gt_scene_valid                  # [M, T] or None
    scene_radius = scenario.gt_scene_radius                # [M] or None

    N, R, T, _ = pred_traj.shape
    has_scene = scene_traj is not None and scene_traj.shape[0] > 0
    M = scene_traj.shape[0] if has_scene else 0

    # Inter-agent distance is undefined with fewer than 2 total agents.
    if N + M < 2:
        return [_empty(_DIST_METRIC)]

    scene_traj_f = scene_traj.astype(np.float32) if has_scene else None
    scene_radius_f = scene_radius.astype(np.float32) if has_scene else None
    scene_valid_b = scene_valid.astype(bool) if has_scene else None

    # --------------------------------------------------------------
    # Predicted min-distance per (i, r, t).
    # --------------------------------------------------------------
    INF = np.float32(1e9)
    pred_min = np.full((N, R, T), INF, dtype=np.float32)

    pv_scope = gt_valid[:, None, :] & gt_valid[None, :, :]            # [N, N, T]
    eye = np.eye(N, dtype=bool)[:, :, None]                            # [N, N, 1]
    pv_scope = pv_scope & ~eye                                          # [N, N, T]

    if has_scene:
        pv_scene = gt_valid[:, None, :] & scene_valid_b[None, :, :]    # [N, M, T]
    else:
        pv_scene = None

    for r in range(R):
        pr = pred_traj[:, r, :, :]                                     # [N, T, 2]
        d_ss = pairwise_signed_distance(pr, gt_radius, pr, gt_radius)
        d_ss_masked = np.where(pv_scope, d_ss, INF)
        min_ss = d_ss_masked.min(axis=1)                                # [N, T]

        if has_scene:
            d_sc = pairwise_signed_distance(
                pr, gt_radius, scene_traj_f, scene_radius_f,
            )                                                           # [N, M, T]
            d_sc_masked = np.where(pv_scene, d_sc, INF)
            min_sc = d_sc_masked.min(axis=1)                            # [N, T]
            pred_min[:, r, :] = np.minimum(min_ss, min_sc)
        else:
            pred_min[:, r, :] = min_ss

    # GT min-distance per (i, t): everyone uses GT positions.
    gt_d_ss = pairwise_signed_distance(gt_traj, gt_radius, gt_traj, gt_radius)
    gt_d_ss_masked = np.where(pv_scope, gt_d_ss, INF)
    gt_min_ss = gt_d_ss_masked.min(axis=1)                              # [N, T]

    if has_scene:
        gt_d_sc = pairwise_signed_distance(
            gt_traj, gt_radius, scene_traj_f, scene_radius_f,
        )
        gt_d_sc_masked = np.where(pv_scene, gt_d_sc, INF)
        gt_min_sc = gt_d_sc_masked.min(axis=1)                          # [N, T]
        gt_min = np.minimum(gt_min_ss, gt_min_sc)
    else:
        gt_min = gt_min_ss

    has_other = pv_scope.any(axis=1)                                    # [N, T]
    if has_scene:
        has_other = has_other | pv_scene.any(axis=1)                    # [N, T]
    dist_valid = gt_valid & has_other                                   # [N, T]

    if not dist_valid.any():
        return [_empty(_DIST_METRIC)]

    pred_min_rt_last = np.moveaxis(pred_min, 1, -1)                     # [N, T, R]
    log_prob = log_kde(pred_min_rt_last, gt_min, bandwidth=bandwidth)   # [N, T]
    total_lp = float(np.sum(log_prob * dist_valid))
    n_valid = int(dist_valid.sum())
    return [{
        "metric": _DIST_METRIC,
        "value": -(total_lp / n_valid),
        "n_agents": int(dist_valid.any(axis=-1).sum()),
        "weight": n_valid,
    }]
