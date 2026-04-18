"""Inter-agent interaction NLL metrics (distance-to-nearest, collision).

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
the KDE log-prob. Collision NLL is a per-(agent, rollout) Bernoulli
indicator of "ever touched another agent", compared against GT.

Records returned:
    distance_to_nearest_nll    (weight = valid feature count)
    collision_nll              (Bernoulli NLL, weight = n_agents evaluated)
    collision_rate             (mean rollout collision rate, diagnostic)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.starcraft.eval.kinematics import pairwise_signed_distance
from src.starcraft.eval.load_rollout import ScenarioRollout
from src.starcraft.eval.log_kde import bernoulli_nll, log_kde


_DEFAULT_BANDWIDTH = 1.0   # override via ctx.bandwidths["distance_to_nearest_nll"]


def _empty(name: str) -> dict:
    return {"metric": name, "value": None, "n_agents": 0, "weight": 0}


def compute(scenario: ScenarioRollout, ctx: Optional[object] = None) -> list:
    bandwidth = _DEFAULT_BANDWIDTH
    if ctx is not None and getattr(ctx, "bandwidths", None):
        bandwidth = float(ctx.bandwidths.get("distance_to_nearest_nll", bandwidth))

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

    # If we have fewer than 2 agents total in the scene, inter-agent metrics
    # are undefined — emit None records with weight=0 so the summarizer skips.
    if N + M < 2:
        return [
            _empty("distance_to_nearest_nll"),
            _empty("collision_nll"),
            _empty("collision_rate"),
        ]

    scene_traj_f = scene_traj.astype(np.float32) if has_scene else None
    scene_radius_f = scene_radius.astype(np.float32) if has_scene else None
    scene_valid_b = scene_valid.astype(bool) if has_scene else None

    # --------------------------------------------------------------
    # Predicted min-distance per (i, r, t).
    # For each rollout r we independently compute the nearest-neighbor
    # signed distance from every scope agent i to every other agent.
    # --------------------------------------------------------------
    INF = np.float32(1e9)

    # Per-rollout min pred distances: [N, R, T]
    pred_min = np.full((N, R, T), INF, dtype=np.float32)
    # Per-rollout collision flag per agent: [N, R]
    pred_collision = np.zeros((N, R), dtype=bool)

    # Validity masks (broadcastable with the min-distance tensors):
    # pair_valid_scope[i, j, t]  = gt_valid[i, t] & gt_valid[j, t], j != i
    # pair_valid_scene[i, k, t]  = gt_valid[i, t] & scene_valid[k, t]
    pv_scope = gt_valid[:, None, :] & gt_valid[None, :, :]            # [N, N, T]
    # Drop the self-pair so agent i never becomes its own nearest neighbor.
    eye = np.eye(N, dtype=bool)[:, :, None]                            # [N, N, 1]
    pv_scope = pv_scope & ~eye                                          # [N, N, T]

    if has_scene:
        pv_scene = gt_valid[:, None, :] & scene_valid_b[None, :, :]    # [N, M, T]
    else:
        pv_scene = None

    for r in range(R):
        pr = pred_traj[:, r, :, :]                                     # [N, T, 2]
        # scope↔scope pairwise signed distance: [N, N, T]
        d_ss = pairwise_signed_distance(pr, gt_radius, pr, gt_radius)
        d_ss_masked = np.where(pv_scope, d_ss, INF)                    # invalid pairs → inf
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

        # Collision flag per agent: any pair at any t has signed distance ≤ 0
        # (mask first so invalid pairs don't count).
        ss_hit = (d_ss_masked <= 0.0).any(axis=(1, 2))                  # [N]
        if has_scene:
            sc_hit = (d_sc_masked <= 0.0).any(axis=(1, 2))              # [N]
            pred_collision[:, r] = ss_hit | sc_hit
        else:
            pred_collision[:, r] = ss_hit

    # --------------------------------------------------------------
    # GT min-distance per (i, t): everyone uses GT positions.
    # --------------------------------------------------------------
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

    # Validity for distance feature: agent i valid AND at least one other
    # agent valid at t (scope j ≠ i or scene k).
    has_other = pv_scope.any(axis=1)                                    # [N, T]
    if has_scene:
        has_other = has_other | pv_scene.any(axis=1)                    # [N, T]
    dist_valid = gt_valid & has_other                                   # [N, T]

    # --------------------------------------------------------------
    # distance_to_nearest_nll via KDE
    # --------------------------------------------------------------
    if dist_valid.any():
        pred_min_rt_last = np.moveaxis(pred_min, 1, -1)                 # [N, T, R]
        log_prob = log_kde(pred_min_rt_last, gt_min, bandwidth=bandwidth)  # [N, T]
        total_lp = float(np.sum(log_prob * dist_valid))
        n_valid = int(dist_valid.sum())
        dist_rec = {
            "metric": "distance_to_nearest_nll",
            "value": -(total_lp / n_valid),
            "n_agents": int(dist_valid.any(axis=-1).sum()),
            "weight": n_valid,
        }
    else:
        dist_rec = _empty("distance_to_nearest_nll")

    # --------------------------------------------------------------
    # Collision: per-agent Bernoulli NLL.
    # An agent is "collision-valid" iff there exists some t and partner
    # forming a time-overlapping valid pair.
    # --------------------------------------------------------------
    collision_valid = pv_scope.any(axis=(1, 2))                          # [N]
    if has_scene:
        collision_valid = collision_valid | pv_scene.any(axis=(1, 2))    # [N]

    if collision_valid.any():
        # GT collision indicator per agent
        gt_hit_ss = (gt_d_ss_masked <= 0.0).any(axis=(1, 2))             # [N]
        if has_scene:
            gt_hit_sc = (gt_d_sc_masked <= 0.0).any(axis=(1, 2))         # [N]
            gt_collision = (gt_hit_ss | gt_hit_sc).astype(np.float64)
        else:
            gt_collision = gt_hit_ss.astype(np.float64)

        # p_hat per agent = mean_r pred_collision[i, r]
        p_hat = pred_collision.mean(axis=1)                              # [N]
        nlls = bernoulli_nll(p_hat, gt_collision)                        # [N]
        nlls_valid = nlls[collision_valid]
        coll_rate = float(p_hat[collision_valid].mean())
        coll_nll = float(nlls_valid.mean())
        n_eval = int(collision_valid.sum())
        coll_rec = {
            "metric": "collision_nll",
            "value": coll_nll,
            "n_agents": n_eval,
            "weight": n_eval,
        }
        rate_rec = {
            "metric": "collision_rate",
            "value": coll_rate,
            "n_agents": n_eval,
            "weight": n_eval,
        }
    else:
        coll_rec = _empty("collision_nll")
        rate_rec = _empty("collision_rate")

    return [dist_rec, coll_rec, rate_rec]
