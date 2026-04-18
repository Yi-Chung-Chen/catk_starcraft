"""Horizon-sliced minADE @ {1s, 3s, 5s, 8s}.

Native fps is 16, future window is 128 steps (= 8 s). Indices used per horizon
are inclusive of the cumulative slice [0..int(seconds*fps)).
"""

from __future__ import annotations

import numpy as np

from src.starcraft.eval.load_rollout import ScenarioRollout

_HORIZONS = (1, 3, 5, 8)


def compute(scenario: ScenarioRollout, ctx=None) -> list:
    pred = scenario.pred_traj.astype(np.float32)        # [N, R, T, 2]
    gt = scenario.gt_traj.astype(np.float32)            # [N, T, 2]
    valid = scenario.gt_valid.astype(bool)              # [N, T]
    fps = scenario.native_fps
    T = pred.shape[2]

    diff = pred - gt[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)                # [N, R, T]

    out: list = []
    for sec in _HORIZONS:
        end = min(int(sec * fps), T)
        if end <= 0:
            out.append({"metric": f"min_ade_{sec}s", "value": None, "n_agents": 0})
            continue
        v_slice = valid[:, :end].astype(np.float32)     # [N, end]
        d_slice = dist[:, :, :end]                      # [N, R, end]
        n_valid = v_slice.sum(axis=-1).clip(min=1.0)    # [N]
        ade_per_rollout = (d_slice * v_slice[:, None, :]).sum(axis=-1) / n_valid[:, None]
        min_ade_per_agent = ade_per_rollout.min(axis=-1)
        agent_has_steps = valid[:, :end].any(axis=1)
        if not agent_has_steps.any():
            out.append({"metric": f"min_ade_{sec}s", "value": None, "n_agents": 0})
            continue
        value = float(min_ade_per_agent[agent_has_steps].mean())
        out.append({
            "metric": f"min_ade_{sec}s",
            "value": value,
            "n_agents": int(agent_has_steps.sum()),
        })
    return out
