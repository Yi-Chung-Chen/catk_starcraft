"""minADE — min over rollouts of mean displacement error to GT.

Mirrors `src/smart/metrics/min_ade.py` exactly so the offline metric and the
live `val_closed/ADE_*_rollout` scalar agree to fp16 precision (the key
parity test in the verification plan).

Live (smart/metrics/min_ade.py):
    dist = norm(pred - target.unsqueeze(1))                      # [N, R, T]
    dist = (dist * valid.unsqueeze(1)).sum(-1).min(-1).values    # [N]
    dist = dist / (valid.sum(-1) + 1e-6)                         # [N]
    sum  += dist.sum()
    count += valid.any(-1).sum()
    return sum / count

Here `valid` is `target_valid_full & metric_scope.unsqueeze(1)` and
`target_valid_full` already includes the `alive_at_current` AND (sc_smart.py
:218-224). The offline loader pre-AND's `alive_at_current` into `gt_valid`
to match exactly.
"""

from __future__ import annotations

import numpy as np

from src.starcraft.eval.load_rollout import ScenarioRollout


def compute(scenario: ScenarioRollout) -> list:
    pred = scenario.pred_traj.astype(np.float32)        # [N, R, T, 2]
    gt = scenario.gt_traj.astype(np.float32)            # [N, T, 2]
    valid = scenario.gt_valid.astype(np.float32)        # [N, T] (already AND'd alive_at_current)

    if not valid.any():
        return [{"metric": "min_ade", "value": None, "n_agents": 0}]

    dist = np.linalg.norm(pred - gt[:, None, :, :], axis=-1)        # [N, R, T]
    masked_sum = (dist * valid[:, None, :]).sum(axis=-1)            # [N, R]
    per_agent_min = masked_sum.min(axis=-1)                          # [N]
    per_agent_norm = per_agent_min / (valid.sum(axis=-1) + 1e-6)    # [N]

    has_any_valid = valid.any(axis=-1)                               # [N]
    n_agents = int(has_any_valid.sum())
    if n_agents == 0:
        return [{"metric": "min_ade", "value": None, "n_agents": 0}]

    # Live computes scalar = sum(per_agent_norm) / sum(has_any_valid). Per-agent
    # contributions where !has_any_valid contribute 0 to the numerator (since
    # masked_sum is 0 there) and 0 to the count. Match exactly.
    value = float(per_agent_norm.sum() / n_agents)
    return [{"metric": "min_ade", "value": value, "n_agents": n_agents}]
