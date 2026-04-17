"""Closed-loop rollout HDF5 I/O for offline metric evaluation.

Per-scenario file with predictions only — GT is re-read from raw replays at
metric-eval time via the SC2 unit_tag stored as agent_id (stable across the
dataset's ever_alive filter).

Schema per scenario file (`{rollout_save_dir}/{scenario_id}.h5`):

  attrs:
    scenario_id, map_name, num_historical_steps, num_future_steps,
    n_rollouts, native_fps, dataset_version, model_config_hash, stage

  /obs_p{1,2}/{own,opponent}    (group, attrs['n_target'] = N)
    agent_id              [N_target]            int64
    pred_traj             [N_target, R, 128, 2] fp16/fp32
    pred_head             [N_target, R, 128]    fp16/fp32
    visible_to_obs_future [N_target, 16]        bool
    aux/                                          # only when use_aux_loss AND mode == "own"
      has_action_pred       [N_target, R, 16]      bool
      has_target_pos_pred   [N_target, R, 16]      bool
      action_class_pred     [N_target, R, 16]      uint8
      target_pos_pred       [N_target, R, 16, 2]   fp16

Single-GPU runs write directly to `{scenario_id}.h5`. Under DDP we use
`{scenario_id}.rank{N}.h5` so two ranks can't clobber the same file under
sampler padding (rare, ≤ `world_size - 1` duplicates per run). Offline harness
globs `*.h5` and doesn't care which flavor it finds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)


_PRED_LOGIT_KEYS = (
    "target_pos_pred",
    "has_target_pos_logits",
    "action_class_logits",
    "has_action_logits",
)


def save_rollout_batch(
    save_dir: Path,
    data,
    filt_e: Dict[str, Tensor],
    keep_mask_e: Tensor,
    metric_scope: Tensor,
    teacher_force_mask: Tensor,
    pred_traj: Tensor,
    pred_head: Tensor,
    vis_to_obs: Tensor,
    aux_target_list: Optional[List[Dict[str, Tensor]]],
    observer_player: int,
    mode: str,
    file_attrs: dict,
    precision: str = "fp16",
    global_rank: int = 0,
    world_size: int = 1,
) -> None:
    """Append one (observer, mode) group per scenario in the batch.

    Saves only `metric_scope` rows — these are guaranteed disjoint from
    `teacher_force_mask`, so the GT-overlay applied to TF rows in the caller
    never bleeds into the saved data. If this ever becomes false (e.g., the
    scope is broadened to include TF rows), `pred_head` would also need a GT
    overlay to stay consistent with `pred_traj`. The assertion below guards
    that invariant.
    """
    assert not (teacher_force_mask & metric_scope).any(), (
        "metric_scope must be disjoint from teacher_force_mask — "
        "broadening the scope requires a parallel pred_head GT overlay."
    )
    assert mode in ("own", "opponent"), f"unknown mode {mode!r}"
    n_kept = metric_scope.shape[0]
    assert pred_traj.shape[0] == n_kept and pred_head.shape[0] == n_kept, (
        f"row mismatch: pred_traj {tuple(pred_traj.shape)} pred_head "
        f"{tuple(pred_head.shape)} vs metric_scope {tuple(metric_scope.shape)}"
    )
    assert pred_traj.shape[1] == pred_head.shape[1], (
        f"rollout dim mismatch: pred_traj {tuple(pred_traj.shape)} vs "
        f"pred_head {tuple(pred_head.shape)}"
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pred_dtype = np.float16 if precision == "fp16" else np.float32

    # Decode aux logits → predictions, but only for "own" mode. Opponent groups
    # would have zero'd intent inputs (filter_agents_for_perspective zeros
    # coarse_action / has_action / rel_target_pos / has_target_pos for
    # non-observer rows), making aux predictions meaningless.
    aux_decoded: Optional[Dict[str, Tensor]] = None
    if aux_target_list is not None and mode == "own":
        def _stack(k: str) -> Tensor:
            return torch.stack([d[k] for d in aux_target_list], dim=1)

        aux_decoded = {
            "has_action_pred":      (_stack("has_action_logits") > 0),
            "has_target_pos_pred":  (_stack("has_target_pos_logits") > 0),
            "action_class_pred":    _stack("action_class_logits").argmax(-1).to(torch.uint8),
            "target_pos_pred":      _stack("target_pos_pred"),
        }

    # Per-scenario split via filt_e["batch"]
    batch_idx = filt_e["batch"].cpu().numpy()
    num_graphs = int(filt_e["num_graphs"])
    scenario_ids = data["scenario_id"]
    map_names = data["map_name"]

    # agent_id is on the raw HeteroData, not propagated through tokenize_agent.
    # `keep_mask_e` is the [N_total] bool that produced filt_e's rows (in order),
    # so applying it to data["agent"]["id"] gives the per-row tags in filt_e ordering.
    agent_ids_np = data["agent"]["id"][keep_mask_e].cpu().numpy().astype(np.int64)
    metric_scope_np = metric_scope.cpu().numpy()
    pred_traj_np = pred_traj.detach().cpu().numpy()
    pred_head_np = pred_head.detach().cpu().numpy()
    vis_to_obs_np = vis_to_obs.cpu().numpy()  # [N_kept, 18]

    aux_decoded_np = None
    if aux_decoded is not None:
        aux_decoded_np = {
            k: v.detach().cpu().numpy() for k, v in aux_decoded.items()
        }

    for g in range(num_graphs):
        g_mask = batch_idx == g
        target_mask = metric_scope_np & g_mask
        n_target = int(target_mask.sum())

        scenario_id = scenario_ids[g] if isinstance(scenario_ids, list) else str(scenario_ids[g])
        map_name = map_names[g] if isinstance(map_names, list) else str(map_names[g])

        # Single-GPU: direct write to the final name, so any interrupt leaves
        # exactly the files that successfully closed. DDP: rank-local filename
        # to dodge the rare sampler-padding collision.
        if world_size > 1:
            path = save_dir / f"{scenario_id}.rank{global_rank}.h5"
        else:
            path = save_dir / f"{scenario_id}.h5"
        with h5py.File(path, "a") as f:
            # File-level attrs: write once
            if "scenario_id" not in f.attrs:
                f.attrs["scenario_id"] = scenario_id
                f.attrs["map_name"] = map_name
                for k, v in file_attrs.items():
                    f.attrs[k] = v

            grp_name = f"obs_p{observer_player}/{mode}"
            if grp_name in f:
                # Idempotent re-write (e.g., dev iteration); drop and recreate.
                del f[grp_name]
            grp = f.require_group(grp_name)
            grp.attrs["n_target"] = n_target

            if n_target == 0:
                # Empty group with shape-consistent zero-length datasets so
                # the offline loader can distinguish "no targets" from
                # "rank didn't run this".
                grp.create_dataset("agent_id", shape=(0,), dtype=np.int64)
                grp.create_dataset(
                    "pred_traj", shape=(0, pred_traj_np.shape[1], pred_traj_np.shape[2], 2),
                    dtype=pred_dtype,
                )
                grp.create_dataset(
                    "pred_head", shape=(0, pred_head_np.shape[1], pred_head_np.shape[2]),
                    dtype=pred_dtype,
                )
                grp.create_dataset("visible_to_obs_future", shape=(0, 16), dtype=np.bool_)
                continue

            # Rows for this scenario × metric_scope.
            # filt_e["id"] is in the kept (post-perspective-filter) ordering,
            # same ordering as pred_traj/pred_head/vis_to_obs/metric_scope.
            agent_id_g = agent_ids_np[target_mask]
            pred_traj_g = pred_traj_np[target_mask].astype(pred_dtype)
            pred_head_g = pred_head_np[target_mask].astype(pred_dtype)
            vis_g = vis_to_obs_np[target_mask, 2:].astype(np.bool_)  # 16 future steps

            grp.create_dataset("agent_id", data=agent_id_g)
            grp.create_dataset(
                "pred_traj", data=pred_traj_g,
                compression="gzip", compression_opts=1,
            )
            grp.create_dataset(
                "pred_head", data=pred_head_g,
                compression="gzip", compression_opts=1,
            )
            grp.create_dataset("visible_to_obs_future", data=vis_g)

            if aux_decoded_np is not None:
                aux_grp = grp.require_group("aux")
                for k, v in aux_decoded_np.items():
                    v_g = v[target_mask]
                    if k == "target_pos_pred":
                        v_g = v_g.astype(np.float16)
                    aux_grp.create_dataset(
                        k, data=v_g,
                        compression="gzip", compression_opts=1,
                    )


