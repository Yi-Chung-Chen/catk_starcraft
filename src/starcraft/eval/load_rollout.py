"""Load saved rollouts and pair them with raw-replay GT.

Predictions live in `{rollouts_dir}/{scenario_id}.h5`; raw GT lives in
`{replays_dir}/{map_name}/{scenario_id}.h5`. The agent_id (SC2 unit_tag)
column on the rollout side joins back to the post-`ever_alive` row ordering
on the replay side. The `ever_alive` filter is reapplied here from the same
helper the dataset uses so any future drift in the filter logic doesn't
silently misjoin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from src.starcraft.utils.sc_replay_io import apply_ever_alive_filter


@dataclass
class ScenarioRollout:
    scenario_id: str
    map_name: str
    observer: int                   # 1 or 2
    mode: str                       # "own" | "opponent"
    n_rollouts: int
    native_fps: int
    num_historical_steps: int

    # Predictions, indexed by target row
    agent_id: np.ndarray             # [N_target] int64 (SC2 unit_tag)
    pred_traj: np.ndarray            # [N_target, R, 128, 2]
    pred_head: np.ndarray            # [N_target, R, 128]
    visible_to_obs_future: np.ndarray  # [N_target, 16] bool

    # Aux predictions (own mode + use_aux_loss only)
    aux: Optional[dict] = None

    # GT pulled from raw replay, sliced/joined to target rows
    gt_traj: Optional[np.ndarray] = None         # [N_target, 128, 2]
    gt_head: Optional[np.ndarray] = None         # [N_target, 128]
    gt_valid: Optional[np.ndarray] = None        # [N_target, 128] bool — already AND'd with alive_at_current
    gt_alive_at_current: Optional[np.ndarray] = None  # [N_target] bool — alive at frame 16
    gt_radius: Optional[np.ndarray] = None       # [N_target] fp32

    # Provenance
    dataset_version: str = ""
    model_config_hash: str = ""
    stage: str = ""


def _load_replay_gt(
    replay_path: Path, agent_id: np.ndarray, num_historical_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read raw replay, apply ever_alive filter, join by agent_id.

    Raises ValueError if any saved agent_id is not found in the replay after
    the ever_alive filter — fail loud rather than silently produce wrong joins.
    """
    with h5py.File(replay_path, "r") as f:
        unit_tag = f["unit_data"]["global"]["unit_tag"][:]
        is_alive = f["unit_data"]["repeated"]["is_alive"][:].astype(bool)
        coordinate = f["unit_data"]["repeated"]["coordinate"][:].astype(np.float32)
        heading = f["unit_data"]["repeated"]["heading"][:].astype(np.float32)
        radius = f["unit_data"]["repeated"]["radius"][:].astype(np.float32)

    keep_idx = apply_ever_alive_filter(is_alive)
    unit_tag_kept = unit_tag[keep_idx]
    is_alive_kept = is_alive[:, keep_idx]
    coord_kept = coordinate[:, keep_idx]
    head_kept = heading[:, keep_idx]
    radius_kept = radius[:, keep_idx]

    tag_to_idx = {int(t): i for i, t in enumerate(unit_tag_kept)}
    try:
        join_idx = np.array(
            [tag_to_idx[int(t)] for t in agent_id], dtype=np.intp,
        )
    except KeyError as e:
        raise ValueError(
            f"agent_id {e} from rollout not found in replay {replay_path.name} "
            "after ever_alive filter — preprocessing or replay version mismatch."
        ) from e

    # Slice future steps and transpose to (N, T, ...) — same convention as
    # SCDataset and the in-decoder pred_traj/pred_head.
    fut_alive = is_alive_kept[num_historical_steps:, join_idx].T   # (N, 128)
    fut_coord = coord_kept[num_historical_steps:, join_idx, :2].transpose(1, 0, 2)  # (N, 128, 2)
    fut_head = head_kept[num_historical_steps:, join_idx].T        # (N, 128)
    # alive_at_current = valid_mask at the LAST historical step (frame 16).
    # Live minADE uses target_valid_full = valid_mask[:, num_hist:] & alive_at_current,
    # so we mirror exactly: AND fut_alive with alive_at_current here.
    alive_at_current = is_alive_kept[num_historical_steps - 1, join_idx]  # (N,)
    fut_alive_gated = fut_alive & alive_at_current[:, None]
    # Radius at current frame (consistent with SCDataset's snapshot at frame 16)
    rad_now = radius_kept[num_historical_steps - 1, join_idx]

    return fut_coord, fut_head, fut_alive_gated, alive_at_current, rad_now


def load_rollout(
    rollout_path: Path,
    replays_dir: Path,
    observer: int,
    mode: str,
    *,
    join_gt: bool = True,
) -> Optional[ScenarioRollout]:
    """Load one (observer, mode) view of a saved rollout file.

    Returns None if the requested group exists but has zero targets (empty
    metric_scope on the live side — e.g. opponent mode in a scenario where
    the observer can't see any opponents at frame 16).

    Raises FileNotFoundError if the rollout file or replay file is missing,
    KeyError if the requested observer/mode group is absent.
    """
    rollout_path = Path(rollout_path)
    replays_dir = Path(replays_dir)
    if not rollout_path.exists():
        raise FileNotFoundError(f"Rollout file not found: {rollout_path}")

    with h5py.File(rollout_path, "r") as f:
        scenario_id = str(f.attrs.get("scenario_id", rollout_path.stem))
        map_name = str(f.attrs["map_name"])
        n_rollouts = int(f.attrs.get("n_rollouts", 0))
        native_fps = int(f.attrs.get("native_fps", 16))
        num_historical_steps = int(f.attrs.get("num_historical_steps", 17))
        dataset_version = str(f.attrs.get("dataset_version", ""))
        model_config_hash = str(f.attrs.get("model_config_hash", ""))
        stage = str(f.attrs.get("stage", ""))

        grp_name = f"obs_p{observer}/{mode}"
        if grp_name not in f:
            raise KeyError(
                f"{rollout_path.name}: group {grp_name!r} not present "
                "(was this observer/mode disabled at save time?)"
            )
        grp = f[grp_name]
        n_target = int(grp.attrs.get("n_target", grp["agent_id"].shape[0]))
        if n_target == 0:
            return None

        agent_id = grp["agent_id"][:].astype(np.int64)
        pred_traj = grp["pred_traj"][:]
        pred_head = grp["pred_head"][:]
        vis_future = grp["visible_to_obs_future"][:]

        aux: Optional[dict] = None
        if "aux" in grp:
            aux = {k: grp["aux"][k][:] for k in grp["aux"].keys()}

    gt_traj = gt_head = gt_valid = gt_alive_at_current = gt_radius = None
    if join_gt:
        # Dataset layout: {replays_dir}/{split}/{map_name}/{scenario_id}.h5
        # where split ∈ {train, val, test}. The save layer stamps `stage` into
        # file attrs as "val" or "test"; that maps 1:1 to the split dir name.
        # Fall back to scanning all three splits if `stage` is missing (older
        # rollouts pre-dating the stage attr, or if the dataset's dir names
        # ever diverge from the stage label).
        candidates = []
        if stage in ("val", "test"):
            candidates.append(replays_dir / stage / map_name / f"{scenario_id}.h5")
        candidates.extend(
            replays_dir / s / map_name / f"{scenario_id}.h5"
            for s in ("test", "val", "train")
            if not candidates or candidates[0].parent.parent.name != s
        )
        replay_path = next((p for p in candidates if p.exists()), None)
        if replay_path is None:
            raise FileNotFoundError(
                f"Replay file not found for scenario {scenario_id} (map {map_name}). "
                f"Tried: {[str(p) for p in candidates]}. "
                "Expected layout: {replays_dir}/{split}/{map_name}/{scenario_id}.h5"
            )
        gt_traj, gt_head, gt_valid, gt_alive_at_current, gt_radius = _load_replay_gt(
            replay_path, agent_id, num_historical_steps,
        )

    return ScenarioRollout(
        scenario_id=scenario_id,
        map_name=map_name,
        observer=observer,
        mode=mode,
        n_rollouts=n_rollouts,
        native_fps=native_fps,
        num_historical_steps=num_historical_steps,
        agent_id=agent_id,
        pred_traj=pred_traj,
        pred_head=pred_head,
        visible_to_obs_future=vis_future,
        aux=aux,
        gt_traj=gt_traj,
        gt_head=gt_head,
        gt_valid=gt_valid,
        gt_alive_at_current=gt_alive_at_current,
        gt_radius=gt_radius,
        dataset_version=dataset_version,
        model_config_hash=model_config_hash,
        stage=stage,
    )
