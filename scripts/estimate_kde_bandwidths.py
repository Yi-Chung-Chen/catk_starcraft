"""Measure per-feature GT sigma for the StarCraft rollout NLL metrics.

Walks raw replay HDF5s under `{replays_dir}/{split}/{map}/*.h5`, applies the
same ever_alive and teleport-filter gates the eval uses, pools the five
kinematic/interaction GT features, and reports σ per feature.

σ is the measurement. Eval-time bandwidth is derived via Silverman's rule
from σ + R (see src/starcraft/eval/feature_stats.py). To refresh the values
committed in `FEATURE_SIGMAS`, run this once and paste the sigma table.

Usage:
    python scripts/estimate_kde_bandwidths.py \\
      --replays_dir datasets/StarCraftMotion_split_v2_adversarial \\
      --splits train
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np

from src.starcraft.eval.kinematics import (
    compute_angular_speed,
    compute_linear_accel,
    compute_speed,
    pairwise_signed_distance,
    teleport_free_mask,
)
from src.starcraft.utils.sc_replay_io import apply_ever_alive_filter


# angular_accel dropped on purpose — SC2 units can switch heading instantly,
# so Δω/dt isn't a meaningful physical constraint (see kinematic_nll.py).
_FEATURE_NAMES = (
    "linear_speed_nll",
    "linear_accel_nll",
    "angular_speed_nll",
    "distance_to_nearest_nll",
)


def _extract_features(
    replay_path: Path, num_historical_steps: int, native_fps: int,
) -> Dict[str, np.ndarray]:
    """Open one replay, slice the future window, compute five GT features.

    Returns flat 1-D arrays of valid values per feature. Pooling across
    replays preserves the dataset-wide std.
    """
    with h5py.File(replay_path, "r") as f:
        is_alive = f["unit_data"]["repeated"]["is_alive"][:].astype(bool)
        coord = f["unit_data"]["repeated"]["coordinate"][:].astype(np.float32)
        heading = f["unit_data"]["repeated"]["heading"][:].astype(np.float32)
        radius = f["unit_data"]["repeated"]["radius"][:].astype(np.float32)

    keep = apply_ever_alive_filter(is_alive)
    is_alive = is_alive[:, keep]
    coord = coord[:, keep, :2]
    heading = heading[:, keep]
    radius_now = radius[num_historical_steps - 1, keep]

    # Mirror the ScenarioRollout convention: future window sliced after
    # num_historical_steps, AND'd with alive_at_current so steps from
    # currently-dead agents don't contribute.
    alive_now = is_alive[num_historical_steps - 1]              # (N,)
    gt_traj = coord[num_historical_steps:].transpose(1, 0, 2)    # (N, T, 2)
    gt_head = heading[num_historical_steps:].T                   # (N, T)
    gt_valid = is_alive[num_historical_steps:].T & alive_now[:, None]  # (N, T)

    dt = 1.0 / float(native_fps)
    N, T = gt_valid.shape

    # Kinematic masks: need consecutive valid timesteps AND non-teleport step.
    no_tele = teleport_free_mask(gt_traj, dt)                    # (N, T-1)
    speed_valid = gt_valid[:, 1:] & gt_valid[:, :-1] & no_tele
    accel_valid = (
        gt_valid[:, 2:] & gt_valid[:, 1:-1] & gt_valid[:, :-2]
        & no_tele[:, 1:] & no_tele[:, :-1]
    )

    gt_speed = compute_speed(gt_traj, dt)
    gt_accel = compute_linear_accel(gt_speed, dt)
    gt_ang_speed = compute_angular_speed(gt_head, dt)

    out: Dict[str, np.ndarray] = {
        "linear_speed_nll": gt_speed[speed_valid],
        "linear_accel_nll": gt_accel[accel_valid],
        "angular_speed_nll": gt_ang_speed[speed_valid],
    }

    # Distance-to-nearest on the full GT scene.
    if N >= 2:
        INF = np.float32(1e9)
        d = pairwise_signed_distance(gt_traj, radius_now, gt_traj, radius_now)
        pv = gt_valid[:, None, :] & gt_valid[None, :, :]
        pv = pv & ~np.eye(N, dtype=bool)[:, :, None]
        d_masked = np.where(pv, d, INF)
        min_dist = d_masked.min(axis=1)                          # (N, T)
        dist_valid = gt_valid & pv.any(axis=1)
        out["distance_to_nearest_nll"] = min_dist[dist_valid]
    else:
        out["distance_to_nearest_nll"] = np.array([], dtype=np.float32)

    return {k: v.astype(np.float32) for k, v in out.items()}


def _worker(
    replay_path: Path, num_historical_steps: int, native_fps: int,
) -> Dict[str, np.ndarray]:
    try:
        return _extract_features(replay_path, num_historical_steps, native_fps)
    except Exception as e:
        print(f"[bw] skipping {replay_path.name}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return {k: np.array([], dtype=np.float32) for k in _FEATURE_NAMES}


def _collect_replays(
    replays_dir: Path, splits: List[str], map_names: List[str] | None,
) -> List[Path]:
    files: List[Path] = []
    for split in splits:
        split_dir = replays_dir / split
        if not split_dir.is_dir():
            print(f"[bw] split dir missing, skipping: {split_dir}",
                  file=sys.stderr)
            continue
        for map_dir in sorted(split_dir.iterdir()):
            if not map_dir.is_dir():
                continue
            if map_names is not None and map_dir.name not in map_names:
                continue
            files.extend(sorted(map_dir.glob("*.h5")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays_dir", type=Path, required=True,
                        help="Replay root; layout {replays_dir}/{split}/{map}/{id}.h5")
    parser.add_argument("--splits", default="test",
                        help="Comma list of splits to scan (default: test)")
    parser.add_argument("--map_names", default=None,
                        help="Optional comma list to restrict to specific maps")
    parser.add_argument("--num_historical_steps", type=int, default=17,
                        help="Historical steps before the future window (default: 17)")
    parser.add_argument("--native_fps", type=int, default=16,
                        help="Frame rate (default: 16)")
    parser.add_argument("--n_workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--max_scenarios", type=int, default=None,
                        help="Optional cap on scenarios, for a quick estimate.")
    args = parser.parse_args()

    replays_dir = args.replays_dir.resolve()
    if not replays_dir.is_dir():
        raise SystemExit(f"--replays_dir not a directory: {replays_dir}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    map_names = None
    if args.map_names:
        map_names = [m.strip() for m in args.map_names.split(",") if m.strip()]

    files = _collect_replays(replays_dir, splits, map_names)
    if args.max_scenarios is not None and args.max_scenarios < len(files):
        # Deterministic shuffle so a small cap samples across all maps,
        # not just the first in alphabetical order.
        rng = np.random.default_rng(0)
        files = [files[i] for i in rng.permutation(len(files))[: args.max_scenarios]]
    if not files:
        raise SystemExit(
            f"No replay .h5 files found under {replays_dir} "
            f"(splits={splits}, maps={map_names})"
        )
    print(f"[bw] scanning {len(files)} replays  splits={splits}  "
          f"maps={'all' if map_names is None else map_names}  "
          f"workers={args.n_workers}")

    worker = partial(
        _worker,
        num_historical_steps=args.num_historical_steps,
        native_fps=args.native_fps,
    )
    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            per_file = pool.map(worker, files)
    else:
        per_file = [worker(p) for p in files]

    pooled: Dict[str, List[np.ndarray]] = {k: [] for k in _FEATURE_NAMES}
    for d in per_file:
        for k, v in d.items():
            if v.size:
                pooled[k].append(v)

    print("\n=== Feature statistics (GT only, teleport-gated) ===")
    sigma_table: Dict[str, float] = {}
    for name in _FEATURE_NAMES:
        values = np.concatenate(pooled[name]) if pooled[name] else np.array([])
        if values.size == 0:
            print(f"  {name:28s}  n=0  (no valid samples)")
            continue
        sigma = float(values.std())
        sigma_table[name] = sigma
        print(
            f"  {name:28s}  n={values.size:>10d}  σ={sigma:>9.4f}  "
            f"min={values.min():>9.3f}  max={values.max():>9.3f}"
        )

    if sigma_table:
        print("\n=== Paste into src/starcraft/eval/feature_stats.py ===")
        print("FEATURE_SIGMAS = {")
        for k, v in sigma_table.items():
            print(f'    "{k}": {v:.4f},')
        print("}")


if __name__ == "__main__":
    main()
