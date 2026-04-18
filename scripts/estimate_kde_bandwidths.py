"""Estimate KDE bandwidths for the StarCraft rollout NLL metrics.

For each KDE feature used by `kinematic_nll` and `interaction_nll`, this
script pools GT feature values across a whole rollout directory and applies
Silverman's rule-of-thumb `bw = 1.06 * σ * n^{-1/5}` to derive a per-metric
bandwidth. Collision and map-violation metrics are Bernoulli and have no
bandwidth, so they're excluded.

The emitted flag string can be pasted into `eval_sc_rollouts.py` via
`--kde_bandwidths`.

Usage:
    python scripts/estimate_kde_bandwidths.py \\
      --rollouts_dir logs/<run>/rollouts \\
      --replays_dir  datasets/StarCraftMotion_split_v2_adversarial \\
      --observers 1,2 --modes own,opponent \\
      --n_workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List

# Ensure repo root is importable when invoked as a plain script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.starcraft.eval.kinematics import (
    compute_angular_accel,
    compute_angular_speed,
    compute_linear_accel,
    compute_speed,
    pairwise_signed_distance,
)
from src.starcraft.eval.load_rollout import load_rollout


_FEATURE_NAMES = (
    "linear_speed_nll",
    "linear_accel_nll",
    "angular_speed_nll",
    "angular_accel_nll",
    "distance_to_nearest_nll",
)


def _extract_gt_features(scenario) -> Dict[str, np.ndarray]:
    """Compute the five GT features and flatten valid entries per scenario.

    Returns a dict feature_name → 1-D array of valid values.
    """
    dt = 1.0 / float(scenario.native_fps)
    gt_traj = scenario.gt_traj.astype(np.float32)          # [N, T, 2]
    gt_head = scenario.gt_head.astype(np.float32)          # [N, T]
    valid = scenario.gt_valid.astype(bool)                 # [N, T]
    gt_radius = scenario.gt_radius.astype(np.float32)

    speed_valid = valid[:, 1:] & valid[:, :-1]             # [N, T-1]
    accel_valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]  # [N, T-2]

    gt_speed = compute_speed(gt_traj, dt)                  # [N, T-1]
    gt_accel = compute_linear_accel(gt_speed, dt)          # [N, T-2]
    gt_ang_speed = compute_angular_speed(gt_head, dt)      # [N, T-1]
    gt_ang_accel = compute_angular_accel(gt_ang_speed, dt)  # [N, T-2]

    out: Dict[str, np.ndarray] = {
        "linear_speed_nll": gt_speed[speed_valid],
        "linear_accel_nll": gt_accel[accel_valid],
        "angular_speed_nll": gt_ang_speed[speed_valid],
        "angular_accel_nll": gt_ang_accel[accel_valid],
    }

    # Distance-to-nearest: pool GT positions of scope agents AND all scene
    # agents, then for each (scope agent, valid t) take min signed distance
    # to any other agent valid at t.
    scene_traj = scenario.gt_scene_traj
    scene_radius = scenario.gt_scene_radius
    scene_valid = scenario.gt_scene_valid

    INF = np.float32(1e9)
    N = gt_traj.shape[0]
    T = gt_traj.shape[1]

    if N >= 2:
        d_ss = pairwise_signed_distance(gt_traj, gt_radius, gt_traj, gt_radius)
        pv = valid[:, None, :] & valid[None, :, :]          # [N, N, T]
        eye = np.eye(N, dtype=bool)[:, :, None]
        pv = pv & ~eye
        d_ss_m = np.where(pv, d_ss, INF)
    else:
        d_ss_m = np.full((N, 1, T), INF, dtype=np.float32)

    if scene_traj is not None and scene_traj.shape[0] > 0:
        d_sc = pairwise_signed_distance(
            gt_traj, gt_radius,
            scene_traj.astype(np.float32), scene_radius.astype(np.float32),
        )
        pv_sc = valid[:, None, :] & scene_valid.astype(bool)[None, :, :]
        d_sc_m = np.where(pv_sc, d_sc, INF)
        min_sc = d_sc_m.min(axis=1)
    else:
        min_sc = np.full((N, T), INF, dtype=np.float32)

    min_ss = d_ss_m.min(axis=1)                              # [N, T]
    min_all = np.minimum(min_ss, min_sc)                     # [N, T]
    has_other = (d_ss_m < INF).any(axis=1) | (min_sc < INF)
    dist_valid = valid & has_other                           # [N, T]
    out["distance_to_nearest_nll"] = min_all[dist_valid]

    return out


def _process_file(
    rollout_path: Path,
    replays_dir: Path,
    observers: List[int],
    modes: List[str],
) -> Dict[str, List[np.ndarray]]:
    """Return per-feature list of flat value arrays from one scenario file."""
    buf: Dict[str, List[np.ndarray]] = {k: [] for k in _FEATURE_NAMES}
    for obs in observers:
        for mode in modes:
            try:
                scenario = load_rollout(rollout_path, replays_dir, obs, mode)
            except KeyError:
                continue
            if scenario is None:
                continue
            feats = _extract_gt_features(scenario)
            for k, v in feats.items():
                if v.size:
                    buf[k].append(v.astype(np.float32))
    return buf


def _silverman(values: np.ndarray, n_kernel: int) -> float:
    """Silverman's rule-of-thumb bandwidth for an `n_kernel`-sample KDE.

    `bw = 1.06 * σ * n_kernel^{-1/5}`. σ comes from the pooled GT values (a
    proxy for per-query feature spread); `n_kernel` is the number of samples
    available to the KDE at eval time — i.e. R, the rollout count. The
    dataset-wide sample count is *not* the right `n` here, because the KDE
    is refit per (agent, timestep) from only R samples.
    """
    if values.size < 2 or n_kernel < 2:
        return float("nan")
    sigma = float(values.std())
    if sigma == 0.0:
        return float("nan")
    return 1.06 * sigma * (n_kernel ** (-1.0 / 5.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts_dir", type=Path, required=True)
    parser.add_argument("--replays_dir", type=Path, required=True)
    parser.add_argument("--observers", default="1,2")
    parser.add_argument("--modes", default="own,opponent")
    parser.add_argument("--n_workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--max_scenarios", type=int, default=None,
                        help="Optional cap on scenarios scanned, for a quick estimate.")
    parser.add_argument("--n_rollouts", type=int, default=None,
                        help="Rollout count to use as n in Silverman's formula. "
                             "Defaults to the n_rollouts attr of the first scenario file.")
    args = parser.parse_args()

    rollouts_dir = args.rollouts_dir.resolve()
    replays_dir = args.replays_dir.resolve()
    if not rollouts_dir.is_dir():
        raise SystemExit(f"--rollouts_dir not a directory: {rollouts_dir}")
    if not replays_dir.is_dir():
        raise SystemExit(f"--replays_dir not a directory: {replays_dir}")

    observers = [int(x) for x in args.observers.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    files = sorted(rollouts_dir.glob("*.h5"))
    if args.max_scenarios is not None:
        files = files[: args.max_scenarios]
    if not files:
        raise SystemExit(f"No .h5 rollout files found under {rollouts_dir}")

    # Pull n_rollouts from the first file so the default matches the data.
    if args.n_rollouts is None:
        import h5py
        with h5py.File(files[0], "r") as f:
            n_rollouts = int(f.attrs.get("n_rollouts", 0))
        if n_rollouts < 2:
            fallback = 8
            print(
                f"[bw] file reports n_rollouts={n_rollouts} (KDE ill-defined); "
                f"falling back to n_kernel={fallback} for Silverman's rule. "
                "Pass --n_rollouts explicitly to override."
            )
            n_rollouts = fallback
    else:
        n_rollouts = int(args.n_rollouts)
    print(f"[bw] scanning {len(files)} scenarios, observers={observers}, "
          f"modes={modes}, workers={args.n_workers}, n_rollouts={n_rollouts}")

    worker = partial(
        _process_file, replays_dir=replays_dir,
        observers=observers, modes=modes,
    )
    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            per_file = pool.map(worker, files)
    else:
        per_file = [worker(p) for p in files]

    pooled: Dict[str, List[np.ndarray]] = {k: [] for k in _FEATURE_NAMES}
    for d in per_file:
        for k, lst in d.items():
            pooled[k].extend(lst)

    print("\n=== Feature statistics (GT only) ===")
    bandwidths: Dict[str, float] = {}
    for name in _FEATURE_NAMES:
        values = np.concatenate(pooled[name]) if pooled[name] else np.array([])
        if values.size == 0:
            print(f"  {name:28s}  n=0  (no valid samples)")
            continue
        sigma = float(values.std())
        bw = _silverman(values, n_kernel=n_rollouts)
        bandwidths[name] = bw
        print(
            f"  {name:28s}  n={values.size:>10d}  σ={sigma:>9.4f}  "
            f"min={values.min():>9.3f}  max={values.max():>9.3f}  "
            f"bw_silverman={bw:>9.4f}"
        )

    if bandwidths:
        flag_str = ",".join(f"{k}={v:.4f}" for k, v in bandwidths.items())
        print("\n=== Suggested flag ===")
        print(f"--kde_bandwidths {flag_str}")


if __name__ == "__main__":
    main()
