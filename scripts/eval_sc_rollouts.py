"""Offline metric harness for saved StarCraft closed-loop rollouts.

Walks `*.h5` files under `--rollouts_dir`, joins each scenario against the
matching raw replay under `--replays_dir/{map_name}/{scenario_id}.h5`, and
runs the requested metric registry over them. Writes a per-(scenario, observer,
mode, metric) CSV plus a printed weighted-mean summary.

Replay layout expected:
    {replays_dir}/{map_name}/{scenario_id}.h5

Example:
    python scripts/eval_sc_rollouts.py \\
      --rollouts_dir logs/<run>/rollouts \\
      --replays_dir  /scratch/.../StarCraftMotion_split_v2_adversarial \\
      --observers 1,2 --modes own,opponent \\
      --metrics min_ade,horizon_ade \\
      --n_workers 8 \\
      --out_csv metrics.csv
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from functools import partial
from pathlib import Path
from typing import List

# Ensure repo root is importable when invoked as a plain script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.starcraft.eval import metrics as metric_registry  # noqa: E402
from src.starcraft.eval.aggregate import summarize, write_csv  # noqa: E402
from src.starcraft.eval.load_rollout import load_rollout  # noqa: E402


def _process_scenario(
    rollout_path: Path,
    replays_dir: Path,
    observers: List[int],
    modes: List[str],
    metric_names: List[str],
) -> list:
    out: list = []
    for obs in observers:
        for mode in modes:
            try:
                scenario = load_rollout(rollout_path, replays_dir, obs, mode)
            except KeyError:
                # Group not present (e.g. opponent disabled at save time).
                continue
            if scenario is None:
                # Group present but n_target == 0 (e.g. opponent mode in a
                # scenario where the observer can't see any opponents at frame 16).
                continue
            for name in metric_names:
                fn = metric_registry.get(name)
                for rec in fn(scenario):
                    rec.update({
                        "scenario_id": scenario.scenario_id,
                        "map_name": scenario.map_name,
                        "observer": obs,
                        "mode": mode,
                        "stage": scenario.stage,
                    })
                    out.append(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts_dir", type=Path, required=True,
                        help="Dir containing per-scenario .h5 rollout files")
    parser.add_argument("--replays_dir", type=Path, required=True,
                        help="Raw replays root; layout {replays_dir}/{map}/{id}.h5")
    parser.add_argument("--observers", default="1,2",
                        help="Comma list of observers to evaluate (1,2)")
    parser.add_argument("--modes", default="own,opponent",
                        help="Comma list of modes to evaluate (own,opponent)")
    parser.add_argument("--metrics", default="min_ade,horizon_ade",
                        help=f"Comma list. Available: {sorted(metric_registry.REGISTRY.keys())}")
    parser.add_argument("--n_workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--out_csv", type=Path, default=None,
                        help="Optional CSV output path (default: <rollouts_dir>/metrics.csv)")
    args = parser.parse_args()

    rollouts_dir = args.rollouts_dir.resolve()
    replays_dir = args.replays_dir.resolve()
    if not rollouts_dir.is_dir():
        raise SystemExit(f"--rollouts_dir not a directory: {rollouts_dir}")
    if not replays_dir.is_dir():
        raise SystemExit(f"--replays_dir not a directory: {replays_dir}")

    observers = [int(x) for x in args.observers.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    metric_names = [x.strip() for x in args.metrics.split(",") if x.strip()]

    # Validate metrics early so a typo doesn't waste the worker pool's startup.
    for name in metric_names:
        metric_registry.get(name)

    rollout_files = sorted(rollouts_dir.glob("*.h5"))
    if not rollout_files:
        raise SystemExit(f"No .h5 rollout files found under {rollouts_dir}")
    print(f"[eval] {len(rollout_files)} scenarios, observers={observers}, "
          f"modes={modes}, metrics={metric_names}, workers={args.n_workers}")

    process_fn = partial(
        _process_scenario,
        replays_dir=replays_dir,
        observers=observers,
        modes=modes,
        metric_names=metric_names,
    )

    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            per_scenario = pool.map(process_fn, rollout_files)
    else:
        per_scenario = [process_fn(p) for p in rollout_files]

    records: list = []
    for r in per_scenario:
        records.extend(r)

    out_csv = args.out_csv or (rollouts_dir / "metrics.csv")
    write_csv(records, out_csv)

    summary = summarize(records)
    print("\n=== Summary (weighted by n_agents) ===")
    for metric, mean in sorted(summary["overall"].items()):
        print(f"  {metric:25s}  overall = {mean:.6f}")
    for metric, by_grp in sorted(summary["breakdown"].items()):
        print(f"\n  {metric}")
        for (obs, mode), v in sorted(by_grp.items()):
            print(f"    obs_p{obs}/{mode:8s}  = {v:.6f}")
    print(f"\n[eval] wrote {len(records)} records → {out_csv}")


if __name__ == "__main__":
    main()
