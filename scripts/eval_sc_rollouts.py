"""Offline metric harness for saved StarCraft closed-loop rollouts.

Walks `*.h5` files under `--rollouts_dir`, joins each scenario against the
matching raw replay under `--replays_dir/{split}/{map_name}/{scenario_id}.h5`,
and runs the requested metric registry over them. Writes a per-(scenario,
observer, mode, metric) CSV plus a printed weighted-mean summary.

Example:
    python scripts/eval_sc_rollouts.py \\
      --rollouts_dir logs/<run>/rollouts \\
      --replays_dir  /scratch/.../StarCraftMotion_split_v2_adversarial \\
      --observers 1,2 --modes own,opponent \\
      --metrics min_ade,horizon_ade,kinematic_nll,interaction_nll,map_violation \\
      --map_dir datasets/map_data/static \\
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
from typing import Dict, List

from tqdm import tqdm

# Ensure repo root is importable when invoked as a plain script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.starcraft.eval import metrics as metric_registry  # noqa: E402
from src.starcraft.eval.aggregate import summarize, write_csv  # noqa: E402
from src.starcraft.eval.load_rollout import load_rollout  # noqa: E402


def _parse_bandwidths(spec: str) -> Dict[str, float]:
    if not spec:
        return {}
    out: Dict[str, float] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(
                f"--kde_bandwidths entries must be name=value, got {chunk!r}"
            )
        name, value = chunk.split("=", 1)
        out[name.strip()] = float(value.strip())
    return out


def _process_scenario(
    rollout_path: Path,
    replays_dir: Path,
    observers: List[int],
    modes: List[str],
    metric_names: List[str],
    ctx: metric_registry.MetricCtx,
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
                for rec in fn(scenario, ctx):
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
                        help="Raw replays root; layout {replays_dir}/{split}/{map}/{id}.h5")
    parser.add_argument("--observers", default="1,2",
                        help="Comma list of observers to evaluate (1,2)")
    parser.add_argument("--modes", default="own,opponent",
                        help="Comma list of modes to evaluate (own,opponent)")
    parser.add_argument("--metrics",
                        default="min_ade,horizon_ade,kinematic_nll,interaction_nll,map_violation",
                        help=f"Comma list. Available: {sorted(metric_registry.REGISTRY.keys())}")
    parser.add_argument("--map_dir", type=Path,
                        default=Path("datasets/map_data/static"),
                        help="Dir containing {map_name}.h5 pathing grids "
                             "(used by map_violation metric)")
    parser.add_argument("--kde_bandwidths", default="",
                        help="Comma list of name=value overrides for KDE bandwidths "
                             "(e.g., linear_speed_nll=0.3)")
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

    bandwidths = _parse_bandwidths(args.kde_bandwidths)
    # map_dir is only meaningful for map_violation; pass as absolute path even
    # if it doesn't exist yet (the metric handles missing dir cleanly).
    map_dir = args.map_dir.resolve() if args.map_dir is not None else None
    ctx = metric_registry.MetricCtx(map_dir=map_dir, bandwidths=bandwidths)

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
        ctx=ctx,
    )

    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            per_scenario = list(tqdm(
                pool.imap_unordered(process_fn, rollout_files, chunksize=8),
                total=len(rollout_files),
                desc="eval",
                unit="scn",
            ))
    else:
        per_scenario = [
            process_fn(p)
            for p in tqdm(rollout_files, desc="eval", unit="scn")
        ]

    records: list = []
    for r in per_scenario:
        records.extend(r)

    out_csv = args.out_csv or (rollouts_dir / "metrics.csv")
    write_csv(records, out_csv)

    summary = summarize(records)
    mode_sensitive = metric_registry.MODE_SENSITIVE_METRICS
    import math

    def _display(metric: str, v: float) -> tuple:
        """Return (display_name, display_value) for a summary line.

        NLL metrics are reported in Waymo-style likelihood form
        `exp(-NLL) = exp(mean_log_prob)`, bounded in (0, ∞) but typically
        (0, 1] when bandwidth is set so the peak density ≤ 1. Lower NLL =
        higher likelihood = better. CSV still carries raw NLL for debugging.
        """
        if metric.endswith("_nll"):
            name = metric[:-len("_nll")] + "_likelihood"
            return name, math.exp(-v) if math.isfinite(v) else float("nan")
        return metric, v

    print("\n=== Summary (weighted) ===")
    for metric, mean in sorted(summary["overall"].items()):
        # For mode-sensitive metrics, mixing modes into a single "overall"
        # number is misleading — suppress the print but leave the value in
        # the summary dict for CSV/programmatic consumers.
        if metric in mode_sensitive:
            continue
        name, v = _display(metric, mean)
        print(f"  {name:28s}  overall = {v:.6f}")
    for metric, by_mode in sorted(summary["breakdown"].items()):
        name_head = _display(metric, 0.0)[0]
        print(f"\n  {name_head}")
        for mode, v in sorted(by_mode.items()):
            _, dv = _display(metric, v)
            print(f"    {mode:10s}  = {dv:.6f}")
    print(f"\n[eval] wrote {len(records)} records → {out_csv}")


if __name__ == "__main__":
    main()
