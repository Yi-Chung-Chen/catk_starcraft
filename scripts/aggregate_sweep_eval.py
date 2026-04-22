"""Aggregate every per-run metrics.csv produced by eval_sweep_rcac.sh.

Walks `<sweep_root>/runs/<variant>_<maps>/metrics.csv`, concatenates into
`<sweep_root>/summary.csv` with added `run`, `variant`, `maps` columns, and
prints (and writes to `summary_table.txt`) a weighted-mean table shaped as:

    metric                   variant            id           ood
    horizon_ade              hmart            3.421       7.882
    horizon_ade              hmart_intent     3.102       7.012
    ...

Also computes Waymo-style composite scores (`own` mode), written to
`summary_composite.txt`:

    variant              maps    Meta↑   Kinematic↑  Interactive↑  Map↑   minADE↓
    smart_intent_aux     id      0.251   0.086       0.333         0.612  1.86
    ...

Composite definitions (equal-weight means of the available category metrics):
    Kinematic  = mean(linear_speed, linear_accel, angular_speed likelihoods)
    Interactive = distance_to_nearest_likelihood
    Map         = map_violation_likelihood
    Meta        = mean(Kinematic, Interactive, Map)
    minADE      = min_ade (reported; lower is better)

Weighting uses the same `weight` / `n_agents` fallback as `src.starcraft.eval.aggregate`.
Mode-sensitive metrics are printed per-mode (e.g. `min_ade/own`, `min_ade/opponent`).
NLL metrics are additionally reported in Waymo-style `exp(-NLL)` likelihood form.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.starcraft.eval import metrics as metric_registry  # noqa: E402


_RUN_DIR_RE = re.compile(r"^(?P<variant>.+)_(?P<maps>id|ood)$")


def _parse_run_dir(name: str) -> Tuple[str, str] | None:
    m = _RUN_DIR_RE.match(name)
    if m is None:
        return None
    return m.group("variant"), m.group("maps")


def _read_rows(csv_path: Path, variant: str, maps: str, run: str) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["run"] = run
            r["variant"] = variant
            r["maps"] = maps
            rows.append(r)
    return rows


def _record_weight(r: dict) -> int:
    w = r.get("weight")
    if w in (None, ""):
        w = r.get("n_agents", 0)
    try:
        return int(float(w or 0))
    except (TypeError, ValueError):
        return 0


def _weighted_mean(rows: List[dict]) -> Dict[Tuple[str, str, str, str], float]:
    """Key = (variant, maps, metric, mode). Value = weighted mean over rows."""
    sums: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    for r in rows:
        v = r.get("value")
        if v in (None, ""):
            continue
        w = _record_weight(r)
        if w == 0:
            continue
        try:
            v = float(v)
        except ValueError:
            continue
        key = (r["variant"], r["maps"], r["metric"], str(r.get("mode", "")))
        sums[key][0] += v * w
        sums[key][1] += w
    return {k: (s / n if n > 0 else float("nan")) for k, (s, n) in sums.items()}


def _display(metric: str, v: float) -> Tuple[str, float]:
    if metric.endswith("_nll"):
        return metric[:-4] + "_likelihood", (
            math.exp(-v) if math.isfinite(v) else float("nan")
        )
    return metric, v


_KINEMATIC_NLL = ("linear_speed_nll", "linear_accel_nll", "angular_speed_nll")
_INTERACTIVE_NLL = ("distance_to_nearest_nll",)
_MAP_NLL = ("map_violation_nll",)


def _as_likelihood(nll: float) -> float:
    return math.exp(-nll) if math.isfinite(nll) else float("nan")


def _mean_finite(xs: List[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _compute_composites(
    means: Dict[Tuple[str, str, str, str], float],
    mode_filter: str = "own",
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Collapse (variant, maps, metric, mode) means into Waymo-style composites.

    For `own` mode (the default), reports Meta/Kinematic/Interactive/Map from
    the NLL-derived likelihoods of that mode, plus min_ade collapsed across
    modes. A metric missing from `means` is dropped from its category average.
    """
    # Reshape: (variant, maps) -> {metric@mode: value}
    grouped: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for (variant, maps, metric, mode), v in means.items():
        grouped[(variant, maps)][f"{metric}@{mode}"] = v

    composites: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (variant, maps), m in grouped.items():
        def pull(names):
            return [
                _as_likelihood(m[f"{n}@{mode_filter}"])
                for n in names
                if f"{n}@{mode_filter}" in m
            ]

        kinematic = _mean_finite(pull(_KINEMATIC_NLL))
        interactive = _mean_finite(pull(_INTERACTIVE_NLL))
        map_score = _mean_finite(pull(_MAP_NLL))
        meta = _mean_finite([kinematic, interactive, map_score])

        # min_ade is typically written with mode="" (overall). Fall back to
        # `own` mode if the overall row is absent.
        min_ade = m.get("min_ade@", m.get(f"min_ade@{mode_filter}", float("nan")))

        composites[(variant, maps)] = {
            "Meta": meta,
            "Kinematic": kinematic,
            "Interactive": interactive,
            "Map": map_score,
            "minADE": min_ade,
        }
    return composites


def _format_composites(
    composites: Dict[Tuple[str, str], Dict[str, float]],
) -> str:
    header = (
        f"{'variant':22s}  {'maps':4s}  {'Meta↑':>8s}  "
        f"{'Kinematic↑':>11s}  {'Interactive↑':>13s}  {'Map↑':>7s}  "
        f"{'minADE↓':>8s}"
    )
    lines: List[str] = [header, "-" * len(header)]
    # Group by variant, id then ood rows consecutively
    variants = sorted({v for v, _ in composites})
    for variant in variants:
        for maps in ("id", "ood"):
            s = composites.get((variant, maps))
            if s is None:
                continue
            lines.append(
                f"{variant:22s}  {maps:4s}  "
                f"{s['Meta']:8.4f}  {s['Kinematic']:11.4f}  "
                f"{s['Interactive']:13.4f}  {s['Map']:7.4f}  "
                f"{s['minADE']:8.4f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_composite_csv_both(
    composites_own: Dict[Tuple[str, str], Dict[str, float]],
    composites_opp: Dict[Tuple[str, str], Dict[str, float]],
    out_path: Path,
) -> None:
    """One CSV with a `mode` column so pivot tables / sheet tools can split it.

    Row order: variant ascending, then (id/ood) × (own/opponent).
    """
    cols = ["variant", "maps", "mode", "Meta", "Kinematic", "Interactive", "Map", "minADE"]
    variants = sorted({v for v, _ in composites_own} | {v for v, _ in composites_opp})
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for variant in variants:
            for maps in ("id", "ood"):
                for mode_name, comp in (("own", composites_own), ("opponent", composites_opp)):
                    s = comp.get((variant, maps))
                    if s is None:
                        continue
                    w.writerow([
                        variant, maps, mode_name,
                        f"{s['Meta']:.6f}", f"{s['Kinematic']:.6f}",
                        f"{s['Interactive']:.6f}", f"{s['Map']:.6f}",
                        f"{s['minADE']:.6f}",
                    ])


def _format_table(means: Dict[Tuple[str, str, str, str], float],
                  mode_sensitive: set) -> str:
    # Always split by mode when present — many metrics (min_ade, horizon_ade,
    # etc.) emit both own and opponent rows, and collapsing them to "overall"
    # by last-write-wins silently hides mode-specific numbers.
    table: Dict[Tuple[str, str, str], float] = {}
    variants: set = set()
    metrics_display: set = set()
    for (variant, maps, metric, mode), v in means.items():
        disp_name, disp_v = _display(metric, v)
        if mode:
            disp_name = f"{disp_name}/{mode}"
        variants.add(variant)
        metrics_display.add(disp_name)
        table[(variant, disp_name, maps)] = disp_v

    variants_sorted = sorted(variants)
    metrics_sorted = sorted(metrics_display)
    # Column = (maps). Rows grouped by metric, variants across columns-of-rows.
    # We flatten as: metric | variant | id | ood
    out_lines: List[str] = []
    header = f"{'metric':30s}  {'variant':22s}  {'id':>10s}  {'ood':>10s}"
    out_lines.append(header)
    out_lines.append("-" * len(header))
    for metric in metrics_sorted:
        for variant in variants_sorted:
            id_v = table.get((variant, metric, "id"), float("nan"))
            ood_v = table.get((variant, metric, "ood"), float("nan"))
            if math.isnan(id_v) and math.isnan(ood_v):
                continue
            out_lines.append(
                f"{metric:30s}  {variant:22s}  {id_v:10.4f}  {ood_v:10.4f}"
            )
        out_lines.append("")
    return "\n".join(out_lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep_root", type=Path, required=True,
                    help="Dir containing runs/<variant>_<maps>/metrics.csv")
    args = ap.parse_args()

    sweep_root = args.sweep_root.resolve()
    runs_dir = sweep_root / "runs"
    if not runs_dir.is_dir():
        raise SystemExit(f"No runs/ under {sweep_root}")

    all_rows: List[dict] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = _parse_run_dir(run_dir.name)
        if parsed is None:
            print(f"(skip unrecognized run dir: {run_dir.name})")
            continue
        variant, maps = parsed
        csv_path = run_dir / "metrics.csv"
        if not csv_path.is_file():
            print(f"(skip missing metrics.csv: {csv_path})")
            continue
        all_rows.extend(_read_rows(csv_path, variant, maps, run_dir.name))

    if not all_rows:
        raise SystemExit(f"No metrics.csv files under {runs_dir}")

    # Write concatenated summary.csv
    summary_csv = sweep_root / "summary.csv"
    fieldnames: List[str] = []
    seen: set = set()
    for r in all_rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    # Weighted-mean table
    means = _weighted_mean(all_rows)
    mode_sensitive = set(metric_registry.MODE_SENSITIVE_METRICS)
    table = _format_table(means, mode_sensitive)

    summary_table = sweep_root / "summary_table.txt"
    summary_table.write_text(table)

    # Composite scores — the "report easily" view. Emit one table per mode.
    composites_own = _compute_composites(means, mode_filter="own")
    composites_opp = _compute_composites(means, mode_filter="opponent")
    own_table = _format_composites(composites_own)
    opp_table = _format_composites(composites_opp)
    stacked = (
        "=== Composite scores (mode=own) ===\n"
        + own_table
        + "\n=== Composite scores (mode=opponent) ===\n"
        + opp_table
    )
    composite_txt = sweep_root / "summary_composite.txt"
    composite_csv = sweep_root / "summary_composite.csv"
    composite_txt.write_text(stacked)
    _write_composite_csv_both(composites_own, composites_opp, composite_csv)

    print(table)
    print(stacked)
    print(f"wrote {len(all_rows)} rows → {summary_csv}")
    print(f"wrote metric table    → {summary_table}")
    print(f"wrote composite table → {composite_txt}")
    print(f"wrote composite csv   → {composite_csv}")


if __name__ == "__main__":
    main()
