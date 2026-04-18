"""Aggregate per-scenario metric records into summary tables."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


def write_csv(records: Iterable[Mapping], out_path: Path) -> None:
    """Dump per-(scenario, observer, mode, metric) records to CSV.

    Records may carry a per-row `weight` field in addition to `n_agents`;
    both are written out verbatim when present.
    """
    records = list(records)
    if not records:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Union of keys across all records so heterogeneous metric outputs
    # (e.g., ADE without `weight`, NLL with `weight`) all fit in one CSV.
    fieldnames: list = []
    seen: set = set()
    for r in records:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _record_weight(r: Mapping) -> int:
    """Per-record aggregation weight.

    Prefers explicit `weight` (set by NLL metrics over valid feature pairs);
    falls back to `n_agents` for legacy metrics like ADE.
    """
    w = r.get("weight")
    if w is None:
        return int(r.get("n_agents", 0) or 0)
    return int(w)


def summarize(records: Iterable[Mapping]) -> dict:
    """Weighted mean per (metric, mode), and per metric overall.

    Weight: uses per-record `weight` when present, else falls back to
    `n_agents`. Skips records with `value is None` or `weight == 0`.

    Return shape (stable for callers):
        {
          "breakdown": {metric: {mode: mean}},
          "overall":   {metric: mean},
        }
    """
    records = list(records)
    by_key: dict = defaultdict(lambda: [0.0, 0])      # (metric, mode) -> [sum, n]
    by_metric_overall: dict = defaultdict(lambda: [0.0, 0])
    for r in records:
        if r.get("value") is None:
            continue
        w = _record_weight(r)
        if w == 0:
            continue
        v = float(r["value"])
        key = (r["metric"], str(r["mode"]))
        by_key[key][0] += v * w
        by_key[key][1] += w
        by_metric_overall[r["metric"]][0] += v * w
        by_metric_overall[r["metric"]][1] += w

    breakdown: dict = defaultdict(dict)
    for (metric, mode), (s, n) in by_key.items():
        breakdown[metric][mode] = s / n if n > 0 else float("nan")
    overall = {
        metric: (s / n if n > 0 else float("nan"))
        for metric, (s, n) in by_metric_overall.items()
    }
    return {"breakdown": dict(breakdown), "overall": overall}
