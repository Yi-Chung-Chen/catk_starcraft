"""Aggregate per-scenario metric records into summary tables."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


def write_csv(records: Iterable[Mapping], out_path: Path) -> None:
    """Dump per-(scenario, observer, mode, metric) records to CSV."""
    records = list(records)
    if not records:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def summarize(records: Iterable[Mapping]) -> dict:
    """Weighted mean per (metric, observer, mode), weighted by `n_agents`.

    Skips records with `value == None` or `n_agents == 0`. Returns
    `{metric: {(observer, mode): mean}}` and a flat overall mean per metric.
    """
    records = list(records)
    by_key: dict = defaultdict(lambda: [0.0, 0])
    by_metric_overall: dict = defaultdict(lambda: [0.0, 0])
    for r in records:
        if r.get("value") is None:
            continue
        n = int(r.get("n_agents", 0) or 0)
        if n == 0:
            continue
        v = float(r["value"])
        key = (r["metric"], int(r["observer"]), str(r["mode"]))
        by_key[key][0] += v * n
        by_key[key][1] += n
        by_metric_overall[r["metric"]][0] += v * n
        by_metric_overall[r["metric"]][1] += n

    breakdown: dict = defaultdict(dict)
    for (metric, obs, mode), (s, n) in by_key.items():
        breakdown[metric][(obs, mode)] = s / n if n > 0 else float("nan")
    overall = {
        metric: (s / n if n > 0 else float("nan"))
        for metric, (s, n) in by_metric_overall.items()
    }
    return {"breakdown": dict(breakdown), "overall": overall}
