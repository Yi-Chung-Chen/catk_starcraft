"""Pathing-grid loader for the offline rollout harness.

Reads the raw `{map_name}.h5` static map and exposes the pathing grid in
raw-SC2 convention: `True == blocked`, `False == walkable`. Intentionally
skips the inversion applied inside the training-time map encoder so metric
code can stay self-consistent with the SC2 API docs and external references.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np


@lru_cache(maxsize=16)
def load_pathing_grid(map_dir: str, map_name: str) -> np.ndarray:
    """Return [H, W] bool array; True = blocked, False = walkable.

    `map_dir` is a str (not Path) so the lru_cache key is hashable.
    """
    path = Path(map_dir) / f"{map_name}.h5"
    if not path.exists():
        raise FileNotFoundError(f"Pathing grid not found: {path}")
    with h5py.File(path, "r") as f:
        raw = f["pathing_grid"][:]
    # Raw convention: 0 = walkable, 1 = blocked (per CLAUDE.md). Cast to bool
    # so a nonzero value (blocked) is True.
    return np.asarray(raw).astype(bool)


def xy_to_grid(x: np.ndarray, y: np.ndarray, height: int, width: int) -> tuple:
    """Convert game coordinates (x, y) to pathing-grid (row, col) indices.

    Game Y increases upward from the bottom of the map; grid row 0 is the
    top. Shapes of x, y must match; returns two int arrays of the same shape.
    """
    col = np.clip(x.astype(np.int64), 0, width - 1)
    row = np.clip((height - y).astype(np.int64), 0, height - 1)
    return row, col
