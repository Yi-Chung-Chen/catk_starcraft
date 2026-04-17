"""Pure-numpy helpers shared between SCDataset (training pipeline) and the
offline rollout evaluation harness.

Kept torch/Lightning-free so the offline harness can be invoked from a
lighter environment (just numpy + h5py).
"""

from __future__ import annotations

import numpy as np


def apply_ever_alive_filter(is_alive: np.ndarray) -> np.ndarray:
    """Return raw-replay row indices that survive the ever-alive filter.

    Load-bearing: shared between SCDataset.get and the offline rollout loader so
    the offline join from saved agent_id back to raw-replay rows uses the same
    surviving subset the model originally saw.
    """
    ever_alive = is_alive.any(axis=0)
    keep_idx = np.where(ever_alive)[0]
    if len(keep_idx) == 0:
        keep_idx = np.array([0])
    return keep_idx
