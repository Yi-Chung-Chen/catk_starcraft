"""Log-KDE and Bernoulli NLL utilities for rollout metric evaluation.

The WOSAC-style recipe: fit a Gaussian KDE to the R rollout samples of a
feature, evaluate the log-probability at the GT value. Same bandwidth is
used across all samples for a given metric.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


_LOG_2PI = float(np.log(2.0 * np.pi))


def log_kde(
    pred: np.ndarray,
    gt: np.ndarray,
    bandwidth: float,
    rollout_axis: int = -1,
) -> np.ndarray:
    """Gaussian-KDE log-probability of `gt` under the R rollout samples in `pred`.

    Parameters
    ----------
    pred
        Rollout samples; the `rollout_axis` dimension has length R.
    gt
        Same shape as `pred` with `rollout_axis` removed.
    bandwidth
        KDE bandwidth (Gaussian std).
    rollout_axis
        Axis of `pred` that indexes rollouts. Defaults to the last axis.

    Returns
    -------
    log_prob
        Same shape as `gt`. NaN-safe aggregation is the caller's job.
    """
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")

    # Move rollout axis to the front for a canonical broadcast against gt.
    pred = np.moveaxis(pred, rollout_axis, 0)                     # [R, ...]
    R = pred.shape[0]
    # gt should broadcast against pred with a leading singleton rollout axis.
    resid = (gt[None, ...] - pred) / bandwidth                    # [R, ...]
    # log N(x | μ=pred[r], σ=bandwidth) = -0.5*z^2 - log(σ) - 0.5*log(2π)
    log_component = -0.5 * resid * resid - np.log(bandwidth) - 0.5 * _LOG_2PI
    return logsumexp(log_component, axis=0) - np.log(R)            # [...]


def bernoulli_nll(
    p_hat: np.ndarray,
    gt_indicator: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-sample Bernoulli NLL `-(g*log p + (1-g)*log(1-p))`.

    `p_hat` and `gt_indicator` must broadcast. `p_hat` is clipped to
    `[eps, 1-eps]` so log(0) never happens on the boundary.
    """
    p = np.clip(np.asarray(p_hat, dtype=np.float64), eps, 1.0 - eps)
    g = np.asarray(gt_indicator, dtype=np.float64)
    return -(g * np.log(p) + (1.0 - g) * np.log(1.0 - p))
