"""Per-feature GT statistics committed into code, used to pick KDE bandwidths
at eval time without re-scanning the replay set on every run.

σ values come from a single one-shot scan of the adversarial train split
(2000 randomly-sampled replays) via `scripts/estimate_kde_bandwidths.py`,
with teleport-filtering (`speed ≤ 10 cells/s`) applied so dropship unloads,
warp-ins, and burrow/unburrow jumps don't dominate the std. Refresh these
numbers if the dataset changes meaningfully.

Bandwidth at eval time is derived via Silverman's rule-of-thumb
`bw = 1.06 · σ · R^{-1/5}` where R is the rollout count from the scenario
file — different R values require different bandwidths (more samples → can
afford a tighter kernel without undersmoothing).
"""

from __future__ import annotations

from typing import Dict


# Measured on datasets/StarCraftMotion_split_v2_adversarial train split,
# 2000 replays, teleport-gated at 10 cells/s, ~88M valid samples per
# feature. See scripts/estimate_kde_bandwidths.py for the extractor.
FEATURE_SIGMAS: Dict[str, float] = {
    "linear_speed_nll": 0.9700,     # cells / s
    "linear_accel_nll": 6.0832,     # cells / s^2
    "angular_speed_nll": 1.9095,    # rad / s
    "distance_to_nearest_nll": 1.7832,  # cells
}


def silverman(sigma: float, n_kernel: int) -> float:
    """Silverman rule-of-thumb KDE bandwidth.

    `bw = 1.06 · σ · n_kernel^{-1/5}`. `n_kernel` is the number of samples
    the KDE is fit from at eval time (R, the rollout count per query).
    """
    if n_kernel < 2:
        raise ValueError(
            f"Silverman requires n_kernel >= 2 (got {n_kernel}). "
            "A one-sample KDE is just a Gaussian at a single point; bandwidth "
            "selection doesn't apply."
        )
    return 1.06 * float(sigma) * (n_kernel ** (-1.0 / 5.0))


def default_bandwidth(metric_name: str, n_rollouts: int) -> float:
    """Convenience: Silverman bandwidth for a registered metric.

    Raises KeyError if the metric has no recorded σ.
    """
    return silverman(FEATURE_SIGMAS[metric_name], n_rollouts)
