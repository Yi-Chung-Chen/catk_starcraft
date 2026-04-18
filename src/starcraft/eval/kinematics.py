"""Pure-numpy kinematic feature extraction for offline rollout metrics.

All functions are torch-free. Shapes follow `[..., T]` for trajectories, with
`dt` in seconds. Angle derivatives are minimum-arc wrapped so they never
double-count across the ±π seam.
"""

from __future__ import annotations

import numpy as np


def wrap_angle_np(theta: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi). The modulo form maps both +pi and -pi to -pi."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def compute_speed(pos: np.ndarray, dt: float) -> np.ndarray:
    """Linear speed per step: ||x_{t+1} - x_t|| / dt.

    pos: [..., T, 2] → [..., T-1]
    """
    delta = np.diff(pos, axis=-2)
    return np.linalg.norm(delta, axis=-1) / dt


def compute_linear_accel(speed: np.ndarray, dt: float) -> np.ndarray:
    """Signed linear acceleration: (speed_{t+1} - speed_t) / dt.

    speed: [..., T-1] → [..., T-2]
    """
    return np.diff(speed, axis=-1) / dt


def compute_angular_speed(heading: np.ndarray, dt: float) -> np.ndarray:
    """Signed angular speed with min-arc wrap: wrap(h_{t+1} - h_t) / dt.

    heading: [..., T] → [..., T-1]
    """
    return wrap_angle_np(np.diff(heading, axis=-1)) / dt


def compute_angular_accel(ang_speed: np.ndarray, dt: float) -> np.ndarray:
    """Second derivative of heading: (w_{t+1} - w_t) / dt.

    No wrapping — angular speed is an unbounded real (rad/s), not an angle
    on the unit circle. Wrapping would corrupt any legitimate acceleration
    where |Δω| > π rad/s between adjacent frames (easy at 16 fps).

    ang_speed: [..., T-1] → [..., T-2]
    """
    return np.diff(ang_speed, axis=-1) / dt


def pairwise_signed_distance(
    pos_a: np.ndarray, rad_a: np.ndarray,
    pos_b: np.ndarray, rad_b: np.ndarray,
) -> np.ndarray:
    """Minkowski signed distance between two sets of circles, per timestep.

    ||p_a[i] - p_b[j]|| - (r_a[i] + r_b[j])

    pos_a: [A, T, 2], rad_a: [A]
    pos_b: [B, T, 2], rad_b: [B]
    returns: [A, B, T]
    """
    # Broadcast: [A, 1, T, 2] - [1, B, T, 2] -> [A, B, T, 2]
    diff = pos_a[:, None, :, :] - pos_b[None, :, :, :]
    dist = np.linalg.norm(diff, axis=-1)                  # [A, B, T]
    return dist - (rad_a[:, None, None] + rad_b[None, :, None])
