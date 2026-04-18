"""Unit tests for the offline rollout metric utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.starcraft.eval.kinematics import (
    compute_angular_accel,
    compute_angular_speed,
    compute_linear_accel,
    compute_speed,
    pairwise_signed_distance,
    wrap_angle_np,
)
from src.starcraft.eval.log_kde import bernoulli_nll, log_kde


# ---------------------------------------------------------------------------
# log_kde
# ---------------------------------------------------------------------------

def test_log_kde_single_sample_matches_gaussian():
    # With R=1 the KDE reduces to Normal(μ=pred[0], σ=bw).
    pred = np.array([[1.5]], dtype=np.float64)       # [N=1, R=1]
    gt = np.array([2.0], dtype=np.float64)            # [N=1]
    bw = 0.5
    log_prob = log_kde(pred, gt, bandwidth=bw, rollout_axis=-1)

    z = (gt - pred[:, 0]) / bw
    expected = -0.5 * z * z - np.log(bw) - 0.5 * np.log(2.0 * np.pi)
    np.testing.assert_allclose(log_prob, expected, atol=1e-8)


def test_log_kde_matches_direct_gaussian_mixture():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(3, 5)).astype(np.float64)   # [N, R]
    gt = rng.normal(size=(3,)).astype(np.float64)       # [N]
    bw = 0.7

    got = log_kde(pred, gt, bandwidth=bw, rollout_axis=-1)

    # Reference: direct mixture of Gaussians
    R = pred.shape[-1]
    diffs = (gt[:, None] - pred) / bw
    comp = -0.5 * diffs ** 2 - np.log(bw) - 0.5 * np.log(2.0 * np.pi)
    ref = np.log(np.exp(comp).mean(axis=-1))
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_log_kde_rollout_axis_non_last():
    rng = np.random.default_rng(1)
    # [R, N, T]
    pred = rng.normal(size=(4, 2, 3))
    gt = rng.normal(size=(2, 3))
    bw = 0.3

    got = log_kde(pred, gt, bandwidth=bw, rollout_axis=0)
    ref = log_kde(np.moveaxis(pred, 0, -1), gt, bandwidth=bw, rollout_axis=-1)
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_log_kde_rejects_nonpositive_bandwidth():
    with pytest.raises(ValueError):
        log_kde(np.zeros((1, 1)), np.zeros((1,)), bandwidth=0.0)


# ---------------------------------------------------------------------------
# bernoulli_nll
# ---------------------------------------------------------------------------

def test_bernoulli_nll_analytic():
    # GT=1, p=0.8 -> -log(0.8);  GT=0, p=0.3 -> -log(0.7)
    p = np.array([0.8, 0.3])
    g = np.array([1.0, 0.0])
    got = bernoulli_nll(p, g)
    np.testing.assert_allclose(got, [-np.log(0.8), -np.log(0.7)], atol=1e-10)


def test_bernoulli_nll_clips_extremes():
    # p=0 with g=1 would blow up without clipping.
    p = np.array([0.0, 1.0])
    g = np.array([1.0, 0.0])
    got = bernoulli_nll(p, g, eps=1e-3)
    assert np.all(np.isfinite(got))


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def test_wrap_angle_np_basic():
    angles = np.array([0.0, np.pi, -np.pi, 3 * np.pi, -3 * np.pi, 0.5])
    got = wrap_angle_np(angles)
    # [-pi, pi) convention (both +pi and -pi map to -pi via the modulo form).
    for v in got:
        assert -np.pi - 1e-12 <= v < np.pi
    np.testing.assert_allclose(got[5], 0.5, atol=1e-12)


def test_compute_speed_linear_trajectory():
    # Agent moving at constant velocity (3, 4) per sec; dt=1 -> speed = 5
    T = 4
    pos = np.stack([np.arange(T) * 3.0, np.arange(T) * 4.0], axis=-1)  # [T, 2]
    speed = compute_speed(pos, dt=1.0)
    np.testing.assert_allclose(speed, np.full(T - 1, 5.0), atol=1e-10)


def test_compute_linear_accel_constant_speed():
    speed = np.array([5.0, 5.0, 5.0, 5.0])
    accel = compute_linear_accel(speed, dt=0.1)
    np.testing.assert_allclose(accel, np.zeros(3), atol=1e-10)


def test_compute_angular_speed_constant_rotation_and_wrap():
    # Rotating 1 rad/step at dt=0.5 -> angular speed = 2 rad/s
    heading = np.array([0.0, 1.0, 2.0, 3.0, -2.2832])  # last step crosses -pi wrap
    ang = compute_angular_speed(heading, dt=0.5)
    # First three diffs: 1, 1, 1 -> 2 rad/s each
    np.testing.assert_allclose(ang[:3], [2.0, 2.0, 2.0], atol=1e-10)
    # Last diff is wrapped — must be within (-2π/dt, 2π/dt] and monotonic in
    # magnitude; just assert it's finite and below the 2π/dt bound.
    assert np.isfinite(ang[-1]) and abs(ang[-1]) <= 2 * np.pi / 0.5 + 1e-9


def test_compute_angular_accel_constant_angular_speed():
    ang_speed = np.array([1.0, 1.0, 1.0, 1.0])
    accel = compute_angular_accel(ang_speed, dt=0.1)
    np.testing.assert_allclose(accel, np.zeros(3), atol=1e-10)


def test_compute_angular_accel_does_not_wrap_large_jumps():
    # ω diff of 4 rad/s: if this were (wrongly) treated as an angle and wrapped,
    # it would come back as ~4 - 2π ≈ -2.28 rad/s. The physically correct
    # behavior keeps 4 rad/s, giving accel = 4/dt.
    ang_speed = np.array([0.0, 4.0])
    accel = compute_angular_accel(ang_speed, dt=0.0625)  # 16 fps
    np.testing.assert_allclose(accel, [4.0 / 0.0625], atol=1e-10)


def test_pairwise_signed_distance_two_circles():
    # Circle A at (0,0) r=1, circle B at (3,0) r=1 -> signed distance = 1
    pos_a = np.array([[[0.0, 0.0]]], dtype=np.float32)         # [A=1, T=1, 2]
    pos_b = np.array([[[3.0, 0.0]]], dtype=np.float32)         # [B=1, T=1, 2]
    rad_a = np.array([1.0], dtype=np.float32)
    rad_b = np.array([1.0], dtype=np.float32)
    d = pairwise_signed_distance(pos_a, rad_a, pos_b, rad_b)
    assert d.shape == (1, 1, 1)
    np.testing.assert_allclose(d[0, 0, 0], 1.0, atol=1e-6)


def test_pairwise_signed_distance_overlap_is_negative():
    # Circles overlap: signed distance < 0
    pos_a = np.array([[[0.0, 0.0]]], dtype=np.float32)
    pos_b = np.array([[[1.0, 0.0]]], dtype=np.float32)
    rad_a = np.array([1.0], dtype=np.float32)
    rad_b = np.array([1.0], dtype=np.float32)
    d = pairwise_signed_distance(pos_a, rad_a, pos_b, rad_b)
    assert d[0, 0, 0] < 0


# ---------------------------------------------------------------------------
# Aggregate weight handling
# ---------------------------------------------------------------------------

def test_aggregate_uses_weight_when_present():
    from src.starcraft.eval.aggregate import summarize

    records = [
        # Scenario A: mode "own", value 1.0, weight 100
        {"metric": "m", "value": 1.0, "n_agents": 2, "weight": 100,
         "observer": 1, "mode": "own"},
        # Scenario B: mode "own", value 3.0, weight 300
        {"metric": "m", "value": 3.0, "n_agents": 2, "weight": 300,
         "observer": 2, "mode": "own"},
    ]
    s = summarize(records)
    # Weighted mean = (1*100 + 3*300) / 400 = 1000/400 = 2.5
    assert s["breakdown"]["m"]["own"] == pytest.approx(2.5)
    assert s["overall"]["m"] == pytest.approx(2.5)


def test_aggregate_falls_back_to_n_agents():
    from src.starcraft.eval.aggregate import summarize

    # No `weight` field -> fall back to n_agents
    records = [
        {"metric": "m", "value": 2.0, "n_agents": 5,
         "observer": 1, "mode": "own"},
        {"metric": "m", "value": 4.0, "n_agents": 5,
         "observer": 1, "mode": "opponent"},
    ]
    s = summarize(records)
    # Per-mode averages: own=2.0, opponent=4.0; overall (by n_agents)=3.0
    assert s["breakdown"]["m"]["own"] == pytest.approx(2.0)
    assert s["breakdown"]["m"]["opponent"] == pytest.approx(4.0)
    assert s["overall"]["m"] == pytest.approx(3.0)


def test_aggregate_skips_none_and_zero_weight():
    from src.starcraft.eval.aggregate import summarize

    records = [
        {"metric": "m", "value": None, "n_agents": 0, "weight": 0,
         "observer": 1, "mode": "own"},
        {"metric": "m", "value": 5.0, "n_agents": 1, "weight": 10,
         "observer": 1, "mode": "own"},
    ]
    s = summarize(records)
    assert s["breakdown"]["m"]["own"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# End-to-end kinematic_nll (synthetic scenario)
# ---------------------------------------------------------------------------

def _make_scenario(N=2, R=3, T=8, fps=16, include_scene=False):
    """Build a minimal ScenarioRollout-shaped object for metric testing."""
    from src.starcraft.eval.load_rollout import ScenarioRollout

    rng = np.random.default_rng(42)
    pred_traj = rng.normal(size=(N, R, T, 2)).astype(np.float32)
    pred_head = rng.normal(size=(N, R, T)).astype(np.float32)
    gt_traj = rng.normal(size=(N, T, 2)).astype(np.float32)
    gt_head = rng.normal(size=(N, T)).astype(np.float32)
    gt_valid = np.ones((N, T), dtype=bool)
    gt_radius = np.ones(N, dtype=np.float32)

    scene_kwargs = {}
    if include_scene:
        M = 1
        scene_kwargs = {
            "gt_scene_traj": rng.normal(size=(M, T, 2)).astype(np.float32),
            "gt_scene_head": rng.normal(size=(M, T)).astype(np.float32),
            "gt_scene_valid": np.ones((M, T), dtype=bool),
            "gt_scene_radius": np.ones(M, dtype=np.float32),
            "gt_scene_is_flying": np.zeros((M, T), dtype=bool),
            "gt_scene_owner": np.array([2], dtype=np.int64),
        }

    return ScenarioRollout(
        scenario_id="test", map_name="TestMap",
        observer=1, mode="own",
        n_rollouts=R, native_fps=fps, num_historical_steps=17,
        agent_id=np.arange(N, dtype=np.int64),
        pred_traj=pred_traj, pred_head=pred_head,
        visible_to_obs_future=np.ones((N, 16), dtype=bool),
        gt_traj=gt_traj, gt_head=gt_head,
        gt_valid=gt_valid, gt_alive_at_current=np.ones(N, dtype=bool),
        gt_radius=gt_radius,
        gt_is_flying=np.zeros((N, T), dtype=bool),
        **scene_kwargs,
    )


def test_kinematic_nll_emits_four_records():
    from src.starcraft.eval.metrics import kinematic_nll

    scenario = _make_scenario()
    out = kinematic_nll.compute(scenario, ctx=None)
    names = {rec["metric"] for rec in out}
    assert names == {
        "linear_speed_nll",
        "linear_accel_nll",
        "angular_speed_nll",
        "angular_accel_nll",
    }
    for rec in out:
        assert rec["value"] is None or np.isfinite(rec["value"])
        assert rec["weight"] >= 0


def test_kinematic_nll_zero_valid_yields_empty():
    from src.starcraft.eval.metrics import kinematic_nll

    scenario = _make_scenario()
    scenario.gt_valid[:] = False
    out = kinematic_nll.compute(scenario, ctx=None)
    for rec in out:
        assert rec["value"] is None
        assert rec["weight"] == 0


def test_kinematic_nll_matched_rollouts_low_nll():
    """If all rollouts exactly equal GT, NLL should be the log-normalizer."""
    from src.starcraft.eval.metrics import kinematic_nll

    N, R, T, fps = 1, 4, 8, 16
    # Straight-line trajectory so speed is constant.
    base = np.stack([np.arange(T) * 2.0, np.zeros(T)], axis=-1).astype(np.float32)
    pred = np.broadcast_to(base, (N, R, T, 2)).copy()
    gt = np.broadcast_to(base, (N, T, 2)).copy()

    from src.starcraft.eval.load_rollout import ScenarioRollout
    s = ScenarioRollout(
        scenario_id="s", map_name="M", observer=1, mode="own",
        n_rollouts=R, native_fps=fps, num_historical_steps=17,
        agent_id=np.zeros(N, dtype=np.int64),
        pred_traj=pred, pred_head=np.zeros((N, R, T), dtype=np.float32),
        visible_to_obs_future=np.ones((N, 16), dtype=bool),
        gt_traj=gt, gt_head=np.zeros((N, T), dtype=np.float32),
        gt_valid=np.ones((N, T), dtype=bool),
        gt_alive_at_current=np.ones(N, dtype=bool),
        gt_radius=np.ones(N, dtype=np.float32),
        gt_is_flying=np.zeros((N, T), dtype=bool),
    )
    out = {r["metric"]: r for r in kinematic_nll.compute(s, ctx=None)}
    # linear_speed_nll with bw=0.5: log p = -log(0.5) - 0.5*log(2π) ≈ 0.0439
    # so mean_nll ≈ -0.0439
    bw = 0.5
    expected_nll = -(-np.log(bw) - 0.5 * np.log(2.0 * np.pi))
    assert abs(out["linear_speed_nll"]["value"] - expected_nll) < 1e-4


# ---------------------------------------------------------------------------
# End-to-end map_violation
# ---------------------------------------------------------------------------

def test_map_violation_center_cell(tmp_path, monkeypatch):
    """Check that a unit placed on a blocked cell is flagged."""
    import h5py
    from src.starcraft.eval.metrics import map_violation
    from src.starcraft.eval import metrics as metric_registry

    # Build a 10×10 pathing grid with a single blocked cell at (row=3, col=7).
    # Game coords: col=7, y = H - row - 0.5? Use integer center (7.5, 6.5).
    H, W = 10, 10
    path = np.zeros((H, W), dtype=bool)
    path[3, 7] = True  # blocked
    map_dir = tmp_path / "static"
    map_dir.mkdir()
    with h5py.File(map_dir / "TestMap.h5", "w") as f:
        f.create_dataset("pathing_grid", data=path.astype(np.uint8))

    # Scenario: 1 agent, trajectory placing center on the blocked cell.
    N, R, T = 1, 2, 3
    from src.starcraft.eval.load_rollout import ScenarioRollout
    # x=7.5, y = H - 3 - 0.5 = 6.5 places us in row=3, col=7
    pred = np.tile(
        np.array([[7.5, 6.5]], dtype=np.float32), (N, R, T, 1),
    ).reshape(N, R, T, 2)
    gt = np.zeros((N, T, 2), dtype=np.float32)  # GT far from blocked cell
    s = ScenarioRollout(
        scenario_id="s", map_name="TestMap", observer=1, mode="own",
        n_rollouts=R, native_fps=16, num_historical_steps=17,
        agent_id=np.zeros(N, dtype=np.int64),
        pred_traj=pred, pred_head=np.zeros((N, R, T), dtype=np.float32),
        visible_to_obs_future=np.ones((N, 16), dtype=bool),
        gt_traj=gt, gt_head=np.zeros((N, T), dtype=np.float32),
        gt_valid=np.ones((N, T), dtype=bool),
        gt_alive_at_current=np.ones(N, dtype=bool),
        gt_radius=np.ones(N, dtype=np.float32),
        gt_is_flying=np.zeros((N, T), dtype=bool),
    )

    ctx = metric_registry.MetricCtx(map_dir=map_dir, bandwidths={})
    out = {r["metric"]: r for r in map_violation.compute(s, ctx=ctx)}

    # All rollouts violate -> rate = 1.0
    assert out["map_violation_rate"]["value"] == pytest.approx(1.0)
    # GT doesn't violate (stays at origin). Bernoulli NLL with p=1, g=0 → -log(eps).
    assert out["map_violation_nll"]["value"] > 5.0


def test_map_violation_out_of_bounds_is_violation(tmp_path):
    """Unit placed outside the pathing grid bounds must count as a violation.

    Relies on the invariant that every SC2 static map's border cells are
    pathing=1 (blocked). With clip-to-bounds indexing, an OOB position maps
    onto a border cell; the border being blocked makes the lookup report
    True (violation). Guards against future dataset changes that might break
    that invariant.
    """
    import h5py
    from src.starcraft.eval.metrics import map_violation
    from src.starcraft.eval import metrics as metric_registry

    H, W = 10, 10
    # Border all blocked (matches the real dataset invariant); interior walkable.
    path = np.zeros((H, W), dtype=bool)
    path[0, :] = True
    path[-1, :] = True
    path[:, 0] = True
    path[:, -1] = True
    map_dir = tmp_path / "static"
    map_dir.mkdir()
    with h5py.File(map_dir / "TestMap.h5", "w") as f:
        f.create_dataset("pathing_grid", data=path.astype(np.uint8))

    N, R, T = 1, 2, 3
    from src.starcraft.eval.load_rollout import ScenarioRollout
    # x = W + 5 (well outside); clip → col=W-1 → border → blocked
    pred = np.tile(
        np.array([[W + 5.0, 5.5]], dtype=np.float32), (N, R, T, 1),
    ).reshape(N, R, T, 2)
    gt = np.tile(
        np.array([[5.5, 5.5]], dtype=np.float32), (N, T, 1),  # interior, walkable
    ).reshape(N, T, 2)
    s = ScenarioRollout(
        scenario_id="s", map_name="TestMap", observer=1, mode="own",
        n_rollouts=R, native_fps=16, num_historical_steps=17,
        agent_id=np.zeros(N, dtype=np.int64),
        pred_traj=pred, pred_head=np.zeros((N, R, T), dtype=np.float32),
        visible_to_obs_future=np.ones((N, 16), dtype=bool),
        gt_traj=gt, gt_head=np.zeros((N, T), dtype=np.float32),
        gt_valid=np.ones((N, T), dtype=bool),
        gt_alive_at_current=np.ones(N, dtype=bool),
        gt_radius=np.ones(N, dtype=np.float32),
        gt_is_flying=np.zeros((N, T), dtype=bool),
    )
    ctx = metric_registry.MetricCtx(map_dir=map_dir, bandwidths={})
    out = {r["metric"]: r for r in map_violation.compute(s, ctx=ctx)}
    assert out["map_violation_rate"]["value"] == pytest.approx(1.0)


def test_map_violation_skips_flying_units(tmp_path):
    import h5py
    from src.starcraft.eval.metrics import map_violation
    from src.starcraft.eval import metrics as metric_registry

    H, W = 10, 10
    path = np.zeros((H, W), dtype=bool)
    path[3, 7] = True
    map_dir = tmp_path / "static"
    map_dir.mkdir()
    with h5py.File(map_dir / "TestMap.h5", "w") as f:
        f.create_dataset("pathing_grid", data=path.astype(np.uint8))

    N, R, T = 1, 2, 3
    from src.starcraft.eval.load_rollout import ScenarioRollout
    pred = np.tile(
        np.array([[7.5, 6.5]], dtype=np.float32), (N, R, T, 1),
    ).reshape(N, R, T, 2)
    gt = np.zeros((N, T, 2), dtype=np.float32)
    s = ScenarioRollout(
        scenario_id="s", map_name="TestMap", observer=1, mode="own",
        n_rollouts=R, native_fps=16, num_historical_steps=17,
        agent_id=np.zeros(N, dtype=np.int64),
        pred_traj=pred, pred_head=np.zeros((N, R, T), dtype=np.float32),
        visible_to_obs_future=np.ones((N, 16), dtype=bool),
        gt_traj=gt, gt_head=np.zeros((N, T), dtype=np.float32),
        gt_valid=np.ones((N, T), dtype=bool),
        gt_alive_at_current=np.ones(N, dtype=bool),
        gt_radius=np.ones(N, dtype=np.float32),
        gt_is_flying=np.ones((N, T), dtype=bool),  # always flying
    )

    ctx = metric_registry.MetricCtx(map_dir=map_dir, bandwidths={})
    out = {r["metric"]: r for r in map_violation.compute(s, ctx=ctx)}
    # Flying units fail the ground gate -> no valid steps, empty records.
    assert out["map_violation_rate"]["value"] is None
    assert out["map_violation_rate"]["weight"] == 0
