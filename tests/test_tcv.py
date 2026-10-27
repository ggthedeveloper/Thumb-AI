import pytest
from thumb_ai.tcv import ComputeProfile, TCVCalculator
from thumb_ai.trajectory import Trajectory, TrajectoryPoint


def test_tcv_calculation_basic():
    traj = Trajectory()
    for s, l in [(100, 2.5), (200, 1.9), (300, 1.5), (400, 1.2), (500, 1.0)]:
        traj.add_point(TrajectoryPoint(step=s, val_loss=l, time_seconds=s * 0.5))

    calc = TCVCalculator(ComputeProfile(cost_per_gpu_hour=3.00, num_gpus=1))
    estimate = calc.calculate_tcv(traj, horizon_steps=200)

    assert estimate.horizon_steps == 200
    assert estimate.expected_delta_compute_cost > 0.0
    assert estimate.current_performance == 1.0
    assert estimate.fit_r2 >= 0.8
    assert 0.0 <= estimate.normalized_tcv <= 1.0


def test_tcv_requires_minimum_points():
    traj = Trajectory([TrajectoryPoint(step=10, val_loss=2.0)])
    calc = TCVCalculator()
    with pytest.raises(ValueError, match="at least 3 points"):
        calc.calculate_tcv(traj)


def test_tcv_cost_estimation_from_explicit_seconds():
    traj = Trajectory()
    for s in [100, 200, 300]:
        traj.add_point(TrajectoryPoint(step=s, val_loss=1.0))

    profile = ComputeProfile(cost_per_gpu_hour=3.60, seconds_per_step=1.0, num_gpus=1)
    calc = TCVCalculator(profile)
    est = calc.calculate_tcv(traj, horizon_steps=100)
    # 100 steps * 1 sec = 100 sec = (100 / 3600) * 3.60 = $0.10
    assert pytest.approx(est.expected_delta_compute_cost, rel=1e-3) == 0.10
