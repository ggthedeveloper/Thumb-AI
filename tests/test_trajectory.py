import tempfile
import pytest
from thumb_ai.trajectory import Trajectory, TrajectoryPoint


def test_trajectory_point_creation():
    pt = TrajectoryPoint(step=100, train_loss=1.5, val_loss=1.6)
    assert pt.step == 100
    assert pt.train_loss == 1.5
    assert pt.val_loss == 1.6
    assert pt.metric_name == "val_loss"


def test_trajectory_point_rejects_negative_step():
    with pytest.raises(ValueError, match="non-negative"):
        TrajectoryPoint(step=-5)


def test_trajectory_point_cleans_nan():
    pt = TrajectoryPoint(step=10, val_loss=float("nan"))
    assert pt.val_loss is None


def test_trajectory_monotonic_steps():
    traj = Trajectory()
    traj.add_point(TrajectoryPoint(step=10, val_loss=2.0))
    traj.add_point(TrajectoryPoint(step=20, val_loss=1.8))
    with pytest.raises(ValueError, match="strictly increasing"):
        traj.add_point(TrajectoryPoint(step=15, val_loss=1.9))


def test_trajectory_from_csv():
    csv_content = """step,train_loss,val_loss,learning_rate
100,2.5,2.6,0.001
200,2.1,2.2,0.001
300,1.8,1.9,0.0005
"""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f_path = f.name

    traj = Trajectory.from_csv(f_path)
    assert len(traj) == 3
    assert traj[0].step == 100
    assert traj[1].val_loss == 2.2
    assert traj[2].learning_rate == 0.0005


def test_trajectory_from_json():
    json_content = """[
        {"step": 10, "val_loss": 3.0},
        {"step": 20, "val_loss": 2.5},
        {"step": 30, "val_loss": 2.1}
    ]"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(json_content)
        f_path = f.name

    traj = Trajectory.from_json(f_path)
    assert len(traj) == 3
    assert traj[2].val_loss == 2.1


def test_rolling_average():
    traj = Trajectory()
    for s, l in [(1, 10.0), (2, 8.0), (3, 6.0), (4, 4.0)]:
        traj.add_point(TrajectoryPoint(step=s, val_loss=l))

    roll = traj.rolling_average("val_loss", window=2)
    assert roll[0] == 10.0
    assert roll[1] == 9.0  # (10 + 8) / 2
    assert roll[2] == 7.0  # (8 + 6) / 2
    assert roll[3] == 5.0  # (6 + 4) / 2


def test_best_checkpoint():
    traj = Trajectory()
    for s, l in [(1, 2.0), (2, 1.2), (3, 1.5)]:
        traj.add_point(TrajectoryPoint(step=s, val_loss=l))

    best = traj.best_checkpoint(key="val_loss", higher_is_better=False)
    assert best is not None
    assert best.step == 2
    assert best.val_loss == 1.2


def test_detect_overfitting():
    traj = Trajectory()
    # Training drops, validation diverges
    points = [
        (100, 2.0, 2.1),
        (200, 1.6, 1.7),
        (300, 1.3, 1.4),
        (400, 1.0, 1.5),
        (500, 0.8, 1.8),
        (600, 0.6, 2.2),
    ]
    for s, tr, vl in points:
        traj.add_point(TrajectoryPoint(step=s, train_loss=tr, val_loss=vl))

    assert traj.detect_overfitting(lookback=5) is True
