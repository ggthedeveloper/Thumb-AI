from thumb_ai.recommendation import Decision, RecommendationEngine
from thumb_ai.tcv import ComputeProfile, TCVCalculator
from thumb_ai.trajectory import Trajectory, TrajectoryPoint


def test_recommendation_warmup():
    traj = Trajectory([
        TrajectoryPoint(step=10, val_loss=3.0),
        TrajectoryPoint(step=20, val_loss=2.8),
    ])
    engine = RecommendationEngine()
    rec = engine.evaluate(traj)
    assert rec.action == Decision.CONTINUE
    assert "warmup" in rec.headline.lower()


def test_recommendation_converging_steep():
    traj = Trajectory()
    for s, l in [(100, 3.0), (200, 2.2), (300, 1.6), (400, 1.2), (500, 0.9)]:
        traj.add_point(TrajectoryPoint(step=s, val_loss=l, time_seconds=s * 0.1))

    engine = RecommendationEngine()
    rec = engine.evaluate(traj, horizon_steps=200)
    assert rec.action == Decision.CONTINUE
    assert rec.tcv.expected_delta_performance > 0.0


def test_recommendation_overfitting():
    traj = Trajectory()
    points = [
        (100, 2.0, 2.0),
        (200, 1.6, 1.6),
        (300, 1.3, 1.3),
        (400, 1.0, 1.5),
        (500, 0.8, 1.8),
        (600, 0.6, 2.2),
    ]
    for s, tr, vl in points:
        traj.add_point(TrajectoryPoint(step=s, train_loss=tr, val_loss=vl))

    engine = RecommendationEngine()
    rec = engine.evaluate(traj)
    assert rec.action == Decision.CHECKPOINT_NOW
    assert rec.best_checkpoint_step == 300


def test_recommendation_budget_cap():
    traj = Trajectory()
    for s, l in [(100, 1.5), (200, 1.4), (300, 1.3)]:
        traj.add_point(TrajectoryPoint(step=s, val_loss=l, time_seconds=s * 100))

    # Low budget that will be exceeded
    profile = ComputeProfile(cost_per_gpu_hour=100.0, max_budget_usd=1.0)
    calc = TCVCalculator(profile)
    engine = RecommendationEngine(calculator=calc)
    rec = engine.evaluate(traj)
    assert rec.action == Decision.STOP
    assert "budget" in rec.headline.lower()
