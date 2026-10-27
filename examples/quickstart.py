#!/usr/bin/env python3
"""
Quickstart demonstration of Thumb-AI Training Continuation Value (TCV) analysis.
"""

from pathlib import Path
from thumb_ai import ComputeProfile, RecommendationEngine, TCVCalculator, Trajectory


def main():
    csv_path = Path(__file__).parent / "sample_training.csv"
    print(f"Loading trajectory from: {csv_path}")
    traj = Trajectory.from_csv(csv_path)

    # Configure hardware profile (e.g. 1x A100 @ $2.50/hr)
    profile = ComputeProfile(
        hardware_name="NVIDIA A100-SXM4-80GB",
        cost_per_gpu_hour=2.50,
        num_gpus=1,
    )
    calc = TCVCalculator(profile)
    engine = RecommendationEngine(calculator=calc)

    rec = engine.evaluate(traj, horizon_steps=500, metric_key="val_loss")

    print("\n--- Evaluation Results ---")
    print(f"Action:              {rec.action.value}")
    print(f"Confidence:          {rec.confidence * 100:.1f}%")
    print(f"Current Val Loss:    {rec.tcv.current_performance:.4f}")
    print(f"Projected Val Loss:  {rec.tcv.projected_performance:.4f}")
    print(f"Expected Gain:       {rec.tcv.expected_delta_performance:.4f}")
    print(f"Expected Cost:       ${rec.tcv.expected_delta_compute_cost:.2f} USD")
    print(f"Normalized TCV:      {rec.tcv.normalized_tcv:.2f}")
    print(f"Headline:            {rec.headline}")
    print(f"Suggested Action:    {rec.suggested_action}\n")


if __name__ == "__main__":
    main()
