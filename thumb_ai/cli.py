"""
Command-line interface for Thumb-AI trajectory evaluation and TCV reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .recommendation import Decision, RecommendationEngine
from .tcv import ComputeProfile, TCVCalculator
from .trajectory import Trajectory, TrajectoryPoint


def _create_synthetic_trajectory(scenario: str = "converging") -> Trajectory:
    """Generate a realistic training trajectory for testing and demonstration."""
    import math

    traj = Trajectory()
    steps = list(range(100, 2100, 100))

    if scenario == "converging":
        for s in steps:
            # Power law decay with slight noise
            train = 2.5 * (s / 100.0) ** (-0.35)
            val = train * 1.08 + 0.02
            traj.add_point(
                TrajectoryPoint(
                    step=s,
                    train_loss=round(train, 4),
                    val_loss=round(val, 4),
                    time_seconds=round(s * 0.4, 2),
                    learning_rate=3e-4,
                )
            )
    elif scenario == "overfitting":
        for s in steps:
            train = 2.5 * (s / 100.0) ** (-0.4)
            val = 2.5 * (s / 100.0) ** (-0.4) if s <= 800 else 1.2 + (s - 800) * 0.0006
            traj.add_point(
                TrajectoryPoint(
                    step=s,
                    train_loss=round(train, 4),
                    val_loss=round(val, 4),
                    time_seconds=round(s * 0.4, 2),
                    learning_rate=1e-4,
                )
            )
    else:  # plateau
        for s in steps:
            train = 0.45 + 0.001 * math.sin(s)
            val = 0.48 + 0.001 * math.cos(s)
            traj.add_point(
                TrajectoryPoint(
                    step=s,
                    train_loss=round(train, 4),
                    val_loss=round(val, 4),
                    time_seconds=round(s * 0.4, 2),
                    learning_rate=5e-4,
                )
            )
    return traj


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="thumb-ai",
        description="Compute-Aware Training Continuation Intelligence for ML/DL/LLM models.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze training curve from CSV or JSON")
    analyze_parser.add_argument("file", type=Path, help="Path to training logs (CSV or JSON)")
    analyze_parser.add_argument("--metric", default="val_loss", help="Metric key to evaluate (default: val_loss)")
    analyze_parser.add_argument("--horizon", type=int, default=500, help="Future steps horizon (default: 500)")
    analyze_parser.add_argument("--higher-is-better", action="store_true", help="Set if metric improves when increasing")
    analyze_parser.add_argument("--gpu-cost", type=float, default=2.50, help="Dollar cost per GPU hour")
    analyze_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run simulation on synthetic training curves")
    sim_parser.add_argument(
        "--scenario",
        choices=["converging", "overfitting", "plateau"],
        default="converging",
        help="Scenario to simulate",
    )
    sim_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "simulate":
        traj = _create_synthetic_trajectory(args.scenario)
        metric_key = "val_loss"
        higher_is_better = False
        horizon = 500
        gpu_cost = 2.50
    else:
        path = args.file
        if not path.is_file():
            print(f"Error: File not found at {path}", file=sys.stderr)
            return 1
        traj = Trajectory.from_csv(path) if path.suffix.lower() == ".csv" else Trajectory.from_json(path)
        metric_key = args.metric
        higher_is_better = args.higher_is_better
        horizon = args.horizon
        gpu_cost = args.gpu_cost

    profile = ComputeProfile(cost_per_gpu_hour=gpu_cost)
    calc = TCVCalculator(compute_profile=profile)
    engine = RecommendationEngine(calculator=calc)

    rec = engine.evaluate(
        trajectory=traj,
        horizon_steps=horizon,
        metric_key=metric_key,
        higher_is_better=higher_is_better,
    )

    if args.json:
        payload = {
            "action": rec.action.value,
            "headline": rec.headline,
            "confidence": round(rec.confidence, 4),
            "suggested_action": rec.suggested_action,
            "best_checkpoint_step": rec.best_checkpoint_step,
            "reasons": rec.reasons,
            "tcv": {
                "raw_tcv": round(rec.tcv.raw_tcv, 4),
                "normalized_tcv": round(rec.tcv.normalized_tcv, 4),
                "expected_delta_performance": round(rec.tcv.expected_delta_performance, 6),
                "expected_delta_compute_cost": round(rec.tcv.expected_delta_compute_cost, 4),
                "current_performance": round(rec.tcv.current_performance, 6),
                "projected_performance": round(rec.tcv.projected_performance, 6),
                "fit_r2": round(rec.tcv.fit_r2, 4),
            },
        }
        print(json.dumps(payload, indent=2))
        return 0

    # Human-readable formatted output
    action_colors = {
        Decision.CONTINUE: "\033[92m",  # Green
        Decision.CHECKPOINT_NOW: "\033[93m",  # Yellow
        Decision.ADJUST_LR: "\033[94m",  # Blue
        Decision.STOP: "\033[91m",  # Red
    }
    reset = "\033[0m"
    bold = "\033[1m"
    color = action_colors.get(rec.action, "")

    print(f"\n{bold}======================================================================{reset}")
    print(f"{bold}🧠 THUMB-AI — TRAINING CONTINUATION REPORT{reset}")
    print(f"{bold}======================================================================{reset}")
    print(f"Evaluated Checkpoints : {len(traj)} points (Steps {traj[0].step} -> {traj[-1].step})")
    print(f"Target Metric         : {metric_key} (higher is better: {higher_is_better})")
    print(f"Current Performance   : {rec.tcv.current_performance:.4f}")
    print(f"Projected Performance : {rec.tcv.projected_performance:.4f} (at +{rec.tcv.horizon_steps} steps)")
    print(f"Expected Gain         : {rec.tcv.expected_delta_performance:.4f}")
    print(f"Expected Compute Cost : ${rec.tcv.expected_delta_compute_cost:.2f} USD")
    print(f"Curve Fit Confidence : R² = {rec.tcv.fit_r2:.2f}")
    print(f"Normalized TCV Score  : {rec.tcv.normalized_tcv:.2f} / 1.00")
    print("----------------------------------------------------------------------")
    print(f"DECISION              : {bold}{color}[ {rec.action.value} ]{reset} (Confidence: {rec.confidence * 100:.1f}%)")
    print(f"SUMMARY               : {rec.headline}")
    print(f"RECOMMENDED ACTION    : {rec.suggested_action}")
    if rec.best_checkpoint_step:
        print(f"BEST CHECKPOINT       : Step {rec.best_checkpoint_step}")
    if rec.reasons:
        print("\nAnalysis Rationale:")
        for r in rec.reasons:
            print(f"  • {r}")
    print(f"{bold}======================================================================{reset}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
