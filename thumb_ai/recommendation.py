"""
Actionable recommendation engine based on TCV and trajectory dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .tcv import ComputeProfile, TCVCalculator, TCVEstimate
from .trajectory import Trajectory


class Decision(str, Enum):
    CONTINUE = "CONTINUE"
    CHECKPOINT_NOW = "CHECKPOINT_NOW"
    ADJUST_LR = "ADJUST_LR"
    STOP = "STOP"


@dataclass
class Recommendation:
    """Actionable advice for the machine learning practitioner or automated scheduler."""

    action: Decision
    tcv: TCVEstimate
    confidence: float  # [0.0, 1.0]
    headline: str
    reasons: List[str] = field(default_factory=list)
    suggested_action: str = ""
    best_checkpoint_step: Optional[int] = None


class RecommendationEngine:
    """
    Evaluates trajectory metrics, overfitting signals, and compute budgets
    to determine whether training should proceed.
    """

    def __init__(
        self,
        calculator: Optional[TCVCalculator] = None,
        tcv_continue_threshold: float = 0.20,
        tcv_stop_threshold: float = 0.08,
    ):
        self.calculator = calculator or TCVCalculator()
        self.tcv_continue_threshold = tcv_continue_threshold
        self.tcv_stop_threshold = tcv_stop_threshold

    def evaluate(
        self,
        trajectory: Trajectory,
        horizon_steps: int = 500,
        metric_key: str = "val_loss",
        higher_is_better: bool = False,
    ) -> Recommendation:
        if len(trajectory) < 3:
            return Recommendation(
                action=Decision.CONTINUE,
                tcv=TCVEstimate(
                    horizon_steps=horizon_steps,
                    expected_delta_performance=0.0,
                    expected_delta_compute_cost=0.0,
                    raw_tcv=0.0,
                    normalized_tcv=0.5,
                    fit_r2=0.0,
                    current_performance=0.0,
                    projected_performance=0.0,
                ),
                confidence=0.3,
                headline="Insufficient warmup data — continue gathering steps",
                reasons=["Need at least 3 evaluation checkpoints to compute reliable TCV."],
                suggested_action="Allow training to proceed through initial warmup.",
            )

        tcv = self.calculator.calculate_tcv(
            trajectory=trajectory,
            horizon_steps=horizon_steps,
            metric_key=metric_key,
            higher_is_better=higher_is_better,
        )

        reasons: List[str] = []
        best_pt = trajectory.best_checkpoint(key=metric_key, higher_is_better=higher_is_better)
        best_step = best_pt.step if best_pt else None

        # 1. Budget check
        profile: ComputeProfile = self.calculator.profile
        if profile.max_budget_usd is not None:
            # Estimate total spent so far
            cost_per_step = self.calculator._estimate_cost_per_step(trajectory)
            spent = cost_per_step * (trajectory[-1].step - trajectory[0].step)
            if spent + tcv.expected_delta_compute_cost > profile.max_budget_usd:
                return Recommendation(
                    action=Decision.STOP,
                    tcv=tcv,
                    confidence=0.95,
                    headline="Compute budget ceiling reached",
                    reasons=[
                        f"Projected cost exceeds maximum budget cap of ${profile.max_budget_usd:.2f}.",
                        f"Current spent: ~${spent:.2f}, next horizon cost: ~${tcv.expected_delta_compute_cost:.2f}.",
                    ],
                    suggested_action="Halt training and deploy the best checkpoint.",
                    best_checkpoint_step=best_step,
                )

        # 2. Overfitting check
        if trajectory.detect_overfitting(lookback=5):
            return Recommendation(
                action=Decision.CHECKPOINT_NOW,
                tcv=tcv,
                confidence=0.88,
                headline="Overfitting detected — validation metric diverging",
                reasons=[
                    "Validation loss is increasing while training loss continues falling.",
                    f"Best checkpoint observed at step {best_step}.",
                ],
                suggested_action="Save model weights immediately and halt further training or increase regularization.",
                best_checkpoint_step=best_step,
            )

        # 3. Learning Rate Adjustment check
        last_pt = trajectory[-1]
        recent_values = trajectory.get_values(metric_key)[-5:]
        flat_curve = False
        if len(recent_values) >= 3 and all(v is not None for v in recent_values):
            variance = max(recent_values) - min(recent_values)
            if variance < 0.005 * abs(recent_values[-1]):
                flat_curve = True

        if flat_curve and last_pt.learning_rate is not None and last_pt.learning_rate > 1e-4:
            return Recommendation(
                action=Decision.ADJUST_LR,
                tcv=tcv,
                confidence=0.82,
                headline="Plateau reached with high learning rate",
                reasons=[
                    "Loss improvements have stagnated over the last evaluation window.",
                    f"Current learning rate ({last_pt.learning_rate:.2e}) may be oscillating around minimum.",
                ],
                suggested_action="Reduce learning rate by 2x-5x or trigger cosine annealing step.",
                best_checkpoint_step=best_step,
            )

        # 4. TCV threshold decisions
        if tcv.normalized_tcv >= self.tcv_continue_threshold:
            reasons.append(
                f"Expected improvement ({tcv.expected_delta_performance:.4f}) justifies compute cost (${tcv.expected_delta_compute_cost:.2f})."
            )
            reasons.append(f"Curve fit confidence (R² = {tcv.fit_r2:.2f}).")
            return Recommendation(
                action=Decision.CONTINUE,
                tcv=tcv,
                confidence=min(0.95, max(0.5, tcv.fit_r2)),
                headline="Strong continuation value — continue training",
                reasons=reasons,
                suggested_action=f"Continue training for the next {horizon_steps} steps.",
                best_checkpoint_step=best_step,
            )
        elif tcv.normalized_tcv <= self.tcv_stop_threshold:
            reasons.append(
                f"Marginal gain ({tcv.expected_delta_performance:.4f}) has dropped below economic threshold."
            )
            reasons.append(f"Spending ${tcv.expected_delta_compute_cost:.2f} for negligible return.")
            return Recommendation(
                action=Decision.STOP,
                tcv=tcv,
                confidence=0.85,
                headline="Diminishing returns — optimal stop point reached",
                reasons=reasons,
                suggested_action=f"Halt training to conserve compute; use checkpoint at step {best_step}.",
                best_checkpoint_step=best_step,
            )
        else:
            return Recommendation(
                action=Decision.CONTINUE,
                tcv=tcv,
                confidence=0.65,
                headline="Moderate continuation value — monitor closely",
                reasons=[
                    f"TCV score ({tcv.normalized_tcv:.2f}) is in the transition zone.",
                    "Monitor next checkpoint for divergence or plateau.",
                ],
                suggested_action="Continue with caution; re-evaluate at next evaluation interval.",
                best_checkpoint_step=best_step,
            )
