"""
Training Continuation Value (TCV) mathematical formulation and trajectory fitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .trajectory import Trajectory


@dataclass
class ComputeProfile:
    """Hardware and cost configuration for the training environment."""

    hardware_name: str = "NVIDIA A100-SXM4-80GB"
    cost_per_gpu_hour: float = 2.50  # USD per GPU hour
    num_gpus: int = 1
    seconds_per_step: Optional[float] = None
    power_watts: float = 400.0  # Watts per GPU
    carbon_intensity_g_per_kwh: float = 380.0  # gCO2eq/kWh global avg
    max_budget_usd: Optional[float] = None


@dataclass
class TCVEstimate:
    """Estimated marginal performance gain vs compute cost for a specified horizon."""

    horizon_steps: int
    expected_delta_performance: float
    expected_delta_compute_cost: float
    raw_tcv: float
    normalized_tcv: float  # Scale [0.0, 1.0]
    fit_r2: float
    current_performance: float
    projected_performance: float
    currency: str = "USD"


class TCVCalculator:
    """
    Computes Training Continuation Value (TCV) by extrapolating learning curves
    and balancing expected gain against computation cost.
    """

    def __init__(self, compute_profile: Optional[ComputeProfile] = None):
        self.profile = compute_profile or ComputeProfile()

    def _estimate_cost_per_step(self, trajectory: Trajectory) -> float:
        """Derive dollar cost per training step."""
        if self.profile.seconds_per_step is not None and self.profile.seconds_per_step > 0:
            step_duration_hours = self.profile.seconds_per_step / 3600.0
            return step_duration_hours * self.profile.cost_per_gpu_hour * self.profile.num_gpus

        # Attempt to compute from trajectory timestamps
        valid_times = [
            (pt.step, pt.time_seconds)
            for pt in trajectory.points
            if pt.time_seconds is not None
        ]
        if len(valid_times) >= 2:
            d_time = valid_times[-1][1] - valid_times[0][1]
            d_step = valid_times[-1][0] - valid_times[0][0]
            if d_step > 0 and d_time > 0:
                sec_per_step = d_time / d_step
                return (sec_per_step / 3600.0) * self.profile.cost_per_gpu_hour * self.profile.num_gpus

        # Default fallback: 0.5s per step
        return (0.5 / 3600.0) * self.profile.cost_per_gpu_hour * self.profile.num_gpus

    def _fit_learning_curve(
        self, steps: np.ndarray, values: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Fits asymptotic power law L(t) = a * t^(-b) via log-log regression.
        Returns: (a, b, r2)
        """
        # Ensure positive steps and positive values
        safe_steps = np.maximum(steps, 1.0)
        safe_vals = np.maximum(values, 1e-7)

        log_x = np.log(safe_steps)
        log_y = np.log(safe_vals)

        # Standard linear least squares on log-log
        A = np.vstack([log_x, np.ones(len(log_x))]).T
        slope, intercept = np.linalg.lstsq(A, log_y, rcond=None)[0]

        pred_log_y = slope * log_x + intercept
        ss_res = np.sum((log_y - pred_log_y) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-9)) if ss_tot > 0 else 0.0
        r2 = float(np.clip(r2, 0.0, 1.0))

        b = -slope  # decay exponent
        a = float(np.exp(intercept))
        return a, b, r2

    def calculate_tcv(
        self,
        trajectory: Trajectory,
        horizon_steps: int = 500,
        metric_key: str = "val_loss",
        higher_is_better: bool = False,
    ) -> TCVEstimate:
        if len(trajectory) < 3:
            raise ValueError("Trajectory must have at least 3 points to calculate TCV")

        steps = []
        vals = []
        for pt in trajectory.points:
            v = getattr(pt, metric_key, None)
            if v is not None and not math.isnan(v):
                steps.append(pt.step)
                vals.append(v)

        if len(steps) < 3:
            raise ValueError(f"Not enough valid data points for metric '{metric_key}'")

        arr_steps = np.array(steps, dtype=float)
        arr_vals = np.array(vals, dtype=float)

        curr_step = arr_steps[-1]
        curr_val = arr_vals[-1]
        target_step = curr_step + max(1, horizon_steps)

        # Fit curve
        a, b, r2 = self._fit_learning_curve(arr_steps, arr_vals)

        if b > 0:
            # Proper decay
            projected_val = float(a * (target_step ** (-b)))
        else:
            # Flat or diverging loss: extrapolate recent slope
            lookback = min(len(arr_steps), 5)
            dx = arr_steps[-1] - arr_steps[-lookback]
            dy = arr_vals[-1] - arr_vals[-lookback]
            slope = dy / max(dx, 1.0)
            projected_val = float(curr_val + slope * horizon_steps)

        if not higher_is_better:
            # Lower is better (e.g. loss): improvement = curr_val - projected_val
            delta_perf = max(0.0, curr_val - projected_val)
        else:
            delta_perf = max(0.0, projected_val - curr_val)

        cost_per_step = self._estimate_cost_per_step(trajectory)
        delta_cost = float(cost_per_step * horizon_steps)

        # Raw TCV = Expected Delta Performance / Expected Delta Compute Cost
        raw_tcv = delta_perf / max(delta_cost, 1e-6)

        # Normalized TCV sigmoid mapping to [0.0, 1.0]
        # Benchmark scaling: an improvement of 0.05 loss per $1.00 compute is high (~0.85)
        k = 20.0
        normalized_tcv = float(1.0 / (1.0 + math.exp(-k * raw_tcv + 2.0)))

        return TCVEstimate(
            horizon_steps=horizon_steps,
            expected_delta_performance=float(delta_perf),
            expected_delta_compute_cost=float(delta_cost),
            raw_tcv=float(raw_tcv),
            normalized_tcv=normalized_tcv,
            fit_r2=r2,
            current_performance=float(curr_val),
            projected_performance=float(projected_val),
        )
