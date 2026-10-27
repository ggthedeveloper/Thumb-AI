"""
Thumb-AI: Compute-Aware Training Continuation Intelligence for ML, DL & LLM Training.

Maximizes useful model improvement per unit of additional computation via
Training Continuation Value (TCV).
"""

__version__ = "0.1.0"
__author__ = "Gaurav Gautam <ggraipurchor@gmail.com>"

from .trajectory import Trajectory, TrajectoryPoint
from .tcv import TCVCalculator, ComputeProfile, TCVEstimate
from .recommendation import RecommendationEngine, Decision, Recommendation

__all__ = [
    "Trajectory",
    "TrajectoryPoint",
    "TCVCalculator",
    "ComputeProfile",
    "TCVEstimate",
    "RecommendationEngine",
    "Decision",
    "Recommendation",
]
