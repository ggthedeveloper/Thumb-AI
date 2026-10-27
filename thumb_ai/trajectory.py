"""
Trajectory models and dataset loaders for training runs.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class TrajectoryPoint:
    """Represents a discrete evaluation snapshot during training."""

    step: int
    epoch: Optional[float] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_metric: Optional[float] = None
    metric_name: str = "val_loss"
    higher_is_better: bool = False
    time_seconds: Optional[float] = None
    compute_cost: Optional[float] = None
    learning_rate: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")
        for k in ("train_loss", "val_loss", "val_metric", "time_seconds", "compute_cost"):
            v = getattr(self, k)
            if v is not None and (math.isnan(v) or math.isinf(v)):
                setattr(self, k, None)


class Trajectory:
    """A series of evaluation checkpoints from an ML/DL/LLM training run."""

    def __init__(self, points: Optional[Sequence[TrajectoryPoint]] = None):
        self._points: List[TrajectoryPoint] = []
        if points:
            for pt in points:
                self.add_point(pt)

    def __len__(self) -> int:
        return len(self._points)

    def __getitem__(self, idx: int) -> TrajectoryPoint:
        return self._points[idx]

    def __iter__(self):
        return iter(self._points)

    @property
    def points(self) -> List[TrajectoryPoint]:
        return list(self._points)

    def add_point(self, point: TrajectoryPoint) -> None:
        if self._points and point.step <= self._points[-1].step:
            raise ValueError(
                f"Trajectory steps must be strictly increasing. "
                f"Last step: {self._points[-1].step}, new step: {point.step}"
            )
        self._points.append(point)

    @classmethod
    def from_records(cls, records: Sequence[Dict[str, Any]]) -> Trajectory:
        traj = cls()
        for r in records:
            step = int(r.get("step", r.get("iteration", len(traj))))
            pt = TrajectoryPoint(
                step=step,
                epoch=float(r["epoch"]) if "epoch" in r and r["epoch"] is not None else None,
                train_loss=float(r["train_loss"]) if "train_loss" in r and r["train_loss"] is not None else None,
                val_loss=float(r["val_loss"]) if "val_loss" in r and r["val_loss"] is not None else None,
                val_metric=float(r["val_metric"]) if "val_metric" in r and r["val_metric"] is not None else None,
                metric_name=str(r.get("metric_name", "val_loss")),
                higher_is_better=bool(r.get("higher_is_better", False)),
                time_seconds=float(r["time_seconds"]) if "time_seconds" in r and r["time_seconds"] is not None else None,
                compute_cost=float(r["compute_cost"]) if "compute_cost" in r and r["compute_cost"] is not None else None,
                learning_rate=float(r["learning_rate"]) if "learning_rate" in r and r["learning_rate"] is not None else None,
            )
            traj.add_point(pt)
        return traj

    @classmethod
    def from_csv(cls, file_path: Union[str, Path]) -> Trajectory:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory CSV file not found: {path}")

        records: List[Dict[str, Any]] = []
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec: Dict[str, Any] = {}
                for k, v in row.items():
                    if v is None or v.strip() == "":
                        rec[k] = None
                    else:
                        try:
                            rec[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                        except ValueError:
                            rec[k] = v
                records.append(rec)
        return cls.from_records(records)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> Trajectory:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory JSON file not found: {path}")

        with open(path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "points" in data:
                data = data["points"]
            if not isinstance(data, list):
                raise ValueError("JSON must contain an array of records or {'points': [...]}")
        return cls.from_records(data)

    def get_steps(self) -> List[int]:
        return [pt.step for pt in self._points]

    def get_values(self, key: str = "val_loss") -> List[Optional[float]]:
        return [getattr(pt, key, None) for pt in self._points]

    def rolling_average(self, key: str = "val_loss", window: int = 3) -> List[Optional[float]]:
        vals = self.get_values(key)
        if window <= 1:
            return vals

        result: List[Optional[float]] = []
        for i in range(len(vals)):
            start = max(0, i - window + 1)
            subset = [v for v in vals[start : i + 1] if v is not None]
            if subset:
                result.append(sum(subset) / len(subset))
            else:
                result.append(None)
        return result

    def best_checkpoint(self, key: str = "val_loss", higher_is_better: bool = False) -> Optional[TrajectoryPoint]:
        valid = [pt for pt in self._points if getattr(pt, key, None) is not None]
        if not valid:
            return None
        if higher_is_better:
            return max(valid, key=lambda pt: getattr(pt, key))
        return min(valid, key=lambda pt: getattr(pt, key))

    def detect_overfitting(self, lookback: int = 5, tolerance_ratio: float = 0.05) -> bool:
        """
        Detects if validation loss is deteriorating while training loss continues decreasing.
        """
        if len(self._points) < lookback + 1:
            return False

        recent = self._points[-lookback:]
        valid_val = [pt.val_loss for pt in recent if pt.val_loss is not None]
        valid_train = [pt.train_loss for pt in recent if pt.train_loss is not None]

        if len(valid_val) < 3 or len(valid_train) < 3:
            return False

        # Train loss is trending down or flat
        train_trend = valid_train[-1] <= valid_train[0] * 1.02
        # Val loss is systematically higher than earlier minimum by tolerance_ratio
        val_min = min(valid_val)
        val_deteriorated = (valid_val[-1] - val_min) / max(abs(val_min), 1e-6) > tolerance_ratio

        return train_trend and val_deteriorated
