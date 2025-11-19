from __future__ import annotations

import json
import math
import pathlib
from typing import List, Sequence

from .data import ProblemInstance, Task


class DRLAgent:
    """Lightweight linear agent implemented without external dependencies."""

    def __init__(self, weights: List[float] | None = None) -> None:
        self.weights = weights if weights is not None else [0.0] * 5

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump({"weights": self.weights}, fh)

    @classmethod
    def load(cls, path: pathlib.Path) -> "DRLAgent":
        if not path.exists():
            raise FileNotFoundError(f"DRL model not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        weights = [float(x) for x in data["weights"]]
        return cls(weights)

    def features(self, instance: ProblemInstance, task: Task, position: int, total: int) -> List[float]:
        depot_dist = instance.distance(None, task)
        norm_pos = position / max(total - 1, 1)
        demand_ratio = task.demand / instance.vehicle_capacity
        angle = math.atan2(task.y - instance.depot[1], task.x - instance.depot[0]) / math.pi
        bias = 1.0
        return [depot_dist / 100, norm_pos, demand_ratio, angle, bias]

    def score_tasks(self, instance: ProblemInstance, order: Sequence[int]) -> List[float]:
        scores: List[float] = []
        total = len(order)
        for pos, idx in enumerate(order):
            features = self.features(instance, instance.tasks[idx], pos, total)
            score = sum(f * w for f, w in zip(features, self.weights))
            scores.append(score)
        return scores

    def select_tasks(self, instance: ProblemInstance, order: Sequence[int], num_remove: int) -> List[int]:
        if not order:
            return []
        num_remove = min(num_remove, len(order))
        scores = self.score_tasks(instance, order)
        indexed = sorted(zip(order, scores), key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in indexed[:num_remove]]
