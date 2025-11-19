from __future__ import annotations

import abc
import math
import random
from typing import List

from .data import ProblemInstance, Task
from .drl_agent import DRLAgent


class RemovalStrategy(abc.ABC):
    def __init__(self, instance: ProblemInstance) -> None:
        self.instance = instance

    @abc.abstractmethod
    def select(self, solution_order: List[int], num_remove: int) -> List[int]:
        raise NotImplementedError


class RandomRemovalStrategy(RemovalStrategy):
    def __init__(self, instance: ProblemInstance, seed: int | None = None) -> None:
        super().__init__(instance)
        self.rng = random.Random(seed)

    def select(self, solution_order: List[int], num_remove: int) -> List[int]:
        num_remove = min(num_remove, len(solution_order))
        return self.rng.sample(solution_order, num_remove)


class ShawRemovalStrategy(RemovalStrategy):
    def __init__(self, instance: ProblemInstance, alpha: float = 1.0) -> None:
        super().__init__(instance)
        self.alpha = alpha
        self.distance_cache = self._precompute_distance()

    def _precompute_distance(self) -> List[List[float]]:
        matrix: List[List[float]] = []
        for a in self.instance.tasks:
            row: List[float] = []
            for b in self.instance.tasks:
                row.append(self.instance.distance(a, b))
            matrix.append(row)
        return matrix

    def similarity(self, a: Task, b: Task) -> float:
        dist = self.distance_cache[a.idx][b.idx]
        demand_diff = abs(a.demand - b.demand)
        return dist + self.alpha * demand_diff

    def select(self, solution_order: List[int], num_remove: int) -> List[int]:
        if not solution_order:
            return []
        num_remove = min(num_remove, len(solution_order))
        removed = []
        remaining = set(solution_order)
        seed_task_idx = random.choice(solution_order)
        removed.append(seed_task_idx)
        remaining.remove(seed_task_idx)
        while len(removed) < num_remove and remaining:
            best_task = min(
                remaining,
                key=lambda idx: self.similarity(self.instance.tasks[seed_task_idx], self.instance.tasks[idx]),
            )
            removed.append(best_task)
            remaining.remove(best_task)
        return removed


class DRLRemovalStrategy(RemovalStrategy):
    def __init__(self, instance: ProblemInstance, agent: DRLAgent) -> None:
        super().__init__(instance)
        self.agent = agent

    def select(self, solution_order: List[int], num_remove: int) -> List[int]:
        return self.agent.select_tasks(self.instance, solution_order, num_remove)


def build_removal_strategy(name: str, instance: ProblemInstance, **kwargs) -> RemovalStrategy:
    name = name.lower()
    if name == "random":
        return RandomRemovalStrategy(instance, seed=kwargs.get("seed"))
    if name == "shaw":
        return ShawRemovalStrategy(instance)
    if name == "drl":
        agent: DRLAgent = kwargs["agent"]
        return DRLRemovalStrategy(instance, agent)
    raise ValueError(f"Unknown removal strategy: {name}")
