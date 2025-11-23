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


class WorstSlackRemovalStrategy(RemovalStrategy):
    def __init__(self, instance: ProblemInstance) -> None:
        super().__init__(instance)

    def _slacks(self, solution_order: List[int]) -> List[tuple[int, float]]:
        time_now = 0.0
        slacks: List[tuple[int, float]] = []
        prev_idx: int | None = None
        for idx in solution_order:
            task = self.instance.tasks[idx]
            travel = self.instance.distance_idx(prev_idx, idx)
            arrive = time_now + travel
            slack = task.due_time - arrive
            slacks.append((idx, slack))
            time_now = arrive + task.service_time
            prev_idx = idx
        return slacks

    def select(self, solution_order: List[int], num_remove: int) -> List[int]:
        if not solution_order:
            return []
        num_remove = min(num_remove, len(solution_order))
        slacks = self._slacks(solution_order)
        worst = sorted(slacks, key=lambda x: x[1])[:num_remove]
        return [idx for idx, _ in worst]


class MaxDistanceRemovalStrategy(RemovalStrategy):
    def __init__(self, instance: ProblemInstance, seed: int | None = None) -> None:
        super().__init__(instance)
        self.rng = random.Random(seed)

    def select(self, solution_order: List[int], num_remove: int) -> List[int]:
        if not solution_order:
            return []
        num_remove = min(num_remove, len(solution_order))
        distances = [
            (idx, self.instance.distance(None, self.instance.tasks[idx])) for idx in solution_order
        ]
        distances.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in distances[:num_remove]]


class DRLRemovalStrategy(RemovalStrategy):
    def __init__(self, instance: ProblemInstance, agent: DRLAgent) -> None:
        super().__init__(instance)
        self.agent = agent

    def select(self, solution_order: List[int], num_remove: int) -> List[int]:
        return self.agent.select_tasks(self.instance, solution_order, num_remove)


class HybridDRLRemovalStrategy(RemovalStrategy):
    def __init__(self, instance: ProblemInstance, agent: DRLAgent, alpha: float = 1.0) -> None:
        super().__init__(instance)
        self.agent = agent
        self.shaw = ShawRemovalStrategy(instance, alpha=alpha)

    def select(self, solution_order: List[int], num_remove: int) -> List[int]:
        if not solution_order:
            return []
        num_remove = min(num_remove, len(solution_order))
        primary = max(1, math.ceil(num_remove * 0.7))
        drl_seed = set(self.agent.select_tasks(self.instance, solution_order, primary))
        remaining = [idx for idx in solution_order if idx not in drl_seed]
        selected = list(drl_seed)
        while len(selected) < num_remove and remaining:
            best_idx = None
            best_score = float("inf")
            for candidate in remaining:
                score = min(
                    self.shaw.similarity(self.instance.tasks[candidate], self.instance.tasks[s])
                    for s in selected
                )
                if score < best_score:
                    best_score = score
                    best_idx = candidate
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
        while len(selected) < num_remove and remaining:
            selected.append(remaining.pop())
        return selected


def build_removal_strategy(name: str, instance: ProblemInstance, **kwargs) -> RemovalStrategy:
    name = name.lower()
    if name == "random":
        return RandomRemovalStrategy(instance, seed=kwargs.get("seed"))
    if name == "shaw":
        return ShawRemovalStrategy(instance)
    if name == "worst_slack":
        return WorstSlackRemovalStrategy(instance)
    if name == "max_distance":
        return MaxDistanceRemovalStrategy(instance, seed=kwargs.get("seed"))
    if name == "drl":
        agent: DRLAgent = kwargs["agent"]
        return DRLRemovalStrategy(instance, agent)
    if name == "drl_hybrid":
        agent = kwargs["agent"]
        return HybridDRLRemovalStrategy(instance, agent)
    raise ValueError(f"Unknown removal strategy: {name}")
