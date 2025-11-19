from __future__ import annotations

import dataclasses
import json
import math
import random
from typing import List, Sequence

from .data import ProblemInstance, Task


@dataclasses.dataclass
class Solution:
    instance: ProblemInstance
    order: List[int]

    def copy(self) -> "Solution":
        return Solution(self.instance, list(self.order))

    @property
    def tasks(self) -> List[Task]:
        return [self.instance.tasks[idx] for idx in self.order]

    def cost(self) -> float:
        distance = 0.0
        prev: Task | None = None
        for task in self.tasks:
            distance += self.instance.distance(prev, task)
            prev = task
        distance += self.instance.distance(prev, None)
        penalty = self.capacity_penalty()
        return distance + penalty

    def capacity_penalty(self) -> float:
        load = 0.0
        penalty = 0.0
        for task in self.tasks:
            load += task.demand
            if load > self.instance.vehicle_capacity:
                penalty += (load - self.instance.vehicle_capacity) * 10
        return penalty

    def remove_tasks(self, task_indices: Sequence[int]) -> List[Task]:
        removed: List[Task] = []
        order_set = set(task_indices)
        new_order: List[int] = []
        for idx in self.order:
            if idx in order_set:
                removed.append(self.instance.tasks[idx])
            else:
                new_order.append(idx)
        self.order = new_order
        return removed

    def insert_task(self, task: Task, position: int) -> None:
        self.order.insert(position, task.idx)

    def save(self, path: str) -> None:
        data = {
            "cost": self.cost(),
            "order": self.order,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)


def random_solution(instance: ProblemInstance, seed: int | None = None) -> Solution:
    rng = random.Random(seed)
    order = list(range(len(instance.tasks)))
    rng.shuffle(order)
    return Solution(instance=instance, order=order)
