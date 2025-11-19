from __future__ import annotations

import dataclasses
import math
import random
from typing import List

from .config import ProblemConfig


@dataclasses.dataclass
class Task:
    idx: int
    x: float
    y: float
    demand: float
    service_time: float

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y


@dataclasses.dataclass
class ProblemInstance:
    tasks: List[Task]
    depot: tuple[float, float] = (0.0, 0.0)
    vehicle_capacity: float = 100.0

    def distance(self, a: Task | None, b: Task | None) -> float:
        ax, ay = self.depot if a is None else a.position
        bx, by = self.depot if b is None else b.position
        return math.hypot(ax - bx, ay - by)


def generate_random_instance(config: ProblemConfig) -> ProblemInstance:
    rng = random.Random(config.seed)
    tasks: List[Task] = []
    for idx in range(config.num_tasks):
        if config.distribution == "clustered":
            cluster = rng.randint(0, 2)
            base_x = 20 + cluster * 30 + rng.random() * 10
            base_y = 20 + cluster * 30 + rng.random() * 10
            x = base_x + rng.uniform(-5, 5)
            y = base_y + rng.uniform(-5, 5)
        else:
            x = rng.uniform(0, 100)
            y = rng.uniform(0, 100)
        demand = rng.uniform(1, 10)
        service_time = rng.uniform(5, 15)
        tasks.append(Task(idx=idx, x=x, y=y, demand=demand, service_time=service_time))
    return ProblemInstance(tasks=tasks)
