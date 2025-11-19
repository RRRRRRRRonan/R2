from __future__ import annotations

from typing import List

from .data import Task
from .solution import Solution


def greedy_repair(solution: Solution, removed: List[Task]) -> None:
    instance = solution.instance
    for task in removed:
        best_pos = 0
        best_cost = float("inf")
        for pos in range(len(solution.order) + 1):
            candidate = solution.copy()
            candidate.insert_task(task, pos)
            cost = candidate.cost()
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        solution.insert_task(task, best_pos)
