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
    routes: List[List[int]]

    def __post_init__(self) -> None:
        # Normalize and repair pickup-delivery order on all routes.
        self.routes = [list(r) for r in self.routes]
        self._repair_pickup_delivery()

    @property
    def order(self) -> List[int]:
        return [idx for route in self.routes for idx in route]

    def copy(self) -> "Solution":
        return Solution(self.instance, [list(r) for r in self.routes])

    @property
    def tasks(self) -> List[Task]:
        return [self.instance.tasks[idx] for idx in self.order]

    def tardiness_by_task(self) -> dict[int, float]:
        """Return per-task tardiness (arrival - due if late, else 0)."""
        inst = self.instance
        tardiness: dict[int, float] = {}
        for route in self.routes:
            time_now = 0.0
            prev_idx: int | None = None
            for idx in route:
                task = inst.tasks[idx]
                travel_time = inst.distance_idx(prev_idx, idx)
                arrival = time_now + travel_time
                tardiness[idx] = max(0.0, arrival - task.due_time)
                time_now = arrival + task.service_time
                prev_idx = idx
        return tardiness

    def cost(self) -> float:
        """Total tardiness across AMRs + soft penalties."""
        inst = self.instance
        tard_sum = 0.0
        penalties = 0.0
        n_pick = len(inst.tasks) // 2
        for route in self.routes:
            energy = inst.e0
            time_now = 0.0
            load = 0.0
            for pos, idx in enumerate(route):
                task = inst.tasks[idx]
                prev_idx = route[pos - 1] if pos > 0 else None
                travel_time = inst.distance_idx(prev_idx, idx)
                travel_energy = travel_time * inst.energy_rate
                arrival = time_now + travel_time
                energy -= travel_energy
                if energy < inst.e_min:
                    deficit = inst.e_min - energy
                    penalties += deficit * 1.0  # mild energy penalty
                    energy = inst.e_min
                # If delivery comes before pickup, add mild penalty instead of hard.
                if not task.is_pickup:
                    pickup_idx = idx - n_pick
                    if pickup_idx not in route[:pos]:
                        penalties += 100.0
                tard_sum += max(0.0, arrival - task.due_time)
                load += task.demand
                if load > inst.vehicle_capacity:
                    penalties += (load - inst.vehicle_capacity) * 10
                time_now = arrival + task.service_time
        return tard_sum + penalties

    def _detour_time_energy(self, from_idx: int | None, to_idx: int) -> tuple[float, float]:
        inst = self.instance
        if not inst.charging_stations:
            return 0.0, 0.0
        from_pos = inst.depot if from_idx is None else inst.tasks[from_idx].position
        to_pos = inst.tasks[to_idx].position
        best_time = float("inf")
        best_energy = 0.0
        for sx, sy in inst.charging_stations:
            leg1 = math.hypot(from_pos[0] - sx, from_pos[1] - sy)
            leg2 = math.hypot(sx - to_pos[0], sy - to_pos[1])
            total = leg1 + leg2
            if total < best_time:
                best_time = total
                best_energy = total * inst.energy_rate
        if best_time == float("inf"):
            return 0.0, 0.0
        return best_time, best_energy

    def remove_tasks(self, task_indices: Sequence[int]) -> List[Task]:
        removed: List[Task] = []
        order_set = set(task_indices)
        for r_idx, route in enumerate(self.routes):
            new_route = []
            for idx in route:
                if idx in order_set:
                    removed.append(self.instance.tasks[idx])
                else:
                    new_route.append(idx)
            self.routes[r_idx] = new_route
        return removed

    def insert_task(self, task: Task, route_idx: int, position: int) -> None:
        while len(self.routes) <= route_idx:
            self.routes.append([])
        # Enforce pickup before delivery on insertion
        if not task.is_pickup and task.pair_idx is not None:
            # ensure pickup exists in this route
            if task.pair_idx not in self.routes[route_idx]:
                # append pickup first
                self.routes[route_idx].insert(min(position, len(self.routes[route_idx])), task.pair_idx)
                position = min(position + 1, len(self.routes[route_idx]))
            # delivery cannot be before its pickup
            pickup_pos = self.routes[route_idx].index(task.pair_idx)
            if position <= pickup_pos:
                position = pickup_pos + 1
        self.routes[route_idx].insert(position, task.idx)

    def save(self, path: str) -> None:
        data = {
            "cost": self.cost(),
            "routes": self.routes,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @staticmethod
    def from_order(instance: ProblemInstance, order: List[int]) -> "Solution":
        routes: List[List[int]] = [[] for _ in range(max(1, instance.amr_count))]
        for i, idx in enumerate(order):
            routes[i % len(routes)].append(idx)
        return Solution(instance, routes)

    def _repair_pickup_delivery(self) -> None:
        """Ensure pickups precede their deliveries and stay on same route."""
        n_pick = len(self.instance.tasks) // 2
        num_routes = max(1, len(self.routes))
        new_routes: List[List[int]] = [[] for _ in range(num_routes)]
        assigned: set[int] = set()
        for r_idx, route in enumerate(self.routes):
            target = r_idx % num_routes
            for idx in route:
                if idx in assigned:
                    continue
                task = self.instance.tasks[idx]
                if task.is_pickup:
                    pickup_idx = idx
                    delivery_idx = idx + n_pick
                else:
                    delivery_idx = idx
                    pickup_idx = idx - n_pick
                if pickup_idx not in assigned:
                    new_routes[target].append(pickup_idx)
                    assigned.add(pickup_idx)
                if delivery_idx not in assigned:
                    new_routes[target].append(delivery_idx)
                    assigned.add(delivery_idx)
        self.routes = new_routes


def random_solution(instance: ProblemInstance, seed: int | None = None) -> Solution:
    rng = random.Random(seed)
    order = list(range(len(instance.tasks)))
    rng.shuffle(order)
    return Solution.from_order(instance=instance, order=order)
