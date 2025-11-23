from __future__ import annotations

import dataclasses
import json
import math
import pathlib
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
    due_time: float
    is_pickup: bool = True
    pair_idx: int | None = None

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y


@dataclasses.dataclass
class ProblemInstance:
    tasks: List[Task]
    depot: tuple[float, float] = (0.0, 0.0)
    vehicle_capacity: float = 100.0
    e_max: float = 1000.0
    e_min: float = 200.0
    e0: float = 1000.0
    energy_rate: float = 0.5  # Wh per unit distance
    charge_rate: float = 10.0  # Wh per unit time
    charging_stations: List[tuple[float, float]] = dataclasses.field(default_factory=list)
    map_size: tuple[float, float] = (100.0, 100.0)
    amr_count: int = 1
    _distance_matrix: List[List[float]] = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._distance_matrix = self._build_distance_matrix()

    def _build_distance_matrix(self) -> List[List[float]]:
        """Pre-compute pairwise distances (depot included) for O(1) lookups."""
        points: List[tuple[float, float]] = [self.depot] + [task.position for task in self.tasks]
        n = len(points)
        matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
        for i, (ax, ay) in enumerate(points):
            for j in range(i, n):
                bx, by = points[j]
                dist = math.hypot(ax - bx, ay - by)
                matrix[i][j] = matrix[j][i] = dist
        return matrix

    def _index_from_obj(self, obj: Task | int | None) -> int:
        if obj is None:
            return 0
        if isinstance(obj, Task):
            return obj.idx + 1
        return int(obj) + 1

    def distance(self, a: Task | None, b: Task | None) -> float:
        return self._distance_matrix[self._index_from_obj(a)][self._index_from_obj(b)]

    def distance_idx(self, a_idx: int | None, b_idx: int | None) -> float:
        return self._distance_matrix[self._index_from_obj(a_idx)][self._index_from_obj(b_idx)]


def _load_instance_from_file(path: pathlib.Path) -> ProblemInstance:
    if not path.exists():
        raise FileNotFoundError(f"Problem data file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    depot = tuple(data.get("depot", (0.0, 0.0)))  # type: ignore[arg-type]
    vehicle_capacity = float(data.get("vehicle_capacity", 100.0))
    map_size = tuple(data.get("map_size", (100.0, 100.0)))  # type: ignore[arg-type]
    amr_count = int(data.get("amr_count", 1))
    e_max = float(data.get("e_max", 600.0))
    e_min = float(data.get("e_min", 120.0))
    e0 = float(data.get("e0", e_max))
    energy_rate = float(data.get("energy_rate", 1.0))
    charge_rate = float(data.get("charge_rate", 6.0))
    stations = [tuple(pt) for pt in data.get("charging_stations", [])]  # type: ignore[arg-type]
    tasks: List[Task] = []
    for idx, task_data in enumerate(data.get("tasks", [])):
        tasks.append(
            Task(
                idx=int(task_data.get("idx", idx)),
                x=float(task_data["x"]),
                y=float(task_data["y"]),
                demand=float(task_data.get("demand", 1.0)),
                service_time=float(task_data.get("service_time", 10.0)),
                due_time=float(task_data.get("due_time", 100.0)),
                is_pickup=bool(task_data.get("is_pickup", True)),
                pair_idx=task_data.get("pair_idx"),
            )
        )
    tasks.sort(key=lambda t: t.idx)
    return ProblemInstance(
        tasks=tasks,
        depot=depot,
        vehicle_capacity=vehicle_capacity,
        map_size=map_size,
        amr_count=amr_count,
        e_max=e_max,
        e_min=e_min,
        e0=e0,
        energy_rate=energy_rate,
        charge_rate=charge_rate,
        charging_stations=stations,
    )


def generate_random_instance(config: ProblemConfig) -> ProblemInstance:
    if config.data_file:
        data_path = pathlib.Path(config.data_file)
        return _load_instance_from_file(data_path)
    rng = random.Random(config.seed)
    tasks: List[Task] = []
    width, height = config.map_size
    # charging stations
    stations: List[tuple[float, float]] = []
    if width <= 100:
        stations = [(0.25 * width, 0.25 * height), (0.75 * width, 0.75 * height)]
    else:
        stations = [
            (0.25 * width, 0.25 * height),
            (0.25 * width, 0.75 * height),
            (0.75 * width, 0.25 * height),
            (0.75 * width, 0.75 * height),
        ]
    num_pickups = config.num_tasks
    for idx in range(num_pickups):
        if config.distribution == "clustered":
            cluster = rng.randint(0, 2)
            base_x = 0.2 * width + cluster * 0.25 * width + rng.random() * 0.1 * width
            base_y = 0.2 * height + cluster * 0.25 * height + rng.random() * 0.1 * height
            x = max(0.0, min(width, base_x + rng.uniform(-0.05 * width, 0.05 * width)))
            y = max(0.0, min(height, base_y + rng.uniform(-0.05 * height, 0.05 * height)))
        else:
            x = rng.uniform(0, width)
            y = rng.uniform(0, height)
        demand = rng.uniform(1, 10)
        service_time = rng.uniform(5, 15)
        depot_dist = math.hypot(x - 0.0, y - 0.0)
        round_trip_base = depot_dist * 2  # depot -> pick -> depot baseline
        workload_factor = 1.0  # no scaling
        k1 = 0.6
        offset = rng.uniform(0.0, 1.0)
        pickup_idx = idx
        delivery_idx = idx + num_pickups
        dx = rng.uniform(-0.1 * width, 0.1 * width)
        dy = rng.uniform(-0.1 * height, 0.1 * height)
        x_del = max(0.0, min(width, x + dx))
        y_del = max(0.0, min(height, y + dy))
        leg_pd = math.hypot(x_del - x, y_del - y)
        leg_dd = math.hypot(x_del - 0.0, y_del - 0.0)
        due_time_pick = ((round_trip_base + leg_pd) * k1 + service_time + offset) * workload_factor
        k2 = 0.5
        due_time_del = due_time_pick + (leg_pd * k2 + service_time) * workload_factor + rng.uniform(0.0, 1.0)
        tasks.append(
            Task(
                idx=pickup_idx,
                x=x,
                y=y,
                demand=demand,
                service_time=service_time,
                due_time=due_time_pick,
                is_pickup=True,
                pair_idx=delivery_idx,
            )
        )
        tasks.append(
            Task(
                idx=delivery_idx,
                x=x_del,
                y=y_del,
                demand=-demand,
                service_time=service_time,
                due_time=due_time_del,
                is_pickup=False,
                pair_idx=pickup_idx,
            )
        )
    tasks.sort(key=lambda t: t.idx)
    return ProblemInstance(
        tasks=tasks,
        map_size=config.map_size,
        amr_count=config.amr_count,
        charging_stations=stations,
    )
