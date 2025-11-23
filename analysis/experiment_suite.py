#!/usr/bin/env python3
"""Run the five requested experiments end-to-end and materialize artefacts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import pathlib
import random
import statistics
import sys
import time
from collections import defaultdict
from typing import Dict, List, Sequence

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

def _ensure_matplotlib_dir() -> None:
    """Force Matplotlib to use a writable cache dir to avoid permission errors."""
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = pathlib.Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    except OSError:
        # If mkdir fails, let Matplotlib fall back to default temp dir.
        return


_ensure_matplotlib_dir()

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib optional in CI
    plt = None
import numpy as np
try:  # Optional Gurobi exact solver
    import gurobipy as gp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    gp = None
from scipy import stats

from src.config import AlgorithmConfig, ProblemConfig
from src.data import ProblemInstance, generate_random_instance
from src.drl_agent import DRLAgent
from src.hea import HEA
from src.solution import Solution

OUTPUT_DIR = pathlib.Path("results/analysis")
DRL_MODEL = pathlib.Path("models/drl_model_transformer_backup.json")
REMOVAL_METHODS = {
    "Genetic Algorithm",
    "Memetic Algorithm",
    "HEA-DRL",
}


@dataclasses.dataclass
class ScaleSetting:
    name: str
    num_tasks: int
    distribution: str
    population_size: int
    generations: int
    remove_count: int
    seeds: List[int]
    map_size: tuple[float, float]
    amr_count: int
    time_budget: float


@dataclasses.dataclass
class MethodSpec:
    name: str
    kind: str  # "hea", "constructive", "aco", "exact"
    applies_to: Sequence[str] = dataclasses.field(
        default_factory=lambda: ("small", "medium", "large")
    )
    removal_strategy: str | None = None
    remove_override: int | None = None
    remove_multiplier: float | None = None
    seed_offset: int = 0
    population_override: int | None = None
    generations_override: int | None = None
    crossover_override: float | None = None
    mutation_override: float | None = None
    elite_override: float | None = None
    time_multiplier: float = 1.0


@dataclasses.dataclass
class ExperimentRecord:
    scale: str
    instance_id: str
    method: str
    tardiness: float
    time_sec: float
    stats: List[tuple[float, float, float]] | None
    timing: dict[str, float]


class ExperimentSuite:
    def __init__(
        self,
        output_dir: pathlib.Path = OUTPUT_DIR,
        enable_plots: bool = False,
        time_budget_scale: float = 1.0,
        seed_cap: int | None = None,
        scales_filter: List[str] | None = None,
        methods_filter: List[str] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_plots = enable_plots and plt is not None
        if enable_plots and plt is None:
            print("Matplotlib 未可用，自动禁用绘图输出。")
        self.time_budget_scale = time_budget_scale
        self.seed_cap = seed_cap
        self.scales_filter = set(s.lower() for s in scales_filter) if scales_filter else None
        self.methods_filter = set(m.lower() for m in methods_filter) if methods_filter else None
        self.scales: Dict[str, ScaleSetting] = {
            "small": ScaleSetting(
                name="small",
                num_tasks=7,
                distribution="uniform",
                population_size=32,
                generations=80,
                remove_count=2,
                seeds=[11, 13, 17, 19],
                map_size=(50.0, 50.0),
                amr_count=2,
                time_budget=30.0,
            ),
            "medium": ScaleSetting(
                name="medium",
                num_tasks=60,
                distribution="uniform",
                population_size=80,
                generations=140,
                remove_count=5,
                seeds=[101, 103, 105, 107],
                map_size=(150.0, 150.0),
                amr_count=8,
                time_budget=300.0,
            ),
            "large": ScaleSetting(
                name="large",
                num_tasks=120,
                distribution="clustered",
                population_size=140,
                generations=160,
                remove_count=7,
                seeds=[201, 203, 205, 207],
                map_size=(250.0, 250.0),
                amr_count=12,
                time_budget=600.0,
            ),
        }
        if self.seed_cap:
            for scale in self.scales.values():
                scale.seeds = scale.seeds[: self.seed_cap]
        self.methods: List[MethodSpec] = [
            MethodSpec(name="Constructive Heuristic", kind="constructive"),
            MethodSpec(name="TRIGA", kind="triga"),
            MethodSpec(
                name="Genetic Algorithm",
                kind="hea",
                removal_strategy="random",
                remove_multiplier=1.0,
                seed_offset=5,
            ),
            MethodSpec(
                name="VIGA",
                kind="viga",
                seed_offset=9,
            ),
            MethodSpec(
                name="Memetic Algorithm",
                kind="hea",
                removal_strategy="shaw",
                seed_offset=7,
            ),
            MethodSpec(name="Ant Colony", kind="aco", seed_offset=11),
            MethodSpec(
                name="HEA-DRL",
                kind="hea",
                removal_strategy="drl_hybrid",
                seed_offset=5,
                remove_multiplier=None,
                population_override=None,
                generations_override=None,
                crossover_override=None,
                mutation_override=None,
                elite_override=None,
                time_multiplier=1.0,
            ),
            MethodSpec(
                name="HEA-Random",
                kind="hea",
                removal_strategy="random",
                seed_offset=5,
                remove_multiplier=None,
                population_override=None,
                generations_override=None,
                crossover_override=None,
                mutation_override=None,
                elite_override=None,
                time_multiplier=1.0,
            ),
            MethodSpec(
                name="HEA-Shaw",
                kind="hea",
                removal_strategy="shaw",
                seed_offset=5,
                time_multiplier=1.0,
            ),
            MethodSpec(
                name="HEA-WorstSlack",
                kind="hea",
                removal_strategy="worst_slack",
                seed_offset=5,
                time_multiplier=1.0,
            ),
            MethodSpec(
                name="HEA-MaxDistance",
                kind="hea",
                removal_strategy="max_distance",
                seed_offset=5,
                time_multiplier=1.0,
            ),
        ]
        self.records: List[ExperimentRecord] = []
        self.transfer_records: List[dict] = []
        self.drl_agent = self._load_drl_agent()

    def _load_drl_agent(self) -> DRLAgent:
        if not DRL_MODEL.exists():
            raise FileNotFoundError(
                "DRL model file models/drl_model.json is required for HEA-DRL experiments"
            )
        return DRLAgent.load(DRL_MODEL)

    # ------------------------------------------------------------------
    # Experiment orchestration
    # ------------------------------------------------------------------
    def run(self) -> None:
        print("[1/6] Running core benchmark experiments...")
        self.records = self._run_core_experiments()
        print("[2/6] Running transfer experiments...")
        self.transfer_records = self._run_transfer_experiments()
        print("[3/6] Generating performance tables...")
        self.generate_performance_tables()
        print("[4/6] Evaluating statistical significance...")
        self.generate_significance_report()
        print("[5/6] Creating anytime and transfer visualisations...")
        self.generate_anytime_plots()
        self.generate_transfer_summary()
        print("[6/6] Detailing runtime breakdown...")
        self.generate_runtime_breakdown()
        print(f"分析完成，输出位于: {self.output_dir}")

    def _run_core_experiments(self) -> List[ExperimentRecord]:
        records: List[ExperimentRecord] = []
        for scale_name, scale in self.scales.items():
            if self.scales_filter and scale_name.lower() not in self.scales_filter:
                continue
            for idx, seed in enumerate(scale.seeds):
                instance = self._build_instance(scale, seed)
                instance_id = f"{scale_name}_I{idx + 1}"
                for method in self.methods:
                    if self.methods_filter and method.name.lower() not in self.methods_filter:
                        continue
                    if scale_name not in method.applies_to:
                        continue
                    record = self._dispatch_method(
                        method, scale_name, instance_id, instance, scale, seed
                    )
                    if record:
                        records.append(record)
        return records

    def _build_instance(self, scale: ScaleSetting, seed: int) -> ProblemInstance:
        cfg = ProblemConfig(
            instance_type="random",
            num_tasks=scale.num_tasks,
            distribution=scale.distribution,
            seed=seed,
            map_size=scale.map_size,
            amr_count=scale.amr_count,
        )
        return generate_random_instance(cfg)

    def _dispatch_method(
        self,
        method: MethodSpec,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
        scale: ScaleSetting,
        seed: int,
    ) -> ExperimentRecord | None:
        if method.kind == "hea":
            return self._run_hea(method, scale_name, instance_id, instance, scale, seed)
        if method.kind == "constructive":
            return self._run_constructive(method.name, scale_name, instance_id, instance)
        if method.kind == "exact":
            return self._run_exact(method.name, scale_name, instance_id, instance)
        if method.kind == "aco":
            return self._run_aco(method.name, scale_name, instance_id, instance, seed)
        if method.kind == "triga":
            return self._run_triga(method.name, scale_name, instance_id, instance, seed)
        if method.kind == "viga":
            return self._run_viga(method.name, scale_name, instance_id, instance, scale, seed)
        raise ValueError(f"Unsupported method kind: {method.kind}")

    # ------------------------------------------------------------------
    # Individual solvers
    # ------------------------------------------------------------------
    def _run_hea(
        self,
        method: MethodSpec,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
        scale: ScaleSetting,
        seed: int,
    ) -> ExperimentRecord:
        algo_seed = seed + method.seed_offset
        remove_count = scale.remove_count
        if method.remove_override is not None:
            remove_count = method.remove_override
        elif method.remove_multiplier is not None:
            remove_count = max(
                1, int(round(scale.remove_count * method.remove_multiplier))
            )
        removal_strategy = method.removal_strategy or "random"
        population_size = method.population_override or scale.population_size
        generations = method.generations_override or scale.generations
        eval_budget = scale.population_size * (scale.generations + 1)
        time_budget = scale.time_budget * self.time_budget_scale * method.time_multiplier
        crossover_rate = method.crossover_override or 0.85
        mutation_rate = method.mutation_override or 0.2
        elite_rate = method.elite_override or 0.1
        algo_cfg = AlgorithmConfig(
            removal_strategy=removal_strategy,
            remove_count=remove_count,
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_rate=elite_rate,
            model_path=str(DRL_MODEL)
            if removal_strategy in {"drl", "drl_hybrid"}
            else None,
            eval_budget=eval_budget,
            time_budget=time_budget,
            seed=algo_seed,
        )
        logger = logging.getLogger(
            f"suite.{scale_name}.{instance_id}.{method.name.replace(' ', '_')}"
        )
        logger.handlers = []
        logger.addHandler(logging.NullHandler())
        agent = (
            self.drl_agent
            if method.removal_strategy in {"drl", "drl_hybrid"}
            else None
        )
        algo = HEA(instance, algo_cfg, logger, agent)
        result = algo.run()
        stats = [(s.elapsed_sec, s.best_cost, s.average_cost) for s in result.stats]
        time_sec = result.timing.get("T_total", 0.0)
        return ExperimentRecord(
            scale=scale_name,
            instance_id=instance_id,
            method=method.name,
            tardiness=result.best_solution.cost(),
            time_sec=time_sec,
            stats=stats,
            timing=result.timing,
        )

    def _run_constructive(
        self,
        method_name: str,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
    ) -> ExperimentRecord:
        start = time.perf_counter()
        order = self._nearest_neighbor_order(instance)
        solution = Solution.from_order(instance, order)
        tardiness = solution.cost()
        elapsed = time.perf_counter() - start
        return ExperimentRecord(
            scale=scale_name,
            instance_id=instance_id,
            method=method_name,
            tardiness=tardiness,
            time_sec=elapsed,
            stats=None,
            timing={"T_total": elapsed},
        )

    def _run_exact(
        self,
        method_name: str,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
    ) -> ExperimentRecord | None:
        if len(instance.tasks) > 14:
            logging.getLogger("suite").warning(
                "Exact solver skipped for %s (%s tasks > 14)", instance_id, len(instance.tasks)
            )
            return None
        start = time.perf_counter()
        order = self._held_karp(instance)
        solution = Solution.from_order(instance, order)
        tardiness = solution.cost()
        elapsed = time.perf_counter() - start
        return ExperimentRecord(
            scale=scale_name,
            instance_id=instance_id,
            method=method_name,
            tardiness=tardiness,
            time_sec=elapsed,
            stats=None,
            timing={"T_total": elapsed},
        )

    def _run_ortools(
        self,
        method_name: str,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
    ) -> ExperimentRecord | None:
        logger = logging.getLogger("suite")
        if pywrapcp is None or routing_enums_pb2 is None:
            logger.warning("OR-Tools not available, skipping %s for %s", method_name, instance_id)
            return None
        n = len(instance.tasks)
        if n == 0:
            return None
        if n > 80:
            logger.warning("OR-Tools PDP-TW skipped for %s (tasks=%s > 80)", instance_id, n)
            return None
        start = time.perf_counter()
        num_vehicles = max(1, instance.amr_count)
        depot_node = 0
        node_count = n + 1  # depot + tasks
        manager = pywrapcp.RoutingIndexManager(node_count, num_vehicles, depot_node)
        routing = pywrapcp.RoutingModel(manager)

        def dist_nodes(a: int, b: int) -> float:
            if a == depot_node:
                from_task = None
            else:
                from_task = instance.tasks[a - 1]
            if b == depot_node:
                to_task = None
            else:
                to_task = instance.tasks[b - 1]
            return instance.distance(from_task, to_task)

        def transit_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(round(dist_nodes(from_node, to_node)))

        transit_index = routing.RegisterTransitCallback(transit_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

        # Time windows with service time
        routing.AddDimension(
            transit_index,
            0,  # no slack
            10000,  # horizon
            False,
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")
        for task_idx, task in enumerate(instance.tasks):
            node = task_idx + 1
            idx = manager.NodeToIndex(node)
            time_dim.CumulVar(idx).SetRange(0, int(round(task.due_time)))
        depot_index = manager.NodeToIndex(depot_node)
        time_dim.CumulVar(depot_index).SetRange(0, 10000)

        # Capacities
        demands = [0] + [int(round(t.demand)) for t in instance.tasks]
        def demand_callback(from_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]
        demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_index,
            0,
            [int(round(instance.vehicle_capacity))] * num_vehicles,
            True,
            "Capacity",
        )

        # Pickup-delivery pairs
        for task in instance.tasks:
            if task.is_pickup and task.pair_idx is not None:
                pick_node = task.idx + 1
                del_node = task.pair_idx + 1
                routing.AddPickupAndDelivery(manager.NodeToIndex(pick_node), manager.NodeToIndex(del_node))
                routing.solver().Add(
                    routing.VehicleVar(manager.NodeToIndex(pick_node))
                    == routing.VehicleVar(manager.NodeToIndex(del_node))
                )
                routing.solver().Add(
                    time_dim.CumulVar(manager.NodeToIndex(pick_node))
                    <= time_dim.CumulVar(manager.NodeToIndex(del_node))
                )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_params.time_limit.FromSeconds(60)
        solution = routing.SolveWithParameters(search_params)
        elapsed = time.perf_counter() - start
        if solution is None:
            logger.warning("OR-Tools solver failed for %s", instance_id)
            return None

        routes: List[List[int]] = [[] for _ in range(num_vehicles)]
        for v in range(num_vehicles):
            index = routing.Start(v)
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != depot_node:
                    routes[v].append(node - 1)
                index = solution.Value(routing.NextVar(index))
        solution_obj = Solution(instance, routes)
        tardiness = solution_obj.cost()
        return ExperimentRecord(
            scale=scale_name,
            instance_id=instance_id,
            method=method_name,
            tardiness=tardiness,
            time_sec=elapsed,
            stats=None,
            timing={"T_total": elapsed},
        )

    def _run_aco(
        self,
        method_name: str,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
        seed: int,
    ) -> ExperimentRecord:
        start = time.perf_counter()
        order = self._ant_colony_opt(instance, seed)
        solution = Solution.from_order(instance, order)
        tardiness = solution.cost()
        elapsed = time.perf_counter() - start
        return ExperimentRecord(
            scale=scale_name,
            instance_id=instance_id,
            method=method_name,
            tardiness=tardiness,
            time_sec=elapsed,
            stats=None,
            timing={"T_total": elapsed},
        )

    def _run_triga(
        self,
        method_name: str,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
        seed: int,
    ) -> ExperimentRecord:
        start = time.perf_counter()
        rng = random.Random(seed + 197)
        best_order: List[int] | None = None
        best_cost = math.inf
        trials = 7
        for t in range(trials):
            order = self._triga_construct(instance, rng.randint(0, 1_000_000))
            improved_order, cost = self._two_opt_improve(instance, order)
            if cost < best_cost:
                best_cost = cost
                best_order = improved_order
        if best_order is None:
            best_order = list(range(len(instance.tasks)))
            best_cost = Solution.from_order(instance, best_order).cost()
        elapsed = time.perf_counter() - start
        return ExperimentRecord(
            scale=scale_name,
            instance_id=instance_id,
            method=method_name,
            tardiness=best_cost,
            time_sec=elapsed,
            stats=None,
            timing={"T_total": elapsed},
        )

    def _run_viga(
        self,
        method_name: str,
        scale_name: str,
        instance_id: str,
        instance: ProblemInstance,
        scale: ScaleSetting,
        seed: int,
    ) -> ExperimentRecord:
        algo_seed = seed + 211
        algo_cfg = AlgorithmConfig(
            removal_strategy="random",
            remove_count=max(1, scale.remove_count - 1),
            population_size=scale.population_size + 6,
            generations=scale.generations + 10,
            crossover_rate=0.9,
            mutation_rate=0.25,
            elite_rate=0.15,
            model_path=None,
            seed=algo_seed,
        )
        logger = logging.getLogger(
            f"suite.{scale_name}.{instance_id}.{method_name.replace(' ', '_')}"
        )
        logger.handlers = []
        logger.addHandler(logging.NullHandler())
        algo = HEA(instance, algo_cfg, logger)
        result = algo.run()
        stats = [(s.elapsed_sec, s.best_cost, s.average_cost) for s in result.stats]
        time_sec = result.timing.get("T_total", 0.0)
        return ExperimentRecord(
            scale=scale_name,
            instance_id=instance_id,
            method=method_name,
            tardiness=result.best_solution.cost(),
            time_sec=time_sec,
            stats=stats,
            timing=result.timing,
        )

    # ------------------------------------------------------------------
    # Helper heuristics
    # ------------------------------------------------------------------
    def _nearest_neighbor_order(self, instance: ProblemInstance) -> List[int]:
        remaining = set(range(len(instance.tasks)))
        order: List[int] = []
        current: int | None = None
        while remaining:
            def distance_to(idx: int) -> float:
                if current is None:
                    return instance.distance(None, instance.tasks[idx])
                return instance.distance(instance.tasks[current], instance.tasks[idx])

            next_idx = min(remaining, key=distance_to)
            order.append(next_idx)
            remaining.remove(next_idx)
            current = next_idx
        return order

    def _held_karp(self, instance: ProblemInstance) -> List[int]:
        n = len(instance.tasks)
        dist = [[instance.distance(a, b) for b in instance.tasks] for a in instance.tasks]
        depot_dist = [instance.distance(None, task) for task in instance.tasks]
        dp: dict[tuple[int, int], tuple[float, int]] = {}
        for j in range(n):
            mask = 1 << j
            dp[(mask, j)] = (depot_dist[j], -1)
        for mask in range(1, 1 << n):
            for j in range(n):
                if not mask & (1 << j):
                    continue
                if mask == (1 << j):
                    continue
                prev_mask = mask ^ (1 << j)
                best_cost = math.inf
                best_prev = -1
                for k in range(n):
                    if not prev_mask & (1 << k):
                        continue
                    prev_cost = dp[(prev_mask, k)][0] + dist[k][j]
                    if prev_cost < best_cost:
                        best_cost = prev_cost
                        best_prev = k
                dp[(mask, j)] = (best_cost, best_prev)
        full_mask = (1 << n) - 1
        best_cost = math.inf
        best_last = -1
        for j in range(n):
            cost = dp[(full_mask, j)][0] + instance.distance(instance.tasks[j], None)
            if cost < best_cost:
                best_cost = cost
                best_last = j
        order: List[int] = []
        mask = full_mask
        curr = best_last
        while curr != -1:
            order.append(curr)
            _, prev = dp[(mask, curr)]
            mask ^= 1 << curr
            curr = prev
        order.reverse()
        return order

    def _ant_colony_opt(self, instance: ProblemInstance, seed: int) -> List[int]:
        rng = random.Random(seed)
        n = len(instance.tasks)
        pheromone = [[1.0 for _ in range(n)] for _ in range(n)]
        depot_pheromone = [1.0 for _ in range(n)]
        alpha = 1.1
        beta = 2.0
        rho = 0.25
        iterations = 25
        ants = 14
        best_order: List[int] | None = None
        best_cost = math.inf
        for _ in range(iterations):
            iteration_best_order: List[int] | None = None
            iteration_best_cost = math.inf
            for _ in range(ants):
                order = self._construct_ant_route(instance, pheromone, depot_pheromone, rng, alpha, beta)
                cost = Solution.from_order(instance, order).cost()
                if cost < iteration_best_cost:
                    iteration_best_cost = cost
                    iteration_best_order = order
                if cost < best_cost:
                    best_cost = cost
                    best_order = order
            if iteration_best_order is None:
                continue
            self._evaporate_pheromone(pheromone, depot_pheromone, rho)
            self._deposit_pheromone(
                instance,
                pheromone,
                depot_pheromone,
                iteration_best_order,
                1.0 / max(iteration_best_cost, 1e-6),
            )
        if best_order is None:
            best_order = list(range(n))
        return best_order

    def _construct_ant_route(
        self,
        instance: ProblemInstance,
        pheromone: List[List[float]],
        depot_pheromone: List[float],
        rng: random.Random,
        alpha: float,
        beta: float,
    ) -> List[int]:
        remaining = set(range(len(instance.tasks)))
        order: List[int] = []
        current: int | None = None
        while remaining:
            weights: List[tuple[int, float]] = []
            for candidate in remaining:
                if current is None:
                    tau = depot_pheromone[candidate]
                    dist = instance.distance(None, instance.tasks[candidate])
                else:
                    tau = pheromone[current][candidate]
                    dist = instance.distance(instance.tasks[current], instance.tasks[candidate])
                eta = 1.0 / (dist + 1e-6)
                weight = (tau ** alpha) * (eta ** beta)
                weights.append((candidate, weight))
            total = sum(w for _, w in weights)
            if total <= 0:
                chosen = rng.choice(list(remaining))
            else:
                threshold = rng.random() * total
                acc = 0.0
                chosen = weights[-1][0]
                for candidate, weight in weights:
                    acc += weight
                    if acc >= threshold:
                        chosen = candidate
                        break
            order.append(chosen)
            remaining.remove(chosen)
            current = chosen
        return order

    def _evaporate_pheromone(
        self,
        pheromone: List[List[float]],
        depot_pheromone: List[float],
        rho: float,
    ) -> None:
        for i in range(len(pheromone)):
            for j in range(len(pheromone)):
                pheromone[i][j] *= (1.0 - rho)
                pheromone[i][j] = max(1e-4, pheromone[i][j])
        for idx in range(len(depot_pheromone)):
            depot_pheromone[idx] *= (1.0 - rho)
            depot_pheromone[idx] = max(1e-4, depot_pheromone[idx])

    def _deposit_pheromone(
        self,
        instance: ProblemInstance,
        pheromone: List[List[float]],
        depot_pheromone: List[float],
        order: Sequence[int],
        amount: float,
    ) -> None:
        if not order:
            return
        first = order[0]
        depot_pheromone[first] += amount
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            pheromone[a][b] += amount
            pheromone[b][a] += amount
        depot_pheromone[order[-1]] += amount

    def _triga_construct(self, instance: ProblemInstance, rand_seed: int) -> List[int]:
        rng = random.Random(rand_seed)
        remaining = list(range(len(instance.tasks)))
        if not remaining:
            return []
        order: List[int] = []
        current = rng.choice(remaining)
        order.append(current)
        remaining.remove(current)
        while remaining:
            scored: List[tuple[float, int]] = []
            for candidate in remaining:
                if current is None:
                    dist = instance.distance(None, instance.tasks[candidate])
                else:
                    dist = instance.distance(instance.tasks[current], instance.tasks[candidate])
                demand_penalty = instance.tasks[candidate].demand / instance.vehicle_capacity
                service_bias = instance.tasks[candidate].service_time / 10.0
                randomness = rng.random() * 0.05
                score = dist + 2.0 * demand_penalty + 0.5 * service_bias + randomness
                scored.append((score, candidate))
            scored.sort(key=lambda item: item[0])
            pick = scored[0][1]
            insert_pos = rng.randint(0, len(order))
            order.insert(insert_pos, pick)
            remaining.remove(pick)
            current = pick
        return order

    def _two_opt_improve(
        self, instance: ProblemInstance, order: List[int], max_iterations: int = 25
    ) -> tuple[List[int], float]:
        if not order:
            return [], 0.0
        best_order = list(order)
        best_cost = Solution.from_order(instance, best_order).cost()
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            for i in range(len(best_order) - 2):
                for j in range(i + 2, len(best_order)):
                    if j - i == 1:
                        continue
                    new_order = (
                        best_order[: i + 1]
                        + best_order[i + 1 : j + 1][::-1]
                        + best_order[j + 1 :]
                    )
                    new_cost = Solution.from_order(instance, new_order).cost()
                    if new_cost < best_cost - 1e-6:
                        best_cost = new_cost
                        best_order = new_order
                        improved = True
                        break
                if improved:
                    break
        return best_order, best_cost

    # ------------------------------------------------------------------
    # Transfer experiments
    # ------------------------------------------------------------------
    def _run_transfer_experiments(self) -> List[dict]:
        try:
            problems: List[tuple[str, pathlib.Path]] = []
            for name, path in [
                ("VRPTW", pathlib.Path("data/vrptw_instance.json")),
                ("E-VRPTW", pathlib.Path("data/evrptw_instance.json")),
            ]:
                if path.exists():
                    problems.append((name, path))
            if not problems:
                return []
            methods = [
                ("HEA-DRL", "drl_hybrid"),
                ("HEA-Shaw", "shaw"),
                ("HEA-Random", "random"),
            ]
            records: List[dict] = []
            for problem_name, path in problems:
                cfg = ProblemConfig(data_file=str(path))
                tmp_instance = generate_random_instance(cfg)
                remove_count = max(1, int(0.15 * len(tmp_instance.tasks)))
                for method_name, strategy in methods:
                    algo_cfg = AlgorithmConfig(
                        removal_strategy=strategy,
                        remove_count=remove_count,
                        population_size=40,
                        generations=60,
                        crossover_rate=0.85,
                        mutation_rate=0.2,
                        elite_rate=0.1,
                        model_path=str(DRL_MODEL) if "drl" in strategy else None,
                    )
                    logger = logging.getLogger(f"transfer.{problem_name}.{method_name}")
                    logger.handlers = []
                    logger.addHandler(logging.NullHandler())
                    agent = DRLAgent.load(DRL_MODEL) if "drl" in strategy else None
                    instance = generate_random_instance(cfg)
                    algo = HEA(instance, algo_cfg, logger, agent)
                    result = algo.run()
                    records.append(
                        {
                            "problem": problem_name,
                            "method": method_name,
                            "tardiness": result.best_solution.cost(),
                            "time_sec": result.timing.get("T_total", 0.0),
                        }
                    )
            return records
        except Exception as exc:
            logging.getLogger("suite").warning("Transfer experiments skipped: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Artefact generation
    # ------------------------------------------------------------------
    def generate_performance_tables(self) -> None:
        best_ref = self._best_reference()
        best_positive: Dict[tuple[str, str], float] = {}
        for r in self.records:
            key = (r.scale, r.instance_id)
            if r.tardiness > 1e-9:
                best_positive[key] = min(best_positive.get(key, float("inf")), r.tardiness)
        grouped: Dict[str, Dict[str, List[ExperimentRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record in self.records:
            grouped[record.scale][record.method].append(record)
        lines = ["# 性能对比总表", ""]
        summary: Dict[str, Dict[str, dict]] = {}
        for scale in ("small", "medium", "large"):
            if scale not in grouped:
                continue
            lines.append(f"## {scale.title()} 实例")
            lines.append("| 策略 | 平均延迟 ± std | 平均时间 (s) | 平均最优性缺口 (%) |")
            lines.append("| --- | --- | --- | --- |")
            summary.setdefault(scale, {})
            for method, records in grouped[scale].items():
                tardiness = [r.tardiness for r in records]
                mean_t = statistics.mean(tardiness)
                std_t = statistics.pstdev(tardiness) if len(tardiness) > 1 else 0.0
                runtimes = [r.time_sec for r in records]
                mean_time = statistics.mean(runtimes)
                gaps = []
                for r in records:
                    denom = best_ref[(r.scale, r.instance_id)]
                    if denom <= 1e-9:
                        denom = best_positive.get((r.scale, r.instance_id), 1.0)
                    gaps.append((r.tardiness - denom) / denom * 100.0)
                mean_gap = statistics.mean(gaps) if gaps else 0.0
                summary[scale][method] = {
                    "mean_tardiness": mean_t,
                    "std_tardiness": std_t,
                    "mean_time": mean_time,
                    "mean_gap": mean_gap,
                }
                lines.append(
                    f"| {method} | {mean_t:.1f} ± {std_t:.1f} | {mean_time:.2f} | {mean_gap:.2f} |"
                )
            lines.append("")
        (self.output_dir / "performance_tables.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        (self.output_dir / "performance_records.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def generate_significance_report(self) -> None:
        lines = ["# 统计显著性检验", "", "置信水平：95%，采用 Wilcoxon (配对) 与 Friedman (整体) 检验。", ""]
        report: Dict[str, dict] = {}
        scale_rankings: Dict[str, dict] = {}
        for scale in ("small", "medium", "large"):
            records = [r for r in self.records if r.scale == scale]
            if not records:
                continue
            per_instance: Dict[str, Dict[str, float]] = defaultdict(dict)
            for r in records:
                per_instance[r.instance_id][r.method] = r.tardiness
            methods = sorted({r.method for r in records})
            # Filter methods that exist on every instance
            complete_methods = [
                m
                for m in methods
                if all(m in per_instance[inst] for inst in per_instance)
            ]
            if "HEA-DRL" not in complete_methods or len(complete_methods) < 2:
                continue
            lines.append(f"## {scale.title()} 实例")
            wilcoxon: List[dict] = []
            instances = sorted(per_instance.keys())
            for method in complete_methods:
                if method == "HEA-DRL":
                    continue
                hea_values = []
                baseline_values = []
                for inst in instances:
                    hea_values.append(per_instance[inst]["HEA-DRL"])
                    baseline_values.append(per_instance[inst][method])
                if len(hea_values) < 2:
                    continue
                diffs = [hea - base for hea, base in zip(hea_values, baseline_values)]
                if all(math.isclose(diff, 0.0, abs_tol=1e-9) for diff in diffs):
                    stat = 0.0
                    p_value = 1.0
                    lines.append(
                        f"- HEA-DRL vs {method}: 样本完全一致，跳过 Wilcoxon 检验 (p=1.0000, statistic=0.00)"
                    )
                else:
                    try:
                        stat, p_value = stats.wilcoxon(
                            hea_values, baseline_values, alternative="less"
                        )
                        lines.append(
                            f"- HEA-DRL vs {method}: p={p_value:.4f}, statistic={stat:.2f}"
                        )
                    except ValueError as exc:
                        lines.append(
                            f"- HEA-DRL vs {method}: 无法执行 Wilcoxon 检验 ({exc})"
                        )
                        continue
                wilcoxon.append(
                    {
                        "baseline": method,
                        "p_value": float(p_value),
                        "statistic": float(stat),
                    }
                )
            # Friedman test across methods with complete data
            friedman_result: dict | None = None
            if len(complete_methods) < 3 or len(instances) < 2:
                lines.append("- Friedman 检验跳过：有效策略或实例数量不足。")
            else:
                matrix = [
                    [per_instance[inst][method] for inst in instances]
                    for method in complete_methods
                ]

                def _degenerate(mat: List[List[float]]) -> bool:
                    for column in zip(*mat):
                        first = column[0]
                        if any(
                            not math.isclose(value, first, rel_tol=1e-9, abs_tol=1e-9)
                            for value in column[1:]
                        ):
                            return False
                    return True

                if _degenerate(matrix):
                    lines.append("- Friedman 检验跳过：所有策略在各实例上的结果完全一致。")
                else:
                    friedman_stat, friedman_p = stats.friedmanchisquare(*matrix)
                    friedman_result = {
                        "statistic": float(friedman_stat),
                        "p_value": float(friedman_p),
                    }
                    lines.append(
                        f"- Friedman χ²={friedman_stat:.3f}, p={friedman_p:.5f}，p<0.05 表示整体差异显著"
                    )
            lines.append("")
            if wilcoxon:
                lines.append("| 对比 | 统计量 | p 值 | 显著性 |")
                lines.append("| --- | --- | --- | --- |")
                for entry in wilcoxon:
                    comparison = f"HEA-DRL vs {entry['baseline']}"
                    stat_val = entry["statistic"]
                    p_val = entry["p_value"]
                    significant = p_val < 0.05
                    p_str = f"{p_val:.4f}"
                    if significant:
                        p_str = f"**{p_str}**"
                    sig_label = "显著 (p<0.05)" if significant else ""
                    lines.append(
                        f"| {comparison} | {stat_val:.2f} | {p_str} | {sig_label} |"
                    )
                lines.append("")
            report[scale] = {
                "wilcoxon": wilcoxon,
                "friedman": friedman_result,
            }
            ranking_info = self._compute_rankings(per_instance, complete_methods, instances)
            if ranking_info:
                scale_rankings[scale] = ranking_info
                lines.append(
                    f"- 平均排名 (N={ranking_info['num_instances']}), CD={ranking_info['critical_difference']:.3f}"
                )
                lines.append("| 算法 | 平均排名 |")
                lines.append("| --- | --- |")
                for method, rank_value in sorted(
                    ranking_info["avg_ranks"].items(), key=lambda item: item[1]
                ):
                    lines.append(f"| {method} | {rank_value:.3f} |")
                lines.append("")
                report[scale]["ranking"] = ranking_info
        # Overall ranking across all scales
        aggregate_instances: Dict[str, Dict[str, float]] = defaultdict(dict)
        for record in self.records:
            key = f"{record.scale}_{record.instance_id}"
            aggregate_instances[key][record.method] = record.tardiness
        all_methods = sorted({r.method for r in self.records})
        overall_methods = [
            m for m in all_methods if all(m in aggregate_instances[inst] for inst in aggregate_instances)
        ]
        overall_instances = sorted(aggregate_instances.keys())
        overall_ranking = self._compute_rankings(
            aggregate_instances, overall_methods, overall_instances
        )
        if overall_ranking:
            lines.append("## 综合平均排名 (Nemenyi)")
            lines.append(
                f"- 使用 {overall_ranking['num_instances']} 个实例计算平均排名，临界差异 CD={overall_ranking['critical_difference']:.3f}"
            )
            lines.append("| 算法 | 平均排名 |")
            lines.append("| --- | --- |")
            for method, rank_value in sorted(
                overall_ranking["avg_ranks"].items(), key=lambda item: item[1]
            ):
                lines.append(f"| {method} | {rank_value:.3f} |")
            lines.append("")
            report["overall"] = {"ranking": overall_ranking}
            self._plot_cd_diagram(
                ranking_info=overall_ranking,
                title="Overall average ranks (Nemenyi)",
                path=self.output_dir / "nemenyi_cd.png",
            )
        removal_methods = [
            method
            for method in REMOVAL_METHODS
            if all(method in aggregate_instances[inst] for inst in aggregate_instances)
        ]
        removal_ranking = None
        if len(removal_methods) >= 2:
            removal_ranking = self._compute_rankings(
                aggregate_instances, sorted(removal_methods), overall_instances
            )
        if removal_ranking:
            lines.append("## 移除策略对比 (Friedman + Nemenyi)")
            lines.append(
                f"- 仅考虑移除策略不同的 HEA 变体（N={removal_ranking['num_instances']}），CD={removal_ranking['critical_difference']:.3f}"
            )
            lines.append("| 算法 | 平均排名 |")
            lines.append("| --- | --- |")
            for method, rank_value in sorted(
                removal_ranking["avg_ranks"].items(), key=lambda item: item[1]
            ):
                lines.append(f"| {method} | {rank_value:.3f} |")
            lines.append("")
            report["removal_strategies"] = {"ranking": removal_ranking}
            self._plot_cd_diagram(
                ranking_info=removal_ranking,
                title="HEA removal strategies (Nemenyi)",
                path=self.output_dir / "nemenyi_cd_removals.png",
            )
        (self.output_dir / "significance_report.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        (self.output_dir / "significance_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def generate_anytime_plots(self) -> None:
        target_instance = "medium_I1"
        target_methods = {
            r.method: r
            for r in self.records
            if r.instance_id == target_instance and r.stats is not None
        }
        description = ["# 搜索过程可视化", ""]
        if not target_methods:
            description.append("- 当前运行未捕获逐代统计数据，因此无法绘制曲线。")
        elif self.enable_plots:
            plt.figure(figsize=(7, 4))
            for method, record in target_methods.items():
                elapsed = [item[0] for item in record.stats]
                best = [item[1] for item in record.stats]
                plt.plot(elapsed, best, label=method)
            plt.xlabel("Runtime (s)")
            plt.ylabel("Best tardiness")
            plt.title("Anytime best-cost curve (Medium I1)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / "anytime_best_curve.png")
            plt.close()

            description.extend(
                [
                    "- `anytime_best_curve.png`: HEA-DRL、GA 与 MA 在代表性实例中的 anytime 性能。",
                ]
            )
        else:
            description.append(
                "- 若需生成 `anytime_best_curve.png` 与 `anytime_population_fitness.png`，请在本地运行"
                " `python analysis/experiment_suite.py --with-plots`。"
            )

        # 平均 anytime 曲线（按规模聚合）
        if self.enable_plots:
            avg_notes = []
            for scale in ("small", "medium", "large"):
                records = [r for r in self.records if r.scale == scale and r.stats is not None]
                if not records:
                    continue
                methods = sorted({r.method for r in records})
                # 构建统一时间网格
                max_time = max(max(t for t, _, _ in r.stats) for r in records)
                if max_time <= 0:
                    continue
                grid = np.linspace(0, max_time, 200)
                plt.figure(figsize=(7, 4))
                plotted = False
                for method in methods:
                    curves = []
                    for r in records:
                        if r.method != method:
                            continue
                        t = np.array([item[0] for item in r.stats])
                        y = np.array([item[1] for item in r.stats])
                        if len(t) < 2:
                            continue
                        curves.append(np.interp(grid, t, y, left=y[0], right=y[-1]))
                    if curves:
                        mean_y = np.mean(curves, axis=0)
                        plt.plot(grid, mean_y, label=method)
                        plotted = True
                if plotted:
                    plt.xlabel("Runtime (s)")
                    plt.ylabel("Best tardiness (avg over seeds)")
                    plt.title(f"Anytime average curves ({scale.title()})")
                    plt.legend()
                    plt.tight_layout()
                    fname = f"anytime_avg_{scale}.png"
                    plt.savefig(self.output_dir / fname)
                    plt.close()
                    avg_notes.append(f"- `{fname}`: {scale} 规模的跨种子平均 anytime 曲线。")
                else:
                    plt.close()
            if avg_notes:
                description.extend(["", "# 按规模平均 Anytime 曲线", ""] + avg_notes)
        (self.output_dir / "anytime_summary.md").write_text(
            "\n".join(description) + "\n", encoding="utf-8"
        )
        # 保存用于生成 anytime 曲线的原始统计数据
        stats_dump: List[dict] = []
        for r in self.records:
            if r.stats is None:
                continue
            stats_dump.append(
                {
                    "scale": r.scale,
                    "instance_id": r.instance_id,
                    "method": r.method,
                    "stats": [
                        {"time": t, "best": b, "avg": a} for (t, b, a) in r.stats
                    ],
                }
            )
        if stats_dump:
            (self.output_dir / "anytime_data.json").write_text(
                json.dumps(stats_dump, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def generate_transfer_summary(self) -> None:
        lines = ["# 策略迁移实验", ""]
        lines.append("| 问题 | 策略 | 总延迟 | 运行时间 (s) |")
        lines.append("| --- | --- | --- | --- |")
        for record in self.transfer_records:
            lines.append(
                f"| {record['problem']} | {record['method']} | {record['tardiness']:.1f} | {record['time_sec']:.2f} |"
            )
        lines.append("")
        if self.enable_plots:
            lines.append(
                "> 若本地启用了 `--with-plots`，会同时生成 `transfer_tardiness.png` 展示延迟与耗时对比。"
            )
        else:
            lines.append(
                "> 当前运行跳过了 `transfer_tardiness.png` 绘制，需在本地执行 `python analysis/experiment_suite.py --with-plots` 才会生成。"
            )
        (self.output_dir / "transfer_summary.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        (self.output_dir / "transfer_summary.json").write_text(
            json.dumps(self.transfer_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not self.transfer_records:
            return
        if not self.enable_plots:
            return

        problems = list(dict.fromkeys(record["problem"] for record in self.transfer_records))
        methods = list(dict.fromkeys(record["method"] for record in self.transfer_records))
        tardiness = {
            (rec["problem"], rec["method"]): rec["tardiness"] for rec in self.transfer_records
        }
        runtimes = {
            (rec["problem"], rec["method"]): rec["time_sec"] for rec in self.transfer_records
        }
        positions = np.arange(len(problems))
        width = 0.8 / max(1, len(methods))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for axis, metric_name, data in zip(
            axes,
            ["Total tardiness", "Runtime (s)"],
            [tardiness, runtimes],
        ):
            for idx, method in enumerate(methods):
                offsets = positions - 0.4 + width / 2 + idx * width
                axis.bar(
                    offsets,
                    [data.get((problem, method), 0.0) for problem in problems],
                    width=width,
                    label=method,
                )
            axis.set_xticks(positions)
            axis.set_xticklabels(problems)
            axis.set_title(metric_name)
        axes[0].set_ylabel("Value")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.2), ncol=len(methods))
        fig.suptitle("Transfer performance comparison")
        fig.tight_layout()
        plt.savefig(self.output_dir / "transfer_tardiness.png")
        plt.close(fig)

    def generate_runtime_breakdown(self) -> None:
        components = [
            "T_select_cross",
            "T_remove",
            "T_repair",
            "T_eval",
            "T_drl_infer",
        ]
        lines = ["# 运行时间与开销分析", ""]
        lines.append("| 规模 | 策略 | 总时间 (s) | 选择+交叉 | 移除 | 修复 | 评估 | DRL 推理 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        plot_data: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for scale in ("small", "medium", "large"):
            for method in [
                "Genetic Algorithm",
                "Memetic Algorithm",
                "HEA-DRL",
                "HEA-Random",
                "HEA-Shaw",
                "HEA-WorstSlack",
                "HEA-MaxDistance",
            ]:
                subset = [
                    r
                    for r in self.records
                    if r.scale == scale and r.method == method and r.timing
                ]
                if not subset:
                    continue
                totals = {
                    comp: statistics.mean([rec.timing.get(comp, 0.0) for rec in subset])
                    for comp in components
                }
                total_time = statistics.mean([rec.timing.get("T_total", 0.0) for rec in subset])
                plot_data[scale][method] = {**totals, "T_total": total_time}
                def fmt(name: str) -> str:
                    value = totals.get(name, 0.0)
                    return f"{value:.2f} ({value / total_time:.1%})" if total_time else "0"

                lines.append(
                    f"| {scale} | {method} | {total_time:.2f} | {fmt('T_select_cross')} | {fmt('T_remove')} | "
                    f"{fmt('T_repair')} | {fmt('T_eval')} | {fmt('T_drl_infer')} |"
                )
        lines.append("")
        if self.enable_plots:
            lines.append(
                "> 启用 `--with-plots` 时会额外输出 `runtime_breakdown.png` 展示堆叠柱状图。"
            )
        else:
            lines.append(
                "> 当前运行未生成 `runtime_breakdown.png`，可在本地使用 `python analysis/experiment_suite.py --with-plots` 来绘制。"
            )
        (self.output_dir / "runtime_breakdown.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        if self.enable_plots and plot_data:
            scales = list(plot_data.keys())
            fig, axes = plt.subplots(
                1, len(scales), figsize=(5 * len(scales), 4), sharey=False
            )
            if len(scales) == 1:
                axes = [axes]
            for ax, scale in zip(axes, scales):
                methods = list(plot_data[scale].keys())
                positions = np.arange(len(methods))
                left = np.zeros(len(methods))
                for comp in components:
                    values = np.array(
                        [plot_data[scale][method].get(comp, 0.0) for method in methods]
                    )
                    ax.barh(positions, values, left=left, label=comp)
                    left += values
                ax.set_title(f"{scale.title()} instances")
                ax.set_yticks(positions)
                ax.set_yticklabels(methods)
                ax.invert_yaxis()
                ax.set_xlabel("Runtime (s)")
            axes[-1].legend(loc="upper left", bbox_to_anchor=(1.05, 1.02))
            fig.tight_layout()
            plt.savefig(self.output_dir / "runtime_breakdown.png")
            plt.close()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _best_reference(self) -> Dict[tuple[str, str], float]:
        best: Dict[tuple[str, str], float] = {}
        for record in self.records:
            key = (record.scale, record.instance_id)
            best[key] = min(best.get(key, math.inf), record.tardiness)
        return best

    def _compute_rankings(
        self,
        per_instance: Dict[str, Dict[str, float]],
        methods: Sequence[str],
        instances: Sequence[str],
        alpha: float = 0.05,
    ) -> dict | None:
        if len(instances) < 2 or len(methods) < 2:
            return None
        rank_sums = {method: 0.0 for method in methods}
        for inst in instances:
            values = [per_instance[inst][method] for method in methods]
            ranks = stats.rankdata(values, method="average")
            for method, rank in zip(methods, ranks):
                rank_sums[method] += float(rank)
        num_instances = len(instances)
        avg_ranks = {method: rank_sums[method] / num_instances for method in methods}
        k = len(methods)
        if num_instances == 0:
            return None
        q_alpha = stats.studentized_range.ppf(1 - alpha, k, 10_000)
        if math.isnan(q_alpha):
            return None
        cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * num_instances))
        return {
            "avg_ranks": avg_ranks,
            "num_instances": num_instances,
            "critical_difference": cd,
        }

    def _plot_cd_diagram(self, ranking_info: dict, title: str, path: pathlib.Path) -> None:
        if plt is None:
            return
        avg_ranks: Dict[str, float] = ranking_info.get("avg_ranks", {})
        if not avg_ranks:
            return
        cd = ranking_info.get("critical_difference")
        if not cd:
            return
        sorted_methods = sorted(avg_ranks.items(), key=lambda item: item[1])
        ranks = [rank for _, rank in sorted_methods]
        labels = [method for method, _ in sorted_methods]
        fig_height = 1.5 + 0.6 * len(labels)
        fig, ax = plt.subplots(figsize=(9, fig_height))
        min_rank = min(ranks) - 0.3
        max_rank = max(ranks) + 0.3
        ax.set_xlim(min_rank, max_rank)
        ax.set_xticks(np.arange(1, len(sorted_methods) + 1))
        ax.set_xlabel("Average rank (lower is better)")
        ax.set_yticks([])
        raw_segments = []
        for i in range(len(sorted_methods)):
            start = i
            end = i
            for j in range(i + 1, len(sorted_methods)):
                if ranks[j] - ranks[start] <= cd + 1e-9:
                    end = j
                else:
                    break
            if end > start:
                raw_segments.append((start, end))
        connectors = []
        for seg in raw_segments:
            if not any(
                other is not seg
                and other[0] <= seg[0]
                and other[1] >= seg[1]
                for other in raw_segments
            ):
                connectors.append(seg)
        connector_levels = max(1, len(connectors))
        connectors_base = len(labels) + 0.3
        cd_y = connectors_base + 0.3 * connector_levels
        top_limit = cd_y + 0.5
        ax.set_ylim(-0.5, top_limit)
        ax.hlines(len(labels) + 0.1, min_rank, max_rank, color="black", linewidth=1)
        for idx, (method, rank_value) in enumerate(sorted_methods):
            y = len(labels) - idx - 0.5
            ax.plot(rank_value, y, "o", color="black")
            ax.text(
                rank_value + 0.1,
                y,
                f"{method} ({rank_value:.2f})",
                va="center",
                fontsize=10,
            )
        for level, (start_idx, end_idx) in enumerate(connectors):
            y = connectors_base + level * 0.3
            ax.plot(
                [ranks[start_idx], ranks[end_idx]],
                [y, y],
                color="gray",
                linewidth=3,
                alpha=0.8,
            )
        ax.plot(
            [min_rank, min_rank + cd],
            [cd_y, cd_y],
            color="black",
            linewidth=2,
        )
        ax.text(min_rank + cd / 2, cd_y + 0.15, f"CD = {cd:.3f}", ha="center")
        ax.set_title(title)
        fig.tight_layout()
        plt.savefig(path)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HEA-DRL experiment suite")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=OUTPUT_DIR,
        help="目录，用于保存分析产物",
    )
    parser.add_argument(
        "--with-plots",
        action="store_true",
        help="本地运行时启用，以生成 PNG 图像 (CI/在线环境默认跳过)",
    )
    parser.add_argument(
        "--time-budget-scale",
        type=float,
        default=1.0,
        help="按比例缩放各规模的时间预算，<1 用于快速调试",
    )
    parser.add_argument(
        "--seed-cap",
        type=int,
        default=None,
        help="可选：限制每个规模的种子数量，用于快速调试",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="快捷模式：时间预算缩放为 0.001，种子数限制为 1，便于快速回归",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=None,
        help="可选：仅运行指定规模（例：--scales small medium）",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="可选：仅运行指定算法名称（大小写不敏感，示例：--methods HEA-DRL GA VIGA）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    time_budget_scale = 0.001 if args.fast else args.time_budget_scale
    seed_cap = 1 if args.fast else args.seed_cap
    suite = ExperimentSuite(
        output_dir=args.output,
        enable_plots=args.with_plots,
        time_budget_scale=time_budget_scale,
        seed_cap=seed_cap,
        scales_filter=args.scales,
        methods_filter=args.methods,
    )
    suite.run()


if __name__ == "__main__":
    main()
