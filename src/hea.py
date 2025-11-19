from __future__ import annotations

import dataclasses
import logging
import math
import random
import time
from typing import List, Tuple

from .config import AlgorithmConfig
from .data import ProblemInstance
from .removal import build_removal_strategy, RemovalStrategy
from .repair import greedy_repair
from .solution import Solution, random_solution
from .drl_agent import DRLAgent


@dataclasses.dataclass
class GenerationStats:
    generation: int
    best_cost: float
    average_cost: float


@dataclasses.dataclass
class HEAResult:
    best_solution: Solution
    stats: List[GenerationStats]
    timing: dict[str, float]


class HEA:
    def __init__(self, instance: ProblemInstance, config: AlgorithmConfig, logger: logging.Logger, agent: DRLAgent | None = None) -> None:
        self.instance = instance
        self.config = config
        self.logger = logger
        self.agent = agent
        self.removal_strategy = self._create_removal_strategy()
        self.rng = random.Random(config.seed if hasattr(config, "seed") else None)

    def _create_removal_strategy(self) -> RemovalStrategy:
        if self.config.removal_strategy.lower() == "drl":
            if not self.agent:
                raise ValueError("DRL strategy requires an agent")
            return build_removal_strategy("drl", self.instance, agent=self.agent)
        if self.config.removal_strategy.lower() == "random":
            return build_removal_strategy("random", self.instance, seed=self.config.seed if hasattr(self.config, "seed") else None)
        return build_removal_strategy("shaw", self.instance)

    def initialize_population(self) -> List[Solution]:
        population = [random_solution(self.instance, seed=self.rng.randint(0, 1_000_000)) for _ in range(self.config.population_size)]
        return population

    def evaluate_population(self, population: List[Solution]) -> Tuple[List[float], float, float]:
        costs = [sol.cost() for sol in population]
        best_cost = min(costs)
        avg_cost = sum(costs) / len(costs)
        return costs, best_cost, avg_cost

    def select_parents(self, population: List[Solution], costs: List[float]) -> Tuple[Solution, Solution]:
        tournament_size = 3
        def tournament() -> Solution:
            participants = self.rng.sample(list(zip(population, costs)), tournament_size)
            return min(participants, key=lambda pc: pc[1])[0]
        return tournament(), tournament()

    def crossover(self, parent_a: Solution, parent_b: Solution) -> Solution:
        size = len(parent_a.order)
        start, end = sorted(self.rng.sample(range(size), 2))
        child_order = [None] * size
        child_order[start:end] = parent_a.order[start:end]
        fill = [idx for idx in parent_b.order if idx not in child_order[start:end]]
        pointer = 0
        for i in range(size):
            if child_order[i] is None:
                child_order[i] = fill[pointer]
                pointer += 1
        return Solution(self.instance, child_order)  # type: ignore[arg-type]

    def mutate(self, solution: Solution) -> None:
        if self.rng.random() < self.config.mutation_rate:
            a, b = self.rng.sample(range(len(solution.order)), 2)
            solution.order[a], solution.order[b] = solution.order[b], solution.order[a]

    def local_search(self, solution: Solution) -> None:
        selected = self.removal_strategy.select(solution.order, self.config.remove_count)
        removed_tasks = solution.remove_tasks(selected)
        greedy_repair(solution, removed_tasks)

    def run(self) -> HEAResult:
        timing: dict[str, float] = {}
        start = time.perf_counter()
        population = self.initialize_population()
        timing["T_init"] = time.perf_counter() - start
        stats: List[GenerationStats] = []
        for gen in range(self.config.generations):
            gen_start = time.perf_counter()
            costs, best_cost, avg_cost = self.evaluate_population(population)
            stats.append(GenerationStats(generation=gen, best_cost=best_cost, average_cost=avg_cost))
            self.logger.info("Gen %s: best=%.3f avg=%.3f", gen, best_cost, avg_cost)

            elites = self._select_elites(population, costs)
            children: List[Solution] = []
            while len(children) < self.config.population_size - len(elites):
                parent_a, parent_b = self.select_parents(population, costs)
                if self.rng.random() < self.config.crossover_rate:
                    child = self.crossover(parent_a, parent_b)
                else:
                    child = parent_a.copy()
                self.mutate(child)
                ls_start = time.perf_counter()
                self.local_search(child)
                timing.setdefault("T_local", 0.0)
                timing["T_local"] += time.perf_counter() - ls_start
                children.append(child)
            population = elites + children
            timing.setdefault("T_generations", 0.0)
            timing["T_generations"] += time.perf_counter() - gen_start

        final_costs, _, _ = self.evaluate_population(population)
        best_idx = min(range(len(population)), key=lambda i: final_costs[i])
        best_solution = population[best_idx]
        timing["T_total"] = sum(timing.values()) + timing.get("T_init", 0.0)
        return HEAResult(best_solution=best_solution, stats=stats, timing=timing)

    def _select_elites(self, population: List[Solution], costs: List[float]) -> List[Solution]:
        elite_count = max(1, math.ceil(self.config.population_size * self.config.elite_rate))
        ranked = sorted(zip(population, costs), key=lambda pc: pc[1])
        return [sol.copy() for sol, _ in ranked[:elite_count]]
