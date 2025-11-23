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
    elapsed_sec: float


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
        self.using_drl = self.config.removal_strategy.lower() in {"drl", "drl_hybrid"}

    def _create_removal_strategy(self) -> RemovalStrategy:
        name = self.config.removal_strategy.lower()
        if name in {"drl", "drl_hybrid"}:
            if not self.agent:
                raise ValueError("DRL strategy requires an agent")
            return build_removal_strategy(name, self.instance, agent=self.agent)
        if name == "random":
            return build_removal_strategy(
                "random",
                self.instance,
                seed=self.config.seed if hasattr(self.config, "seed") else None,
            )
        return build_removal_strategy("shaw", self.instance)

    def initialize_population(self) -> List[Solution]:
        population: List[Solution] = []
        population.append(random_solution(self.instance, seed=self.rng.randint(0, 1_000_000)))
        population.append(random_solution(self.instance, seed=self.rng.randint(0, 1_000_000)))
        while len(population) < self.config.population_size:
            population.append(random_solution(self.instance, seed=self.rng.randint(0, 1_000_000)))
        return population

    def _nearest_neighbor_solution(self) -> Solution:
        remaining = set(range(len(self.instance.tasks)))
        order: List[int] = []
        current: int | None = None
        while remaining:
            def distance_to(idx: int) -> float:
                if current is None:
                    return self.instance.distance(None, self.instance.tasks[idx])
                return self.instance.distance(self.instance.tasks[current], self.instance.tasks[idx])

            next_idx = min(remaining, key=distance_to)
            order.append(next_idx)
            remaining.remove(next_idx)
            current = next_idx
        return Solution.from_order(self.instance, order)

    def _sorted_by_depot_solution(self) -> Solution:
        order = sorted(
            range(len(self.instance.tasks)),
            key=lambda idx: self.instance.distance(None, self.instance.tasks[idx]),
        )
        return Solution.from_order(self.instance, order)

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
        child_order: List[int | None] = [None] * size
        child_order[start:end] = parent_a.order[start:end]
        fill = [idx for idx in parent_b.order if idx not in child_order[start:end]]
        pointer = 0
        all_ids = set(range(len(self.instance.tasks)))
        pb_len = len(parent_b.order)
        for i in range(size):
            if child_order[i] is None:
                if pointer < len(fill):
                    child_order[i] = fill[pointer]
                    pointer += 1
                else:
                    remaining = all_ids - set(x for x in child_order if x is not None)
                    if remaining:
                        child_order[i] = self.rng.choice(list(remaining))
                    else:
                        # Fallback: reuse a random id to keep array filled
                        child_order[i] = parent_b.order[self.rng.randrange(pb_len)]
        # type: ignore[arg-type]
        return Solution.from_order(self.instance, [idx for idx in child_order if idx is not None])

    def mutate(self, solution: Solution) -> None:
        if self.rng.random() < self.config.mutation_rate:
            order = solution.order
            a, b = self.rng.sample(range(len(order)), 2)
            order[a], order[b] = order[b], order[a]
            solution.routes = Solution.from_order(self.instance, order).routes

    def local_search(self, solution: Solution) -> dict[str, float]:
        timing: dict[str, float] = {"remove": 0.0, "repair": 0.0, "drl_infer": 0.0}
        remove_start = time.perf_counter()
        selected = self.removal_strategy.select(solution.order, self.config.remove_count)
        timing["remove"] = time.perf_counter() - remove_start
        if self.using_drl:
            timing["drl_infer"] = timing["remove"]
        removed_tasks = solution.remove_tasks(selected)
        repair_start = time.perf_counter()
        greedy_repair(solution, removed_tasks)
        timing["repair"] = time.perf_counter() - repair_start
        return timing

    def run(self) -> HEAResult:
        timing: dict[str, float] = {}
        start = time.perf_counter()
        population = self.initialize_population()
        timing["T_init"] = time.perf_counter() - start
        eval_budget = self.config.eval_budget
        time_budget = self.config.time_budget
        eval_count = 0
        stats: List[GenerationStats] = []
        for gen in range(self.config.generations):
            if time_budget is not None and time.perf_counter() - start >= time_budget:
                self.logger.info("Stopping at generation %s due to time budget %.2fs", gen, time_budget)
                break
            gen_start = time.perf_counter()
            eval_start = time.perf_counter()
            costs, best_cost, avg_cost = self.evaluate_population(population)
            eval_count += len(population)
            eval_time = time.perf_counter() - eval_start
            timing.setdefault("T_eval", 0.0)
            timing["T_eval"] += eval_time
            elapsed = time.perf_counter() - start
            stats.append(
                GenerationStats(
                    generation=gen,
                    best_cost=best_cost,
                    average_cost=avg_cost,
                    elapsed_sec=elapsed,
                )
            )
            self.logger.info("Gen %s: best=%.3f avg=%.3f", gen, best_cost, avg_cost)

            elites = self._select_elites(population, costs)
            children: List[Solution] = []
            select_cross_time = 0.0
            remove_time = 0.0
            repair_time = 0.0
            drl_time = 0.0
            while len(children) < self.config.population_size - len(elites):
                op_start = time.perf_counter()
                parent_a, parent_b = self.select_parents(population, costs)
                if self.rng.random() < self.config.crossover_rate:
                    child = self.crossover(parent_a, parent_b)
                else:
                    child = parent_a.copy()
                self.mutate(child)
                select_cross_time += time.perf_counter() - op_start

                ls_timing = self.local_search(child)
                remove_time += ls_timing["remove"]
                repair_time += ls_timing["repair"]
                drl_time += ls_timing["drl_infer"]
                children.append(child)
            population = elites + children
            timing.setdefault("T_select_cross", 0.0)
            timing["T_select_cross"] += select_cross_time
            timing.setdefault("T_remove", 0.0)
            timing["T_remove"] += remove_time
            timing.setdefault("T_repair", 0.0)
            timing["T_repair"] += repair_time
            if drl_time:
                timing.setdefault("T_drl_infer", 0.0)
                timing["T_drl_infer"] += drl_time
            gen_time = time.perf_counter() - gen_start
            timing.setdefault("T_generations", 0.0)
            timing["T_generations"] += gen_time
            self.logger.info(
                "[Time] Generation %s: Select+Cross=%.4fs, Remove=%.4fs, Repair=%.4fs, Eval=%.4fs, TotalGen=%.4fs",
                gen,
                select_cross_time,
                remove_time,
                repair_time,
                eval_time,
                gen_time,
            )
            if eval_budget is not None and eval_count >= eval_budget:
                self.logger.info(
                    "Stopping at generation %s due to evaluation budget (used %s / %s)",
                    gen,
                    eval_count,
                    eval_budget,
                )
                break

        eval_start = time.perf_counter()
        final_costs, _, _ = self.evaluate_population(population)
        eval_count += len(population)
        timing.setdefault("T_eval", 0.0)
        timing["T_eval"] += time.perf_counter() - eval_start
        best_idx = min(range(len(population)), key=lambda i: final_costs[i])
        best_solution = population[best_idx]
        total_runtime = time.perf_counter() - start
        timing["T_total"] = total_runtime
        self.logger.info(
            "[Time] Total runtime: %.3fs (Init=%.3fs, Select+Cross=%.3fs, Remove=%.3fs, Repair=%.3fs, Eval=%.3fs, DRL_infer=%.3fs)",
            total_runtime,
            timing.get("T_init", 0.0),
            timing.get("T_select_cross", 0.0),
            timing.get("T_remove", 0.0),
            timing.get("T_repair", 0.0),
            timing.get("T_eval", 0.0),
            timing.get("T_drl_infer", 0.0),
        )
        return HEAResult(best_solution=best_solution, stats=stats, timing=timing)

    def _select_elites(self, population: List[Solution], costs: List[float]) -> List[Solution]:
        elite_count = max(1, math.ceil(self.config.population_size * self.config.elite_rate))
        ranked = sorted(zip(population, costs), key=lambda pc: pc[1])
        return [sol.copy() for sol, _ in ranked[:elite_count]]
