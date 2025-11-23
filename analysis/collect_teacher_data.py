#!/usr/bin/env python3
"""Collect removal labels from MA (Shaw) and VIGA (random) as teacher data."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Sequence, Tuple
import random

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import AlgorithmConfig, ProblemConfig  # noqa: E402
from src.data import ProblemInstance, generate_random_instance  # noqa: E402
from src.drl_agent import DRLAgent  # noqa: E402
from src.hea import HEA  # noqa: E402
from src.removal import build_removal_strategy, RemovalStrategy  # noqa: E402
from src.solution import Solution, random_solution  # noqa: E402


class LoggingRemoval(RemovalStrategy):
    def __init__(self, base: RemovalStrategy, instance: ProblemInstance, agent: DRLAgent, log_X: List[list], log_y: List[int]) -> None:
        super().__init__(instance)
        self.base = base
        self.agent = agent
        self.log_X = log_X
        self.log_y = log_y

    def select(self, solution_order: List[int], num_remove: int) -> List[int]:
        selected = set(self.base.select(solution_order, num_remove))
        total = len(solution_order)
        for pos, idx in enumerate(solution_order):
            feats = self.agent.features(self.instance, self.instance.tasks[idx], pos, total, solution_order)
            self.log_X.append(feats)
            self.log_y.append(1 if idx in selected else 0)
        return list(selected)


def nearest_neighbor_solution(instance: ProblemInstance) -> Solution:
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
    return Solution.from_order(instance, order)


def build_instances() -> List[Tuple[ProblemInstance, int]]:
    scales = [
        ("small", 10, "uniform", 3, [11, 13, 17, 19]),
        ("medium", 30, "uniform", 4, [101, 103, 105, 107]),
        ("large", 50, "clustered", 5, [201, 203, 205, 207]),
    ]
    batches: List[Tuple[ProblemInstance, int]] = []
    for _, num_tasks, dist, remove_count, seeds in scales:
        for seed in seeds:
            cfg = ProblemConfig(
                instance_type="random",
                num_tasks=num_tasks,
                distribution=dist,
                seed=seed,
            )
            batches.append((generate_random_instance(cfg), remove_count))
    for path in [
        ROOT / "data/vrptw_instance.json",
        ROOT / "data/evrptw_instance.json",
        ROOT / "data/solomon/C101_sample.json",
    ]:
        if path.exists():
            cfg = ProblemConfig(data_file=str(path))
            batches.append((generate_random_instance(cfg), 6))
    return batches


def collect_for_method(method: str, instance: ProblemInstance, remove_count: int, agent: DRLAgent) -> Tuple[List[list], List[int]]:
    log_X: List[list] = []
    log_y: List[int] = []
    if method == "ma":
        cfg = AlgorithmConfig(
            removal_strategy="shaw",
            remove_count=remove_count,
            population_size=32,
            generations=30,
            crossover_rate=0.85,
            mutation_rate=0.2,
            elite_rate=0.1,
            model_path=None,
            seed=42,
        )
    elif method == "viga":
        cfg = AlgorithmConfig(
            removal_strategy="random",
            remove_count=remove_count,
            population_size=36,
            generations=30,
            crossover_rate=0.9,
            mutation_rate=0.25,
            elite_rate=0.15,
            model_path=None,
            seed=84,
        )
    else:
        raise ValueError("Unknown method")
    hea = HEA(instance, cfg, logger=None, agent=None)  # type: ignore[arg-type]
    base = build_removal_strategy(cfg.removal_strategy, instance, seed=cfg.seed if hasattr(cfg, "seed") else None)
    hea.removal_strategy = LoggingRemoval(base, instance, agent, log_X, log_y)
    # Warm start a few heuristic solutions for diversity
    def warm_population(pop_size: int) -> List[Solution]:
        pop: List[Solution] = []
        pop.append(nearest_neighbor_solution(instance))
        pop.append(random_solution(instance, seed=cfg.seed))
        while len(pop) < pop_size:
            pop.append(random_solution(instance, seed=random.randint(0, 1_000_000)))
        return pop
    population = warm_population(cfg.population_size)
    for _ in range(cfg.generations):
        # simple evolution: evaluate costs, select best half, mutate/crossover rudimentarily
        costs = [sol.cost() for sol in population]
        ranked = [sol for sol, _ in sorted(zip(population, costs), key=lambda p: p[1])]
        elites = ranked[: max(1, cfg.population_size // 4)]
        children: List[Solution] = []
        while len(children) + len(elites) < cfg.population_size:
            parent = random.choice(elites).copy()
            # light shuffle
            a, b = random.sample(range(len(parent.order)), 2)
            parent.order[a], parent.order[b] = parent.order[b], parent.order[a]
            hea.local_search(parent)
            children.append(parent)
        population = elites + children
    return log_X, log_y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "data/drl_teacher_ma_viga.npz")
    args = parser.parse_args()
    agent = DRLAgent()
    X_all: List[list] = []
    y_all: List[int] = []
    for instance, remove_count in build_instances():
        for method in ("ma", "viga"):
            X, y = collect_for_method(method, instance, remove_count, agent)
            X_all.extend(X)
            y_all.extend(y)
    X_mat = np.array(X_all, dtype=np.float32)
    y_vec = np.array(y_all, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=X_mat, y=y_vec)
    print(f"Saved teacher data to {args.output} (samples={len(y_vec)})")


if __name__ == "__main__":
    main()
