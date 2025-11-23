#!/usr/bin/env python3
"""Collect sequence-level teacher data (routes + removal sets) from MA/VIGA."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import List, Sequence, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import AlgorithmConfig, ProblemConfig  # noqa: E402
from src.data import ProblemInstance, generate_random_instance  # noqa: E402
from src.drl_agent import DRLAgent  # noqa: E402
from src.hea import HEA  # noqa: E402
from src.removal import build_removal_strategy, RemovalStrategy  # noqa: E402
from src.solution import Solution, random_solution  # noqa: E402


class CapturingRemoval(RemovalStrategy):
    def __init__(self, base: RemovalStrategy, instance: ProblemInstance, agent: DRLAgent, records: list, method: str) -> None:
        super().__init__(instance)
        self.base = base
        self.agent = agent
        self.records = records
        self.method = method

    def select(self, solution_order: List[int], num_remove: int) -> List[int]:
        selected = self.base.select(solution_order, num_remove)
        total = len(solution_order)
        feats: List[List[float]] = []
        for pos, idx in enumerate(solution_order):
            feats.append(self.agent.features(self.instance, self.instance.tasks[idx], pos, total, solution_order))
        self.records.append(
            {
                "order": list(solution_order),
                "removed": list(selected),
                "method": self.method,
                "features": feats,
            }
        )
        return selected


def nearest_neighbor_solution(instance: ProblemInstance) -> Solution:
    remaining = set(range(len(instance.tasks)))
    order: List[int] = []
    current: int | None = None
    while remaining:
        def distance_to(idx: int) -> float:
            if current is None:
                return instance.distance(None, instance.tasks[idx])
            return instance.distance(instance.tasks[current], instance.tasks[idx])

        nxt = min(remaining, key=distance_to)
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return Solution.from_order(instance, order)


def build_instances() -> List[Tuple[ProblemInstance, int]]:
    scales = [
        ("small", 10, "uniform", 3, [11, 13, 17, 19]),
        ("medium", 30, "uniform", 4, [101, 103, 105, 107]),
        ("large", 50, "clustered", 5, [201, 203, 205, 207]),
    ]
    batches: List[Tuple[ProblemInstance, int]] = []
    for _, n_tasks, dist, remove_count, seeds in scales:
        for seed in seeds:
            cfg = ProblemConfig(
                instance_type="random",
                num_tasks=n_tasks,
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


def run_teacher(
    instance: ProblemInstance,
    remove_count: int,
    method: str,
    records: list,
    agent: DRLAgent,
) -> None:
    if method == "ma":
        cfg = AlgorithmConfig(
            removal_strategy="shaw",
            remove_count=remove_count,
            population_size=32,
            generations=20,
            crossover_rate=0.85,
            mutation_rate=0.2,
            elite_rate=0.1,
            seed=42,
        )
    elif method == "viga":
        cfg = AlgorithmConfig(
            removal_strategy="random",
            remove_count=remove_count,
            population_size=36,
            generations=20,
            crossover_rate=0.9,
            mutation_rate=0.25,
            elite_rate=0.15,
            seed=84,
        )
    else:
        raise ValueError("Unknown teacher")
    hea = HEA(instance, cfg, logger=None, agent=None)  # type: ignore[arg-type]
    base = build_removal_strategy(cfg.removal_strategy, instance, seed=cfg.seed if hasattr(cfg, "seed") else None)
    hea.removal_strategy = CapturingRemoval(base, instance, agent, records, method)

    def warm_population(pop_size: int) -> List[Solution]:
        pop: List[Solution] = []
        pop.append(nearest_neighbor_solution(instance))
        pop.append(random_solution(instance, seed=cfg.seed))
        while len(pop) < pop_size:
            pop.append(random_solution(instance, seed=random.randint(0, 1_000_000)))
        return pop

    population = warm_population(cfg.population_size)
    # simple evolution with capture
    for _ in range(cfg.generations):
        costs = [sol.cost() for sol in population]
        ranked = [sol for sol, _ in sorted(zip(population, costs), key=lambda p: p[1])]
        elites = ranked[: max(1, cfg.population_size // 4)]
        children: List[Solution] = []
        while len(children) + len(elites) < cfg.population_size:
            parent = random.choice(elites).copy()
            a, b = random.sample(range(len(parent.order)), 2)
            parent.order[a], parent.order[b] = parent.order[b], parent.order[a]
            hea.local_search(parent)
            children.append(parent)
        population = elites + children


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "data/drl_teacher_sequences.json")
    args = parser.parse_args()
    agent = DRLAgent()
    records: list = []
    for instance, remove_count in build_instances():
        for method in ("ma", "viga"):
            run_teacher(instance, remove_count, method, records, agent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records), encoding="utf-8")
    print(f"Saved {len(records)} sequence samples to {args.output}")


if __name__ == "__main__":
    main()
