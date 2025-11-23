#!/usr/bin/env python3
"""Offline training for the DRL removal strategy.

We approximate teacher behaviour (Shaw removal on constructive solutions)
and fit a simple linear model (ridge regression) to rank tasks.
"""

from __future__ import annotations

import json
import pathlib
from typing import List, Sequence, Tuple

import numpy as np

import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import ProblemInstance, Task, generate_random_instance  # noqa: E402
from src.config import ProblemConfig  # noqa: E402
from src.drl_agent import DRLAgent  # noqa: E402
from src.removal import ShawRemovalStrategy  # noqa: E402


def nearest_neighbor_order(instance: ProblemInstance) -> List[int]:
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


def collect_samples(
    instance: ProblemInstance, remove_count: int, agent: DRLAgent
) -> Tuple[List[List[float]], List[int]]:
    order = nearest_neighbor_order(instance)
    teacher = ShawRemovalStrategy(instance)
    selected = set(teacher.select(order, remove_count))
    X: List[List[float]] = []
    y: List[int] = []
    total = len(order)
    for pos, idx in enumerate(order):
        feats = agent.features(instance, instance.tasks[idx], pos, total)
        X.append(feats)
        y.append(1 if idx in selected else 0)
    return X, y


def build_instances() -> List[Tuple[ProblemInstance, int]]:
    scales = [
        ("small", 10, "uniform", 3, [11, 13, 17, 19, 23, 29, 31, 37]),
        ("medium", 30, "uniform", 4, [101, 103, 105, 107, 109, 111, 113, 127]),
        ("large", 50, "clustered", 5, [201, 203, 205, 207, 209, 211, 219, 223]),
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
    # add real-world style instances if present
    for path in [
        ROOT / "data/vrptw_instance.json",
        ROOT / "data/evrptw_instance.json",
        ROOT / "data/solomon/C101_sample.json",
    ]:
        if path.exists():
            cfg = ProblemConfig(data_file=str(path))
            batches.append((generate_random_instance(cfg), 6))
    return batches


def train_ridge(X: np.ndarray, y: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    # Center features for better conditioning
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    XtX = Xc.T @ Xc
    XtX += reg * np.eye(XtX.shape[0])
    Xty = Xc.T @ y
    w_centered = np.linalg.solve(XtX, Xty)
    # Adjust bias term via mean; bias feature is the last dimension (value 1.0)
    bias = y.mean() - float((mean @ w_centered).squeeze())
    w = w_centered.copy()
    w[-1] += bias
    return w


def main() -> None:
    agent = DRLAgent()
    X_all: List[List[float]] = []
    y_all: List[int] = []
    for instance, remove_count in build_instances():
        X, y = collect_samples(instance, remove_count, agent)
        X_all.extend(X)
        y_all.extend(y)
    X_mat = np.array(X_all, dtype=np.float64)
    y_vec = np.array(y_all, dtype=np.float64)
    weights = train_ridge(X_mat, y_vec, reg=1e-3)
    model_path = ROOT / "models" / "drl_model.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps({"weights": weights.tolist()}, indent=2), encoding="utf-8")
    print(f"Trained weights saved to {model_path}")
    print("Weights:", weights.tolist())
    print(f"Dataset size: {len(y_all)} samples")


if __name__ == "__main__":
    main()
