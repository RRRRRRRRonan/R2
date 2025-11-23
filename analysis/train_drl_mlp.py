#!/usr/bin/env python3
"""Train a small MLP removal scorer via imitation of Shaw on constructive solutions."""

from __future__ import annotations

import json
import math
import pathlib
import random
import sys
from typing import List, Sequence, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import ProblemConfig  # noqa: E402
from src.data import ProblemInstance, generate_random_instance  # noqa: E402
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
    for path in [
        ROOT / "data/vrptw_instance.json",
        ROOT / "data/evrptw_instance.json",
        ROOT / "data/solomon/C101_sample.json",
    ]:
        if path.exists():
            cfg = ProblemConfig(data_file=str(path))
            batches.append((generate_random_instance(cfg), 6))
    return batches


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden: int = 32,
    epochs: int = 200,
    lr: float = 1e-2,
    batch: int = 256,
    reg: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n, d = X.shape
    W1 = rng.standard_normal((d, hidden)) * 0.1
    b1 = np.zeros((hidden,))
    W2 = rng.standard_normal((hidden, 1)) * 0.1
    b2 = np.zeros((1,))

    def batch_iter():
        idx = np.arange(n)
        rng.shuffle(idx)
        for start in range(0, n, batch):
            sel = idx[start : start + batch]
            yield X[sel], y[sel][:, None]

    for _ in range(epochs):
        for xb, yb in batch_iter():
            h = relu(xb @ W1 + b1)
            logits = h @ W2 + b2
            preds = 1 / (1 + np.exp(-logits))
            # binary cross entropy
            loss_grad = preds - yb
            loss_grad /= xb.shape[0]
            # grads
            dW2 = h.T @ loss_grad + reg * W2
            db2 = loss_grad.sum(axis=0)
            dh = loss_grad @ W2.T
            dh[h <= 0] = 0
            dW1 = xb.T @ dh + reg * W1
            db1 = dh.sum(axis=0)
            # update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
    return W1, b1, W2, b2


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
    W1, b1, W2, b2 = train_mlp(X_mat, y_vec, hidden=32, epochs=300, lr=5e-3, batch=256, reg=1e-4)
    model_path = ROOT / "models" / "drl_model.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mlp": {
            "w1": W1.tolist(),
            "b1": b1.tolist(),
            "w2": W2.tolist(),
            "b2": b2.tolist(),
        }
    }
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Trained MLP saved to {model_path}")
    print(f"Dataset size: {len(y_all)} samples")


if __name__ == "__main__":
    main()
