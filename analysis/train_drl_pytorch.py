#!/usr/bin/env python3
"""Train a small MLP removal scorer via imitation of MA/VIGA (if available) or Shaw fallback."""

from __future__ import annotations

import json
import pathlib
import random
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import ProblemConfig  # noqa: E402
from src.data import ProblemInstance, generate_random_instance  # noqa: E402
from src.drl_agent import DRLAgent  # noqa: E402
from src.removal import ShawRemovalStrategy  # noqa: E402
from src.solution import Solution, random_solution  # noqa: E402


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
    instance: ProblemInstance,
    remove_count: int,
    agent: DRLAgent,
    teacher: str = "shaw",
) -> Tuple[List[List[float]], List[int]]:
    order = nearest_neighbor_order(instance)
    if teacher == "shaw":
        removal = ShawRemovalStrategy(instance)
        selected = set(removal.select(order, remove_count))
    else:
        removal = ShawRemovalStrategy(instance)
        selected = set(removal.select(order, remove_count))
    X: List[List[float]] = []
    y: List[int] = []
    total = len(order)
    for pos, idx in enumerate(order):
        feats = agent.features(instance, instance.tasks[idx], pos, total, order)
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


class MLP(nn.Module):
    def __init__(self, input_dim: int = 7, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(
    X: torch.Tensor,
    y: torch.Tensor,
    hidden: int = 64,
    epochs: int = 10,
    batch: int = 512,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
) -> tuple[MLP, list[float], list[float]]:
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = MLP(input_dim=X.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    accs: list[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in dl:
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(xb)
            preds = (logits.detach() > 0).float()
            correct += int((preds == yb).sum().item())
            total += len(xb)
        losses.append(epoch_loss / max(1, len(ds)))
        accs.append(correct / max(1, total))
    return model, losses, accs


def main() -> None:
    random.seed(42)
    agent = DRLAgent()
    teacher_path = ROOT / "data/drl_teacher_ma_viga.npz"
    if teacher_path.exists():
        data = np.load(teacher_path)
        X_np = data["X"]
        y_np = data["y"]
        # subsample if very large to speed up training
        max_samples = 200_000
        if len(y_np) > max_samples:
            idx = np.random.default_rng(42).choice(len(y_np), size=max_samples, replace=False)
            X_np = X_np[idx]
            y_np = y_np[idx]
        X = torch.tensor(X_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)
        print(f"Loaded teacher data from {teacher_path}, samples={len(y)} (subsampled if too large)")
    else:
        X_all: List[List[float]] = []
        y_all: List[int] = []
        for instance, remove_count in build_instances():
            Xc, yc = collect_samples(instance, remove_count, agent, teacher="shaw")
            X_all.extend(Xc)
            y_all.extend(yc)
        X = torch.tensor(X_all, dtype=torch.float32)
        y = torch.tensor(y_all, dtype=torch.float32)
        print(f"Collected Shaw teacher data on the fly, samples={len(y)}")
    model, losses, accs = train_model(X, y, hidden=64, epochs=30, batch=256, lr=1e-3, weight_decay=1e-4)
    with torch.no_grad():
        w1 = model.net[0].weight.cpu().numpy().tolist()
        b1 = model.net[0].bias.cpu().numpy().tolist()
        w2 = model.net[2].weight.cpu().numpy().tolist()
        b2 = model.net[2].bias.cpu().numpy().tolist()
        w3 = model.net[4].weight.cpu().numpy().tolist()
        b3 = model.net[4].bias.cpu().numpy().tolist()
    payload: Dict[str, dict] = {
        "mlp3": {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "w3": w3,
            "b3": b3,
        }
    }
    out_path = ROOT / "models" / "drl_model.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved 3-layer MLP weights to {out_path}, dataset size={len(y)}")

    # Save training diagnostics
    metrics = {
        "epochs": len(losses),
        "loss": losses,
        "accuracy": accs,
        "samples": len(y),
    }
    metrics_path = ROOT / "results" / "analysis" / "train_drl_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved training diagnostics to {metrics_path}")
    if plt is not None:
        plt.figure(figsize=(6, 3))
        plt.plot(range(1, len(losses) + 1), losses, label="Loss")
        plt.plot(range(1, len(accs) + 1), accs, label="Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("DRL removal MLP training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(metrics_path.with_suffix(".png"))
        plt.close()
        print(f"Saved training plot to {metrics_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
