from __future__ import annotations

import json
import math
import pathlib
from typing import List, Sequence
import numpy as np

# Torch is only needed when using transformer-based agents. Make it optional to
# allow lightweight runs without installing PyTorch.
try:
    import torch  # type: ignore
    from torch import nn  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

from .data import ProblemInstance, Task


class DRLAgent:
    """Lightweight agent supporting linear or small MLP scoring."""

    def __init__(self, weights: List[float] | None = None, mlp: dict | None = None) -> None:
        self.weights = weights if weights is not None else [0.0] * 5
        self.mlp = mlp
        self.transformer: nn.Module | None = None
        self.transformer_meta: dict | None = None

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {"weights": self.weights}
        if self.mlp:
            payload = {"mlp": self.mlp}
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: pathlib.Path) -> "DRLAgent":
        if not path.exists():
            raise FileNotFoundError(f"DRL model not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "mlp3" in data:
            return cls(None, {"mlp3": data["mlp3"]})
        if "mlp" in data:
            return cls(None, {"mlp": data["mlp"]})
        if "transformer" in data:
            agent = cls(None, {"transformer": data["transformer"]})
            return agent
        weights = [float(x) for x in data.get("weights", [0.0] * 5)]
        return cls(weights, None)

    def features(
        self,
        instance: ProblemInstance,
        task: Task,
        position: int,
        total: int,
        order: Sequence[int] | None = None,
    ) -> List[float]:
        depot_dist = instance.distance(None, task)
        norm_pos = position / max(total - 1, 1)
        demand_ratio = task.demand / instance.vehicle_capacity
        angle = math.atan2(task.y - instance.depot[1], task.x - instance.depot[0]) / math.pi
        # Context distances to predecessor/successor in the current order
        if order is None:
            prev_dist = 0.0
            next_dist = 0.0
        else:
            prev_idx = order[position - 1] if position > 0 else None
            next_idx = order[position + 1] if position + 1 < total else None
            prev_dist = instance.distance(instance.tasks[prev_idx], task) if prev_idx is not None else 0.0
            next_dist = instance.distance(task, instance.tasks[next_idx]) if next_idx is not None else 0.0
        bias = 1.0
        return [
            depot_dist / 100,
            norm_pos,
            demand_ratio,
            angle,
            prev_dist / 100,
            next_dist / 100,
            bias,
        ]

    def score_tasks(self, instance: ProblemInstance, order: Sequence[int]) -> List[float]:
        scores: List[float] = []
        total = len(order)
        use_mlp3 = self.mlp is not None and "mlp3" in self.mlp
        use_mlp = self.mlp is not None and "mlp" in self.mlp
        use_transformer = self.mlp is not None and "transformer" in self.mlp and torch is not None
        if use_mlp3:
            mlp = self.mlp["mlp3"]
            w1 = np.array(mlp["w1"], dtype=float)
            b1 = np.array(mlp["b1"], dtype=float)
            w2 = np.array(mlp["w2"], dtype=float)
            b2 = np.array(mlp["b2"], dtype=float)
            w3 = np.array(mlp["w3"], dtype=float)
            b3 = np.array(mlp["b3"], dtype=float)
        elif use_mlp:
            mlp = self.mlp["mlp"]
            w1 = np.array(mlp["w1"], dtype=float)
            b1 = np.array(mlp["b1"], dtype=float)
            w2 = np.array(mlp["w2"], dtype=float)
            b2 = np.array(mlp["b2"], dtype=float)
            w3 = b3 = None
        else:
            w1 = b1 = w2 = b2 = w3 = b3 = None
        if use_transformer and self.transformer is None:
            if torch is None or nn is None:
                raise RuntimeError("Transformer model requested but PyTorch is not installed")
            trans = self.mlp["transformer"]
            self.transformer = self._build_transformer(trans["meta"], trans["state_dict"])
        if use_transformer and self.transformer is not None:
            feats = []
            for pos, idx in enumerate(order):
                feats.append(self.features(instance, instance.tasks[idx], pos, total, order))
            x = torch.tensor([feats], dtype=torch.float32)
            mask = torch.ones(x.shape[:2], dtype=torch.bool)
            with torch.no_grad():
                logits = self.transformer(x, mask)
            return logits.squeeze(0).tolist()
        for pos, idx in enumerate(order):
            features = self.features(instance, instance.tasks[idx], pos, total, order)
            if use_mlp3 and w1 is not None:
                x = np.array(features, dtype=float)
                h1 = np.maximum(0.0, x @ w1.T + b1)
                h2 = np.maximum(0.0, h1 @ w2.T + b2)
                score = float(h2 @ w3.T + b3)  # type: ignore[arg-type]
            elif use_mlp and w1 is not None:
                x = np.array(features, dtype=float)
                h = np.maximum(0.0, x @ w1.T + b1)
                score = float(h @ w2[:, 0] + b2[0])
            else:
                score = sum(f * w for f, w in zip(features, self.weights))
            scores.append(score)
        return scores

    def select_tasks(self, instance: ProblemInstance, order: Sequence[int], num_remove: int) -> List[int]:
        if not order:
            return []
        num_remove = min(num_remove, len(order))
        scores = self.score_tasks(instance, order)
        indexed = sorted(zip(order, scores), key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in indexed[:num_remove]]

    def _build_transformer(self, meta: dict, state: dict) -> nn.Module:
        if nn is None or torch is None:
            raise RuntimeError("Transformer support requires PyTorch to be installed")
        class T(nn.Module):
            def __init__(self, fd: int, dm: int, nh: int, nl: int) -> None:
                super().__init__()
                self.input_proj = nn.Linear(fd, dm)
                layer = nn.TransformerEncoderLayer(d_model=dm, nhead=nh, dim_feedforward=2 * dm, batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=nl)
                self.head = nn.Linear(dm, 1)

            def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                h = self.input_proj(x)
                key_padding_mask = ~mask
                h = self.encoder(h, src_key_padding_mask=key_padding_mask)
                return self.head(h).squeeze(-1)

        model = T(
            fd=meta["feat_dim"],
            dm=meta["d_model"],
            nh=meta["nhead"],
            nl=meta["num_layers"],
        )
        state_t = {k: torch.tensor(v) for k, v in state.items()}
        model.load_state_dict(state_t, strict=False)
        model.eval()
        return model
