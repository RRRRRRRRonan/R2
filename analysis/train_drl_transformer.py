#!/usr/bin/env python3
"""Train a small Transformer removal scorer via sequence-level imitation."""

from __future__ import annotations

import json
import pathlib
import random
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


class SeqDataset(Dataset):
    def __init__(self, records: List[dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        rec = self.records[idx]
        feats = np.array(rec["features"], dtype=np.float32)
        labels = np.zeros((len(rec["order"]),), dtype=np.float32)
        for ridx in rec["removed"]:
            # removed indices refer to task ids; map to position
            if ridx in rec["order"]:
                pos = rec["order"].index(ridx)
                labels[pos] = 1.0
        return feats, labels


def collate(batch):
    feats, labels = zip(*batch)
    max_len = max(f.shape[0] for f in feats)
    feat_dim = feats[0].shape[1]
    feat_pad = np.zeros((len(batch), max_len, feat_dim), dtype=np.float32)
    label_pad = np.zeros((len(batch), max_len), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=np.bool_)
    for i, (f, l) in enumerate(zip(feats, labels)):
        n = f.shape[0]
        feat_pad[i, :n] = f
        label_pad[i, :n] = l
        mask[i, :n] = True
    return (
        torch.tensor(feat_pad, dtype=torch.float32),
        torch.tensor(label_pad, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool),
    )


class TransformerScorer(nn.Module):
    def __init__(self, feat_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        key_padding_mask = ~mask
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        logits = self.head(h).squeeze(-1)
        return logits


def load_records(path: pathlib.Path, max_samples: int | None = 5000) -> List[dict]:
    records = json.loads(path.read_text())
    random.Random(42).shuffle(records)
    if max_samples:
        records = records[:max_samples]
    return records


def train_model(records: List[dict], epochs: int = 10, batch: int = 64, lr: float = 1e-3) -> TransformerScorer:
    ds = SeqDataset(records)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate)
    feat_dim = ds[0][0].shape[1]
    model = TransformerScorer(feat_dim=feat_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for feats, labels, mask in dl:
            logits = model(feats, mask)
            loss = bce(logits[mask], labels[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def main() -> None:
    data_path = ROOT / "data/drl_teacher_sequences.json"
    records = load_records(data_path, max_samples=8000)
    model = train_model(records, epochs=15, batch=128, lr=1e-3)
    with torch.no_grad():
        state = model.state_dict()
        meta = {
            "feat_dim": model.input_proj.in_features,
            "d_model": model.input_proj.out_features,
            "nhead": 4,
            "num_layers": 2,
        }
        out = {"transformer": {"state_dict": {k: v.cpu().numpy().tolist() for k, v in state.items()}, "meta": meta}}
    out_path = ROOT / "models" / "drl_model.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved transformer weights to {out_path} (samples={len(records)})")


if __name__ == "__main__":
    main()
