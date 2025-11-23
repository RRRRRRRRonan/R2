#!/usr/bin/env python3
"""Train a larger Transformer removal scorer with pairwise ranking loss on sequence data."""

from __future__ import annotations

import argparse
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
        order = rec["order"]
        for ridx in rec["removed"]:
            if ridx in order:
                pos = order.index(ridx)
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
    def __init__(self, feat_dim: int, d_model: int = 128, nhead: int = 8, num_layers: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        key_padding_mask = ~mask
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        logits = self.head(h).squeeze(-1)
        return logits


def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, neg_samples: int = 2) -> torch.Tensor:
    # logits/labels/mask shape: [B, L]
    losses = []
    for logit, label, m in zip(logits, labels, mask):
        valid_idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        l = label[valid_idx]
        pos_idx = torch.nonzero(l > 0.5, as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(l <= 0.5, as_tuple=False).squeeze(-1)
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        sel_neg = neg_idx
        if neg_idx.numel() > neg_samples:
            sel_neg = neg_idx[torch.randperm(neg_idx.numel())[:neg_samples]]
        pos_logits = logit[valid_idx][pos_idx]
        neg_logits = logit[valid_idx][sel_neg]
        # Broadcast pos vs neg
        diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
        losses.append(-nn.functional.logsigmoid(diff).mean())
    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


def load_records(path: pathlib.Path, max_samples: int | None, seed: int = 42) -> List[dict]:
    records = json.loads(path.read_text())
    rng = random.Random(seed)
    rng.shuffle(records)
    if max_samples:
        records = records[:max_samples]
    return records


def train_model(
    records: List[dict],
    epochs: int = 15,
    batch: int = 64,
    lr: float = 1e-3,
    neg_samples: int = 3,
) -> TransformerScorer:
    ds = SeqDataset(records)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate)
    feat_dim = ds[0][0].shape[1]
    model = TransformerScorer(feat_dim=feat_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for feats, labels, mask in dl:
            logits = model(feats, mask)
            # Hybrid loss: BCE for absolute calibration + pairwise ranking
            loss_bce = bce(logits[mask], labels[mask])
            loss_rank = pairwise_ranking_loss(logits, labels, mask, neg_samples=neg_samples)
            loss = loss_bce + loss_rank
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=ROOT / "data" / "drl_teacher_sequences.json",
    )
    parser.add_argument("--max-samples", type=int, default=8000)
    args = parser.parse_args()
    records = load_records(args.data, max_samples=args.max_samples)
    model = train_model(records, epochs=20, batch=128, lr=1e-3, neg_samples=3)
    state = model.state_dict()
    payload = {
        "transformer": {
            "state_dict": {k: v.cpu().numpy().tolist() for k, v in state.items()},
            "meta": {
                "feat_dim": model.input_proj.in_features,
                "d_model": model.input_proj.out_features,
                "nhead": 8,
                "num_layers": 3,
            },
        }
    }
    out_path = ROOT / "models" / "drl_model.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved ranking transformer weights to {out_path} (samples={len(records)})")


if __name__ == "__main__":
    main()
