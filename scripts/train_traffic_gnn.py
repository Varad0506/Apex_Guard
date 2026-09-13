"""Phase 5: trains TrafficGAT on the synthetic battle-graph dataset.

Requires torch (not installable in this build environment -- see the PPO
section's note on disk space). Run in your own environment:

    pip install torch
    python scripts/generate_battle_graph_data.py
    python scripts/train_traffic_gnn.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from app.models.traffic_graph import TrafficGAT, _ARTIFACT_PATH

DATA_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "training" / "battle_graph_windows.jsonl"


class BattleGraphDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = []
        with path.open() as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        node_feats = torch.tensor(row["node_features"], dtype=torch.float32)
        edge_index = torch.tensor(row["edge_index"], dtype=torch.long)
        edge_feats = torch.tensor(row["edge_features"], dtype=torch.float32)
        node_mask = torch.tensor(row["node_mask"], dtype=torch.float32)
        label = torch.tensor([row["tow_strength"], row["counterattack_risk"],
                               row["post_pass_traffic_risk"]], dtype=torch.float32)
        return node_feats, edge_index, edge_feats, node_mask, label


def collate_single(batch):
    # Graphs have variable edge counts (different nodes masked in/out per
    # sample), so this trains with an effective batch size of 1 -- fine at
    # this graph scale (5 nodes, <=4 edges) and dataset size (a few
    # thousand rows trains in seconds either way).
    return batch[0]


def train(epochs: int = 15, lr: float = 1e-3, seed: int = 0):
    torch.manual_seed(seed)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found -- run scripts/generate_battle_graph_data.py first")

    dataset = BattleGraphDataset(DATA_PATH)
    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_single)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_single)

    model = TrafficGAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for node_feats, edge_index, edge_feats, node_mask, label in train_loader:
            optimizer.zero_grad()
            pred = model(node_feats, edge_index, edge_feats, node_mask)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for node_feats, edge_index, edge_feats, node_mask, label in val_loader:
                pred = model(node_feats, edge_index, edge_feats, node_mask)
                val_loss += loss_fn(pred, label).item()
        val_loss /= len(val_loader)

        print(f"epoch {epoch+1:3d}/{epochs}  train_mse={train_loss:.5f}  val_mse={val_loss:.5f}")

    _ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), _ARTIFACT_PATH)
    print(f"\nSaved -> {_ARTIFACT_PATH}")


if __name__ == "__main__":
    train()