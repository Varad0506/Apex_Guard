"""Phase 5: a small Graph Attention Network (GAT) over the local battle
graph (app/models/battle_graph.py), predicting the same 3 outputs
traffic_engine.py's deterministic baseline produces (tow_strength,
counterattack_risk, post_pass_traffic_risk) -- so it can drop in as a
direct swap if (and only if) it wins the ablation test in
scripts/ablation_test_gnn.py.

Implemented in plain PyTorch (manual multi-head attention over the graph's
edges) rather than PyTorch Geometric, to avoid adding a second heavy graph
library dependency on top of stable-baselines3/torch for what is a 5-node
graph -- PyG's benefits (efficient batched sparse ops, established layer
zoo) matter at a scale this problem doesn't reach. If this graph ever grows
past a handful of nodes (e.g. real per-car telemetry becomes available),
switching to PyG's GATConv would be the right call.

torch is an OPTIONAL dependency, scoped to this module only. The rest of
the app (rule engine, verifier, overtake classifier, PPO adapter) must keep
working even when torch is missing or broken in the environment (not
installed, no disk space, no CUDA loader libs, etc.) -- this module is the
only place that pays that cost, and it fails soft: model_is_available()
returns False and traffic_engine.py falls back to its deterministic
baseline. Do NOT replace the try/except below with a bare `import torch`;
that previously took down the entire app on import (traffic_engine.py
imports this module unconditionally), not just the GNN path.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from app.models.battle_graph import BattleGraph, N_NODES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM

_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "data" / "artifacts" / "traffic_gat.pt"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False


_model = None
_load_attempted = False


if TORCH_AVAILABLE:

    class GraphAttentionLayer(nn.Module):
        """Single-head-per-call additive attention over edges, matching the
        classic GAT formulation: attention coefficients from concatenated
        (source, target, edge) features, softmax-normalized per destination
        node, then a weighted sum of transformed source features."""

        def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
            super().__init__()
            self.W = nn.Linear(in_dim, out_dim, bias=False)
            self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False)
            self.attn = nn.Linear(2 * out_dim + out_dim, 1, bias=False)
            self.out_dim = out_dim

        def forward(self, node_feats: "torch.Tensor", edge_index: "torch.Tensor", edge_feats: "torch.Tensor") -> "torch.Tensor":
            # node_feats: (N, in_dim), edge_index: (2, E), edge_feats: (E, edge_dim)
            n = node_feats.shape[0]
            h = self.W(node_feats)  # (N, out_dim)
            e_proj = self.edge_proj(edge_feats)  # (E, out_dim)

            src, dst = edge_index[0], edge_index[1]
            h_src, h_dst = h[src], h[dst]  # (E, out_dim) each

            attn_input = torch.cat([h_src, h_dst, e_proj], dim=-1)  # (E, 3*out_dim)
            scores = F.leaky_relu(self.attn(attn_input)).squeeze(-1)  # (E,)

            # Softmax normalized per destination node -- scatter-softmax done
            # manually since this graph is tiny (<=5 nodes, <=5 edges).
            out = torch.zeros(n, self.out_dim, device=node_feats.device)
            for node in range(n):
                incoming = (dst == node).nonzero(as_tuple=True)[0]
                if len(incoming) == 0:
                    out[node] = h[node]  # no incoming edges -- pass through self
                    continue
                node_scores = scores[incoming]
                weights = F.softmax(node_scores, dim=0)
                out[node] = (weights.unsqueeze(-1) * (h_src[incoming] + e_proj[incoming])).sum(dim=0)

            return F.elu(out)

    class TrafficGAT(nn.Module):
        def __init__(self, hidden_dim: int = 16):
            super().__init__()
            self.gat1 = GraphAttentionLayer(NODE_FEATURE_DIM, hidden_dim, EDGE_FEATURE_DIM)
            self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, EDGE_FEATURE_DIM)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),  # tow_strength, counterattack_risk, post_pass_traffic_risk
            )

        def forward(self, node_feats: "torch.Tensor", edge_index: "torch.Tensor",
                    edge_feats: "torch.Tensor", node_mask: "torch.Tensor") -> "torch.Tensor":
            h = self.gat1(node_feats, edge_index, edge_feats)
            h = self.gat2(h, edge_index, edge_feats)

            # Masked mean pool over present nodes only -- padding nodes (mask=0)
            # shouldn't influence the graph-level prediction.
            mask = node_mask.unsqueeze(-1)  # (N, 1)
            pooled = (h * mask).sum(dim=0) / mask.sum().clamp(min=1.0)

            logits = self.head(pooled)
            return torch.sigmoid(logits)  # all 3 outputs are risk/strength scores in [0, 1]

    def _try_load() -> Optional["TrafficGAT"]:
        global _model, _load_attempted
        if _load_attempted:
            return _model
        _load_attempted = True
        if not _ARTIFACT_PATH.exists():
            return None
        try:
            model = TrafficGAT()
            model.load_state_dict(torch.load(_ARTIFACT_PATH, map_location="cpu"))
            model.eval()
            _model = model
        except Exception:
            _model = None
        return _model

    def model_is_available() -> bool:
        return _try_load() is not None

    def predict(graph: BattleGraph) -> dict:
        """Returns {'tow_strength', 'counterattack_risk', 'post_pass_traffic_risk'}
        -- same keys traffic_engine.TrafficAssessment exposes, for drop-in
        comparison/swap. Raises if no trained artifact is available; callers
        should check model_is_available() first or catch the exception, same
        fail-safe pattern as overtake_probability.py and policy_adapter.py."""
        model = _try_load()
        if model is None:
            raise RuntimeError("No trained traffic_gat.pt artifact found. Run scripts/train_traffic_gnn.py first.")

        with torch.no_grad():
            node_feats = torch.from_numpy(graph.node_features).float()
            edge_index = torch.from_numpy(graph.edge_index).long()
            edge_feats = torch.from_numpy(graph.edge_features).float()
            node_mask = torch.from_numpy(graph.node_mask).float()
            out = model(node_feats, edge_index, edge_feats, node_mask).numpy()

        return {
            "tow_strength": float(out[0]),
            "counterattack_risk": float(out[1]),
            "post_pass_traffic_risk": float(out[2]),
        }

else:

    def model_is_available() -> bool:
        """torch isn't available in this environment -- the GNN path is
        disabled and traffic_engine.py falls back to its deterministic
        baseline. This is expected/handled behavior, not an error."""
        return False

    def predict(graph: BattleGraph) -> dict:
        raise RuntimeError(
            "torch is not installed/available, so the Phase 5 GNN traffic "
            "model can't run. Callers should check model_is_available() "
            "first (traffic_engine.py already does) and use the "
            "deterministic baseline instead. Install torch to enable this "
            "path."
        )
