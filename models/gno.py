"""
Vanilla GNO model - ablation baseline.
Uses standard kernel layers without physics corrections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from models.kernel_layer import VanillaKernelLayer
from data.dataset        import per_graph_edge_norm


class VanillaGNO(nn.Module):
    """
    Vanilla Graph Neural Operator (ablation baseline) — v4.

    Node features (NODE_DIM = 10):
        [0] is_slack       [1] has_load    [2] P_pu      [3] Q_pu
        [4] log_diag_y
        [5] depth_rel      [6] r_path_rel  [7] x_path_rel
        [8] downstream_rel [9] degree_norm

    Targets (OUT_DIM = 3):
        [0] |V|_pu    [1] sin(θ)    [2] cos(θ)
    """

    def __init__(
        self,
        in_dim=10,
        out_dim=3,
        edge_dim=9,
        hidden_dim=64,
        T=5,
        mlp_hidden=64,
        mlp_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.T       = T
        self.dropout = dropout

        self.lift = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.kernel_layers = nn.ModuleList([
            VanillaKernelLayer(
                d_in=hidden_dim, d_out=hidden_dim,
                edge_dim=edge_dim, mlp_hidden=mlp_hidden,
                mlp_layers=mlp_layers, aggr="mean",
            )
            for _ in range(T)
        ])
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        batch = getattr(data, 'batch', None)

        # Per-graph normalise ALL edge features (cols 0-8) for cross-grid transfer.
        # col 4 is graph-relative by construction (see dataset.py) — no runtime node norm.
        edge_attr = per_graph_edge_norm(data.edge_attr, data.edge_index, batch,
                                        cols=slice(0, 9))

        v = self.lift(data.x)
        for layer in self.kernel_layers:
            v = layer(v, data.edge_index, edge_attr)
            v = F.dropout(v, p=self.dropout, training=self.training)
        return self.project(v)   # (N, 3)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
