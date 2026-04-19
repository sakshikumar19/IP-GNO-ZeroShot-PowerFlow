"""
Baseline models for power flow prediction.
Includes GAT, GINE, and SAGE architectures for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, SAGEConv, GINEConv

from data.dataset import per_graph_edge_norm


def _mlp(in_dim, hidden, out_dim, layers=3):
    assert layers >= 2
    mods = [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
    for _ in range(layers - 2):
        mods += [nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


class PowerFlowGAT(nn.Module):
    def __init__(self, in_dim=10, out_dim=3, edge_dim=9,
                 hidden_dim=256, T=5, heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % heads == 0
        self.dropout = dropout
        out_per_head = hidden_dim // heads

        self.lift = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, out_per_head, heads=heads,
                      edge_dim=edge_dim, concat=True, dropout=dropout,
                      add_self_loops=False)
            for _ in range(T)
        ])
        self.bns     = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(T)])
        self.project = _mlp(hidden_dim, hidden_dim, out_dim, layers=3)

    def forward(self, data):
        batch     = getattr(data, 'batch', None)
        edge_attr = per_graph_edge_norm(data.edge_attr, data.edge_index, batch, cols=slice(0, 9))
        x = self.lift(data.x)
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(bn(conv(x, data.edge_index, edge_attr)))
        return self.project(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PowerFlowGINE(nn.Module):
    def __init__(self, in_dim=10, out_dim=3, edge_dim=9,
                 hidden_dim=256, T=5, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.lift      = nn.Linear(in_dim, hidden_dim)
        self.edge_lift = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(T):
            inner = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(inner, edge_dim=hidden_dim))
        self.bns     = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(T)])
        self.project = _mlp(hidden_dim, hidden_dim, out_dim, layers=3)

    def forward(self, data):
        batch     = getattr(data, 'batch', None)
        edge_attr = per_graph_edge_norm(data.edge_attr, data.edge_index, batch, cols=slice(0, 9))
        x = F.relu(self.lift(data.x))
        e = F.relu(self.edge_lift(edge_attr))

        xs = [x]
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index, e)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        x = torch.stack(xs, dim=0).max(dim=0).values
        return self.project(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PowerFlowSAGE(nn.Module):
    def __init__(self, in_dim=10, out_dim=3, edge_dim=9,
                 hidden_dim=384, T=5, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim, normalize=True))
        for _ in range(T - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, normalize=True))
        self.bns     = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(T)])
        self.project = _mlp(hidden_dim, hidden_dim, out_dim, layers=3)

    def forward(self, data):
        # SAGE ignores edge features (structural ablation)
        # col 4 is graph-relative by construction, no runtime norm needed
        x = data.x
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(x, data.edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.project(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


BASELINE_REGISTRY = {"gat": PowerFlowGAT, "gine": PowerFlowGINE, "sage": PowerFlowSAGE}

BASELINE_DEFAULTS = {
    "gat":  {"hidden_dim": 256, "T": 5, "heads": 8},
    "gine": {"hidden_dim": 256, "T": 5},
    "sage": {"hidden_dim": 384, "T": 5},
}


def build_baseline(name, **override_kwargs):
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name!r}. Choose from {list(BASELINE_REGISTRY)}")
    kwargs = {**BASELINE_DEFAULTS[name], **override_kwargs}
    return BASELINE_REGISTRY[name](**kwargs)
