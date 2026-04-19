"""
dataset.py - Dataset generation and utilities for power grid graphs

Handles IEEE 33-bus and 69-bus systems, PETE features, normalization, and subgraph extraction.
Key features:
- PETE: Physics-Embedded Tree Encoding for scale-invariant positional features
- Output format: [|V|, sin(θ), cos(θ)] for better angle handling
- Per-graph edge normalization for cross-grid transfer
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from data.ieee33 import (
    ALL_BUSES as ALL_BUSES_33,
    IEEE33BusSimulator,
)
from data.ieee69 import (
    ALL_BUSES_69,
    IEEE69BusSimulator,
)

BASE_KVA   = 1000.0
BASE_KV    = 12.66
Z_BASE     = (BASE_KV ** 2) / (BASE_KVA / 1000.0)
SLACK_V_PU = 1.05
V0_SQ      = SLACK_V_PU ** 2

NODE_DIM   = 10   # 5 base + 5 PETE features
EDGE_DIM   = 9
OUT_DIM    = 3    # [|V|_pu, sin(θ), cos(θ)]
SLACK_BUS  = "1"


# ---------------------------------------------------------------------------
# Admittance helpers (unchanged)
# ---------------------------------------------------------------------------

def impedance_to_admittance(R, X):
    eps = 1e-6
    z2  = R * R + X * X
    mag = 1.0 / math.sqrt(z2) if z2 > 1e-12 else eps
    ang = math.atan2(-X, R)
    g   =  R / z2 if z2 > 1e-12 else eps
    b   = -X / z2 if z2 > 1e-12 else eps
    return mag, ang, g, b, math.log(mag + eps), math.log(abs(g) + eps), math.log(abs(b) + eps)


def build_ybus_sparse(buses, enabled_lines):
    n = len(buses)
    bus_to_idx = {b: i for i, b in enumerate(buses)}
    Y = torch.zeros(n, n, dtype=torch.cfloat)
    for fb, tb, R, X in enabled_lines:
        if fb not in bus_to_idx or tb not in bus_to_idx:
            continue
        i, j = bus_to_idx[fb], bus_to_idx[tb]
        z2 = R * R + X * X
        if z2 < 1e-12:
            continue
        y_ij = complex(R / z2, -X / z2) * Z_BASE
        Y[i, i] += y_ij; Y[j, j] += y_ij
        Y[i, j] -= y_ij; Y[j, i] -= y_ij
    return torch.stack([Y.real, Y.imag], dim=0)


# ---------------------------------------------------------------------------
# LinDistFlow matrix builder (unchanged from v3)
# ---------------------------------------------------------------------------

def build_ldf_matrices(n_buses, slack_idx, edge_index, edge_attr_raw):
    """Build R_ldf, X_ldf from RAW edge_attr (R_ohm in col 0, X_ohm in col 1)."""
    ns = n_buses - 1
    src_all = edge_index[0]; dst_all = edge_index[1]
    fwd = src_all < dst_all
    src = src_all[fwd].tolist()
    dst = dst_all[fwd].tolist()
    ea  = edge_attr_raw[fwd]
    L   = len(src)

    if L != ns or L == 0:
        return torch.zeros(ns, ns), torch.zeros(ns, ns)

    r_pu = (ea[:, 0] / Z_BASE).numpy()
    x_pu = (ea[:, 1] / Z_BASE).numpy()

    adj = {i: [] for i in range(n_buses)}
    for k, (s, d) in enumerate(zip(src, dst)):
        adj[s].append((d, k)); adj[d].append((s, k))

    oriented_src = np.zeros(L, dtype=np.int32)
    oriented_dst = np.zeros(L, dtype=np.int32)
    visited = [False] * n_buses
    queue   = [slack_idx]
    visited[slack_idx] = True
    while queue:
        u = queue.pop(0)
        for v, k in adj[u]:
            if not visited[v]:
                visited[v] = True
                oriented_src[k] = u
                oriented_dst[k] = v
                queue.append(v)

    if not all(visited):
        return torch.zeros(ns, ns), torch.zeros(ns, ns)

    non_slack = [i for i in range(n_buses) if i != slack_idx]
    b2c = {b: c for c, b in enumerate(non_slack)}

    A = np.zeros((L, ns), dtype=np.float64)
    for k in range(L):
        if oriented_src[k] in b2c:
            A[k, b2c[oriented_src[k]]] = +1.0
        if oriented_dst[k] in b2c:
            A[k, b2c[oriented_dst[k]]] = -1.0

    try:
        F = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return torch.zeros(ns, ns), torch.zeros(ns, ns)

    R_ldf = (F * r_pu) @ F.T
    X_ldf = (F * x_pu) @ F.T
    return torch.tensor(R_ldf, dtype=torch.float32), torch.tensor(X_ldf, dtype=torch.float32)


# ---------------------------------------------------------------------------
# PETE: Physics-Embedded Tree Encoding
# ---------------------------------------------------------------------------

def compute_pete_features(n_buses, slack_idx, edge_index, edge_attr_raw):
    """
    Compute PETE features: scale-invariant positional encodings for power grids.
    Returns 5 features per bus: depth, path resistance/reactance, downstream fraction, degree.
    All normalized to [0,1] within the graph.
    """
    EPS = 1e-8

    # Build adjacency from raw edge_attr (R_ohm in col 0, X_ohm in col 1)
    adj = {i: [] for i in range(n_buses)}
    seen = set()
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    for k, (s, d) in enumerate(zip(src_list, dst_list)):
        key = (min(s, d), max(s, d))
        if key not in seen:
            seen.add(key)
            r_pu = float(edge_attr_raw[k, 0]) / Z_BASE
            x_pu = float(edge_attr_raw[k, 1]) / Z_BASE
            adj[s].append((d, r_pu, x_pu))
            adj[d].append((s, r_pu, x_pu))

    # BFS from slack
    depth  = [-1] * n_buses
    r_path = [0.0] * n_buses
    x_path = [0.0] * n_buses
    parent = [-1] * n_buses

    depth[slack_idx] = 0
    queue     = [slack_idx]
    bfs_order = []
    while queue:
        u = queue.pop(0)
        bfs_order.append(u)
        for v, r, x in adj[u]:
            if depth[v] < 0:
                depth[v]  = depth[u] + 1
                r_path[v] = r_path[u] + r
                x_path[v] = x_path[u] + x
                parent[v] = u
                queue.append(v)

    # Fix any isolated nodes (shouldn't occur in valid tree)
    for i in range(n_buses):
        if depth[i] < 0:
            depth[i] = 0

    # Downstream count (reverse BFS: leaves first)
    downstream = [1] * n_buses
    for u in reversed(bfs_order):
        p = parent[u]
        if p >= 0:
            downstream[p] += downstream[u]

    # Node degrees (undirected)
    degree = [len(adj[i]) for i in range(n_buses)]

    # Graph-relative normalisation denominators
    max_depth   = max(max(depth), 1)
    max_r_path  = max(max(r_path), EPS)
    max_x_path  = max(max(x_path), EPS)
    n_non_slack = max(n_buses - 1, 1)
    max_degree  = max(max(degree), 1)

    pete = torch.zeros(n_buses, 5, dtype=torch.float32)
    for i in range(n_buses):
        pete[i, 0] = depth[i] / max_depth
        pete[i, 1] = r_path[i] / max_r_path
        pete[i, 2] = x_path[i] / max_x_path
        pete[i, 3] = (downstream[i] - 1) / n_non_slack
        pete[i, 4] = degree[i] / max_degree

    return pete


# ---------------------------------------------------------------------------
# Per-graph normalisation utilities
# ---------------------------------------------------------------------------

def per_graph_edge_norm(edge_attr, edge_index, batch=None, cols=slice(0, 9)):
    """
    Normalize edge features to zero-mean, unit-std within each graph.
    Enables cross-grid transfer by making features scale-invariant.
    """
    ea = edge_attr.clone()
    if ea.size(0) == 0:
        return ea

    feats = ea[:, cols]

    if batch is None:
        mu  = feats.mean(0, keepdim=True)
        std = feats.std(0, keepdim=True)
        std = torch.nan_to_num(std, nan=1.0).clamp(min=1e-8)   # NaN-safe clamp
        ea[:, cols] = (feats - mu) / std
    else:
        edge_batch = batch[edge_index[0]]
        n_graphs   = int(edge_batch.max().item()) + 1
        for g in range(n_graphs):
            mask = edge_batch == g
            if mask.sum() <= 1:
                continue
            f_g = feats[mask]
            mu  = f_g.mean(0, keepdim=True)
            std = f_g.std(0, keepdim=True)
            std = torch.nan_to_num(std, nan=1.0).clamp(min=1e-8)   # NaN-safe clamp
            ea[mask, cols] = (f_g - mu) / std
    return ea


def per_graph_node_norm(x, batch=None, col: int = 4):
    """
    Normalize a node feature column to zero-mean, unit-std within each graph.
    Used for log self-admittance to enable cross-grid transfer.
    """
    x = x.clone()
    col_data = x[:, col:col+1]

    if batch is None:
        mu  = col_data.mean(0, keepdim=True)
        std = col_data.std(0, keepdim=True).clamp(min=1e-8)
        x[:, col:col+1] = (col_data - mu) / std
    else:
        n_graphs = int(batch.max().item()) + 1
        for g in range(n_graphs):
            mask = batch == g
            f_g  = col_data[mask]
            mu   = f_g.mean(0, keepdim=True)
            std  = f_g.std(0, keepdim=True).clamp(min=1e-8)
            x[mask, col:col+1] = (f_g - mu) / std
    return x


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_pyg_graph(buses, scenario, enabled_lines, include_ybus=True):
    bus_to_idx = {b: i for i, b in enumerate(buses)}
    n          = len(buses)
    slack_idx  = bus_to_idx[SLACK_BUS]

    if include_ybus:
        ybus_tensor = build_ybus_sparse(buses, enabled_lines)
        diag_mag    = torch.sqrt(ybus_tensor[0].diagonal()**2 + ybus_tensor[1].diagonal()**2)
        log_diag_y_raw = torch.log(diag_mag + 1e-6)
    else:
        ybus_tensor    = None
        log_diag_y_raw = torch.zeros(n)

    # Normalize log self-admittance per graph for cross-grid consistency
    ldy_mean = log_diag_y_raw.mean()
    ldy_std  = log_diag_y_raw.std().clamp(min=1e-8)
    if not torch.isfinite(ldy_std) or ldy_std < 1e-8:
        ldy_std = torch.tensor(1e-8)
    log_diag_y = (log_diag_y_raw - ldy_mean) / ldy_std

    # Base node features (5) + targets (3: |V|, sinθ, cosθ)
    x_base_rows, y_rows = [], []
    for idx, bus in enumerate(buses):
        p_kw   = scenario.get(f"P_{bus}", 0.0)
        q_kvar = scenario.get(f"Q_{bus}", 0.0)
        x_base_rows.append([
            1.0 if bus == SLACK_BUS else 0.0,
            1.0 if (abs(p_kw) > 1e-6 or abs(q_kvar) > 1e-6) else 0.0,
            p_kw  / BASE_KVA,
            q_kvar/ BASE_KVA,
            float(log_diag_y[idx]),   # graph-relative self-admittance
        ])
        v_mag = scenario.get(f"V_{bus}", 1.0)
        theta = scenario.get(f"Angle_{bus}", 0.0) * (math.pi / 180.0)
        y_rows.append([v_mag, math.sin(theta), math.cos(theta)])

    x_base = torch.tensor(x_base_rows, dtype=torch.float32)
    y      = torch.tensor(y_rows,      dtype=torch.float32)

    # Edge attributes (raw — NOT per-graph normalised here; done in model forward)
    ei_rows, ea_rows = [], []
    for fb, tb, R, Xl in enabled_lines:
        if fb not in bus_to_idx or tb not in bus_to_idx:
            continue
        i, j = bus_to_idx[fb], bus_to_idx[tb]
        y_mag_v, y_ang, G, B, log_y, log_g, log_b = impedance_to_admittance(R, Xl)
        attr = [R, Xl, y_mag_v, y_ang, G, B, log_y, log_g, log_b]
        for s, d in [(i, j), (j, i)]:
            ei_rows.append([s, d]); ea_rows.append(attr)

    if ei_rows:
        edge_index = torch.tensor(ei_rows, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(ea_rows, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, EDGE_DIM), dtype=torch.float32)

    # LDF matrices (from raw edge_attr)
    R_ldf, X_ldf = build_ldf_matrices(n, slack_idx, edge_index, edge_attr)

    # PETE features (graph-relative, all in [0,1])
    pete = compute_pete_features(n, slack_idx, edge_index, edge_attr)

    x = torch.cat([x_base, pete], dim=1)  # (N, 10)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.R_ldf_flat = R_ldf.flatten()
    data.X_ldf_flat = X_ldf.flatten()
    data.ldf_n      = torch.tensor(n - 1)
    data.v0_sq      = torch.tensor(V0_SQ)
    data.slack_idx  = torch.tensor(slack_idx)

    if include_ybus:
        data.ybus = ybus_tensor

    return data


# ---------------------------------------------------------------------------
# Subgraph utilities
# ---------------------------------------------------------------------------

def _build_nx(data):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    for u, v in data.edge_index.t().tolist():
        G.add_edge(u, v)
    return G


def _bfs_nodes(G, start, size):
    visited, queue = set(), [start]
    while queue and len(visited) < size:
        v = queue.pop(0)
        if v in visited:
            continue
        visited.add(v)
        queue.extend(nb for nb in G.neighbors(v) if nb not in visited)
    return list(visited)


def extract_subgraph(data, node_ids):
    node_ids = sorted(set(node_ids))
    remap    = {old: new for new, old in enumerate(node_ids)}

    # Base features (first 5 cols) are valid after slicing; PETE cols are NOT
    # (they're full-graph relative), so we recompute PETE below.
    x_base_s = data.x[node_ids, :5]   # (N_sub, 5)
    y_s      = data.y[node_ids]        # (N_sub, 3)

    mask = [k for k, (u, v) in enumerate(data.edge_index.t().tolist())
            if u in remap and v in remap]
    if mask:
        m  = torch.tensor(mask, dtype=torch.long)
        ei = torch.tensor(
            [[remap[int(u)], remap[int(v)]]
             for u, v in data.edge_index.t()[m].tolist()],
            dtype=torch.long).t().contiguous()
        ea = data.edge_attr[m]   # raw edge_attr (R_ohm/X_ohm in cols 0,1 — needed for LDF/PETE)
    else:
        ei = torch.empty((2, 0), dtype=torch.long)
        ea = torch.empty((0, EDGE_DIM), dtype=torch.float32)

    # Re-normalise col 4 (log_diag_y) per-subgraph.
    # The values were graph-relative for the FULL graph; after slicing, they're
    # no longer zero-mean within the subgraph.  Re-centering and rescaling makes
    # col 4 consistently zero-mean unit-std for any graph size (subgraph or full).
    col4      = x_base_s[:, 4]
    c4_mean   = col4.mean()
    c4_std    = col4.std()
    c4_std    = c4_std if (torch.isfinite(c4_std) and c4_std > 1e-8) else torch.tensor(1e-8)
    x_base_s  = x_base_s.clone()
    x_base_s[:, 4] = (col4 - c4_mean) / c4_std

    is_slack_col = x_base_s[:, 0]
    slack_in_sub = int(is_slack_col.argmax().item()) if is_slack_col.max() > 0.5 else 0
    n_sub = len(node_ids)

    R_ldf_s, X_ldf_s = build_ldf_matrices(n_sub, slack_in_sub, ei, ea)

    # Recompute PETE for this subgraph (depth/path/downstream all change)
    pete_s = compute_pete_features(n_sub, slack_in_sub, ei, ea)
    x_s    = torch.cat([x_base_s, pete_s], dim=1)   # (N_sub, 10)

    sub = Data(x=x_s, edge_index=ei, edge_attr=ea, y=y_s)
    sub.R_ldf_flat = R_ldf_s.flatten()
    sub.X_ldf_flat = X_ldf_s.flatten()
    sub.ldf_n      = torch.tensor(n_sub - 1)
    sub.v0_sq      = data.v0_sq.clone()
    sub.slack_idx  = torch.tensor(slack_in_sub)

    if hasattr(data, 'ybus') and data.ybus is not None:
        sub.ybus = data.ybus[:, node_ids, :][:, :, node_ids]

    return sub


def sample_subgraphs(data, n_subgraphs, min_size=20, max_size=25):
    G          = _build_nx(data)
    n          = data.num_nodes
    slack_node = int(data.x[:, 0].argmax().item())
    out        = []
    for _ in range(n_subgraphs):
        size  = random.randint(min_size, min(max_size, n))
        start = random.randint(0, n - 1)
        nodes = _bfs_nodes(G, start, size)
        if slack_node not in nodes:
            if len(nodes) >= size:
                nodes = nodes[:-1]
            nodes = [slack_node] + nodes
        if len(nodes) >= min_size:
            out.append(extract_subgraph(data, nodes))
    return out


# ---------------------------------------------------------------------------
# Dataset generation (unchanged from v3)
# ---------------------------------------------------------------------------

def generate_dataset(
    num_configs=40, scenarios_per_cfg=5, subgraphs_per_scen=5,
    min_sub_size=20, max_sub_size=25, train_frac=0.70, val_frac=0.15,
    seed=42, include_ybus=True, grid="33",
):
    random.seed(seed); np.random.seed(seed)

    if grid == "69":
        sim, all_buses = IEEE69BusSimulator(), ALL_BUSES_69
    else:
        sim, all_buses = IEEE33BusSimulator(), ALL_BUSES_33

    full_graphs, cfg_ids = [], []
    for cfg_idx in range(num_configs):
        G_cfg   = sim.generate_radial_topology()
        sw_vec  = sim.topology_to_switch_vector(G_cfg)
        enabled = sim.enabled_lines_from_vector(sw_vec)
        for _ in range(scenarios_per_cfg):
            scenario = sim.sample_load_scenario()
            result   = sim.solve(load_scenario=scenario, switch_vector=sw_vec)
            if result is None:
                continue
            graph = build_pyg_graph(all_buses, result, enabled, include_ybus=include_ybus)
            full_graphs.append(graph); cfg_ids.append(cfg_idx)

    unique_cfgs = sorted(set(cfg_ids))
    n_train = int(train_frac * len(unique_cfgs))
    n_val   = int(val_frac   * len(unique_cfgs))
    train_cfgs = set(unique_cfgs[:n_train])
    val_cfgs   = set(unique_cfgs[n_train: n_train + n_val])
    test_cfgs  = set(unique_cfgs[n_train + n_val:])

    train_sub, val_sub, test_sub, full_test, train_full = [], [], [], [], []
    for g, cfg in zip(full_graphs, cfg_ids):
        subs = sample_subgraphs(g, n_subgraphs=subgraphs_per_scen,
                                min_size=min_sub_size, max_size=max_sub_size)
        if cfg in train_cfgs:
            train_sub.extend(subs); train_full.append(g)
        elif cfg in val_cfgs:       val_sub.extend(subs)
        elif cfg in test_cfgs:
            test_sub.extend(subs);  full_test.append(g)

    print(f"[dataset] configs: {len(unique_cfgs)}  "
          f"(train={len(train_cfgs)}, val={len(val_cfgs)}, test={len(test_cfgs)})")
    print(f"[dataset] subgraphs: train={len(train_sub)}, val={len(val_sub)}, test={len(test_sub)}")
    print(f"[dataset] full test graphs: {len(full_test)}")
    print(f"[dataset] train full graphs: {len(train_full)}")
    return train_sub, val_sub, test_sub, full_test, train_full


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

@dataclass
class NormStats:
    mx:  torch.Tensor
    sx:  torch.Tensor
    my:  torch.Tensor
    sy:  torch.Tensor
    me:  torch.Tensor
    se:  torch.Tensor
    EPS: float = 1e-8

    def normalise(self, data):
        data = data.clone()
        data.x = (data.x - self.mx) / (self.sx + self.EPS)
        data.y = (data.y - self.my) / (self.sy + self.EPS)
        if data.edge_attr.numel() > 0:
            data.edge_attr = (data.edge_attr - self.me) / (self.se + self.EPS)
        return data

    def denormalise_y(self, y_norm):
        return y_norm * (self.sy.to(y_norm.device) + self.EPS) + self.my.to(y_norm.device)

    def normalise_y(self, y_phys):
        return ((y_phys - self.my.to(y_phys.device))
                / (self.sy.to(y_phys.device) + self.EPS))

    def denormalise_x_col(self, x_norm_col, col):
        d = x_norm_col.device
        return x_norm_col * (self.sx[col].to(d) + self.EPS) + self.mx[col].to(d)

    def to_polar(self, y_norm):
        """
        Convert normalised model output to [|V|_pu, θ_deg].

        y_norm shape: (N, 3) — normalised [|V|, sin(θ), cos(θ)]
        returns      (N, 2) — [|V|_pu, θ_deg]
        """
        v = self.denormalise_y(y_norm)       # (N, 3): physical [|V|, sinθ, cosθ]
        v_mag = v[:, 0]
        v_ang = torch.atan2(v[:, 1], v[:, 2]) * (180.0 / math.pi)
        return torch.stack([v_mag, v_ang], dim=-1)

    def save(self, path):
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path, weights_only=True))


def fit_normalisation(train_graphs, full_train_graphs=None):
    """
    Fit global normalisation statistics.
    Edge normalisation (me, se) is for cols 0-1 (R, X); cols 2-8 are
    per-graph normalised at model forward time via per_graph_edge_norm().
    """
    stat_graphs = full_train_graphs if full_train_graphs else train_graphs
    all_x = torch.cat([g.x         for g in train_graphs],  dim=0)
    all_y = torch.cat([g.y         for g in stat_graphs],   dim=0)
    all_e = torch.cat([g.edge_attr for g in train_graphs if g.edge_attr.numel() > 0], dim=0)
    return NormStats(
        mx=all_x.mean(0), sx=all_x.std(0),
        my=all_y.mean(0), sy=all_y.std(0),
        me=all_e.mean(0), se=all_e.std(0),
    )


def normalise_splits(stats, *splits):
    return tuple([[stats.normalise(g) for g in split] for split in splits])
