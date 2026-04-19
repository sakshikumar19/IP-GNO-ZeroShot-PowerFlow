"""
IP-GNO model with physics-informed corrections.
Combines GNO predictions with LinDistFlow post-processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from models.kernel_layer  import AdmittanceKernelLayer
from models.distflow      import DistFlowCorrection
from data.dataset         import NormStats, per_graph_edge_norm


def kirchhoff_residual(v_pred_norm, data, stats, eps=1e-6):
    """
    KCL residual for evaluation.  Updated for [|V|, sinθ, cosθ] output:
    recover V_re, V_im from |V|, sinθ, cosθ before computing current balance.
    """
    device  = v_pred_norm.device
    ei      = data.edge_index
    ea_norm = data.edge_attr
    N       = v_pred_norm.size(0)

    fwd_mask = ei[0] < ei[1]
    src = ei[0][fwd_mask]; dst = ei[1][fwd_mask]
    ea  = ea_norm[fwd_mask]

    if src.numel() == 0:
        return torch.tensor(0.0, device=device)

    # Recover physical [|V|, sinθ, cosθ] then convert to rectangular
    V_phy = stats.denormalise_y(v_pred_norm)   # (N, 3)
    V_mag = V_phy[:, 0]
    sin_t = V_phy[:, 1]
    cos_t = V_phy[:, 2]
    V_re  = V_mag * cos_t
    V_im  = V_mag * sin_t

    G = ea[:, 4]; B = ea[:, 5]
    dV_re = V_re[src] - V_re[dst]
    dV_im = V_im[src] - V_im[dst]
    I_re  = G * dV_re - B * dV_im
    I_im  = B * dV_re + G * dV_im

    KCL_re = torch.zeros(N, device=device); KCL_im = torch.zeros(N, device=device)
    KCL_re.scatter_add_(0, src,  I_re); KCL_re.scatter_add_(0, dst, -I_re)
    KCL_im.scatter_add_(0, src,  I_im); KCL_im.scatter_add_(0, dst, -I_im)

    P_n = stats.denormalise_x_col(data.x[:, 2], 2)
    Q_n = stats.denormalise_x_col(data.x[:, 3], 3)
    V_for_inj = V_mag.clamp(min=0.1)
    I_inj_re  = -P_n / V_for_inj
    I_inj_im  =  Q_n / V_for_inj

    sx = stats.sx.to(device); mx = stats.mx.to(device)
    x0 = data.x[:, 0] * (sx[0] + stats.EPS) + mx[0]
    non_slack = x0 < 0.5

    if non_slack.sum() == 0:
        return torch.tensor(0.0, device=device)

    dI_re = KCL_re[non_slack] - I_inj_re[non_slack]
    dI_im = KCL_im[non_slack] - I_inj_im[non_slack]
    result = (dI_re**2 + dI_im**2).mean()
    return result if torch.isfinite(result) else torch.tensor(0.0, device=device)


class IPGNO(nn.Module):
    """
    Physics-Informed Graph Neural Operator  (v4).

    in_dim=10   : 5 base node features + 5 PETE positional encodings
    out_dim=3   : [|V|_pu, sin(θ), cos(θ)] — direct magnitude + circular angle
    per-graph edge norm: admittance features (cols 2-8) normalised per graph
                         in forward pass, enabling cross-grid zero-shot transfer
    """

    def __init__(
        self,
        in_dim=10,       # NODE_DIM (5 base + 5 PETE)
        out_dim=3,       # OUT_DIM  ([|V|, sinθ, cosθ])
        edge_dim=9,
        hidden_dim=64,
        T=5,
        mlp_hidden=64,
        mlp_layers=3,
        dropout=0.1,
        alpha=0.5,
        alpha_init=0.0,   # kept for checkpoint compat
        lambda_phys=0.0,
    ):
        super().__init__()
        self.T       = T
        self.dropout = dropout

        self.lift = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.kernel_layers = nn.ModuleList([
            AdmittanceKernelLayer(
                d_in=hidden_dim, d_out=hidden_dim,
                edge_dim=edge_dim, mlp_hidden=mlp_hidden,
                mlp_layers=mlp_layers, aggr="mean",
            )
            for _ in range(T)
        ])
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.correction = DistFlowCorrection(alpha=alpha)

    def _gno_forward(self, data):
        """Lift → per-graph edge norm → kernel layers → project."""
        batch = getattr(data, 'batch', None)

        # Per-graph normalize edge features for cross-grid transfer
        edge_attr  = per_graph_edge_norm(data.edge_attr, data.edge_index, batch,
                                         cols=slice(0, 9))
        log_diag_y = data.x[:, 4:5]   # (N, 1) — graph-relative by construction
        v0 = self.lift(data.x)
        v  = v0

        for layer in self.kernel_layers:
            v = layer(v, data.edge_index, edge_attr, v0=v0, log_diag_y=log_diag_y)
            v = F.dropout(v, p=self.dropout, training=self.training)
        return self.project(v)   # (N, 3) normalised

    def train_forward(self, data):
        """Training: pure GNO output, no DistFlow correction."""
        return self._gno_forward(data)

    def forward(self, data, stats=None):
        """Inference: GNO + optional DistFlow correction."""
        v_raw = self._gno_forward(data)
        if stats is not None:
            return self.correction(v_raw, data, stats)
        return v_raw

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
