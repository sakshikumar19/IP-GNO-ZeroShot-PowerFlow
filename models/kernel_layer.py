"""
kernel_layer.py - GNO kernel implementations for power grid modeling

Contains VanillaKernelLayer (impedance-based) and AdmittanceKernelLayer (physics-informed).
The latter uses admittance features and includes self-admittance gating and source residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing


# ---------------------------------------------------------------------------
# Shared MLP factory
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 3) -> nn.Sequential:
    """Build a small MLP with BatchNorm and ReLU activations."""
    assert layers >= 2
    mods = [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
    for _ in range(layers - 2):
        mods += [nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


# ---------------------------------------------------------------------------
# VanillaKernelLayer  (ablation baseline — raw impedance edge features)
# ---------------------------------------------------------------------------

class VanillaKernelLayer(MessagePassing):
    """
    One GNO integral-operator layer.  Uses raw impedance (R, X) as edge
    features — the ablation baseline without admittance weighting.

    Kernel input:  [v_i ‖ v_j ‖ e_ij[0:2]]   shape (2·d_in + 2,)
    """

    EDGE_SLICE = slice(0, 2)   # (R, X)

    def __init__(
        self,
        d_in:       int,
        d_out:      int,
        edge_dim:   int = 9,
        mlp_hidden: int = 64,
        mlp_layers: int = 3,
        aggr:       str = "mean",
    ) -> None:
        super().__init__(aggr=aggr)
        self.d_in  = d_in
        self.d_out = d_out

        n_edge_feats = 2   # (R, X)
        kernel_in    = d_in + d_in + n_edge_feats
        kernel_out   = d_out * d_in

        self.kernel_mlp = _mlp(kernel_in, mlp_hidden, kernel_out, mlp_layers)
        self.W          = nn.Linear(d_in, d_out, bias=False)
        self.bn         = nn.BatchNorm1d(d_out)

    def forward(self, v: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, v=v, edge_attr=edge_attr, size=None)

    def message(self, v_i: Tensor, v_j: Tensor, edge_attr: Tensor) -> Tensor:
        e_feats = edge_attr[:, self.EDGE_SLICE]
        k_in    = torch.cat([v_i, v_j, e_feats], dim=-1)
        K       = self.kernel_mlp(k_in)
        K       = K.view(-1, self.d_out, self.d_in)
        return torch.bmm(K, v_j.unsqueeze(-1)).squeeze(-1)

    def update(self, aggr_out: Tensor, v: Tensor) -> Tensor:
        return F.relu(self.bn(self.W(v) + aggr_out))


# ---------------------------------------------------------------------------
# AdmittanceKernelLayer  (IP-GNO Proposition 1, v2)
# ---------------------------------------------------------------------------

class AdmittanceKernelLayer(MessagePassing):
    """
    Physics-informed GNO layer using admittance features.
    Includes self-admittance gating and source residual connections for better performance.
    """

    EDGE_SLICE = slice(2, 9)   # 7 admittance features

    def __init__(
        self,
        d_in:       int,
        d_out:      int,
        edge_dim:   int = 9,
        mlp_hidden: int = 64,
        mlp_layers: int = 3,
        aggr:       str = "mean",
    ) -> None:
        super().__init__(aggr=aggr)
        self.d_in  = d_in
        self.d_out = d_out

        n_edge_feats = 7
        kernel_in    = d_in + d_in + n_edge_feats
        kernel_out   = d_out * d_in

        self.kernel_mlp = _mlp(kernel_in, mlp_hidden, kernel_out, mlp_layers)
        self.W          = nn.Linear(d_in, d_out, bias=False)
        self.bn         = nn.BatchNorm1d(d_out)

        # Self-admittance gate: maps [v_i (d_in), log_diag_Y (1)] -> d_out gates
        self.gate_net   = nn.Sequential(
            nn.Linear(d_in + 1, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, d_out),
            nn.Sigmoid(),
        )

        # Source residual: learned blend weight, bounded to (0, 1) via sigmoid.
        # Stored as a logit so gradient flow is unconstrained; sigmoid applied in forward.
        # Init to logit(0.1) ≈ -2.197 → src_weight starts near 0.1, grows with training.
        # This prevents the weight going large-negative at initialisation, which would
        # zero out entire feature dimensions via ReLU for unseen graph structures.
        self._src_logit = nn.Parameter(torch.tensor(-2.197))  # sigmoid(-2.197) ≈ 0.1

        # Zero-init the kernel_mlp output layer so messages start near zero.
        # This is the standard "residual branch init" trick: at epoch 0 the model
        # acts approximately like W·v (linear), then gradually learns non-linear
        # corrections.  Without this, large random messages can destabilise BN
        # statistics on validation graphs that differ from training subgraphs.
        with torch.no_grad():
            self.kernel_mlp[-1].weight.zero_()
            self.kernel_mlp[-1].bias.zero_()

    def forward(
        self,
        v:          Tensor,   # (N, d_in)   current latent state
        edge_index: Tensor,   # (2, E)
        edge_attr:  Tensor,   # (E, 9)
        v0:         Tensor,   # (N, d_in)   original lifted input (source term)
        log_diag_y: Tensor,   # (N, 1)      log self-admittance
    ) -> Tensor:
        agg = self.propagate(edge_index, v=v, edge_attr=edge_attr, size=None)
        gate = self.gate_net(torch.cat([v, log_diag_y], dim=-1))     # (N, d_out)
        src_w = torch.sigmoid(self._src_logit)
        out   = self.W(v) + gate * agg + src_w * v0
        return F.relu(self.bn(out))

    def message(self, v_i: Tensor, v_j: Tensor, edge_attr: Tensor) -> Tensor:
        e_feats = edge_attr[:, self.EDGE_SLICE]            # (E, 7)
        k_in    = torch.cat([v_i, v_j, e_feats], dim=-1)  # (E, 2·d_in + 7)
        K       = self.kernel_mlp(k_in)                    # (E, d_out·d_in)
        K       = K.view(-1, self.d_out, self.d_in)
        return torch.bmm(K, v_j.unsqueeze(-1)).squeeze(-1) # (E, d_out)

    # update() is not used — logic is in forward()