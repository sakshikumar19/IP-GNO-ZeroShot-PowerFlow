"""
distflow.py - LinDistFlow correction layer for IP-GNO

Post-processes GNO predictions using linearized power flow equations for improved accuracy.
Corrects voltage magnitudes while preserving angles.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from data.dataset import NormStats


class DistFlowCorrection(nn.Module):
    """
    Post-processes GNO voltage predictions using LinDistFlow equations.
    Blends GNO predictions with physics-based corrections for better accuracy.
    """

    def __init__(self, alpha=0.5, eps=1e-4):
        super().__init__()
        self.register_buffer('_alpha', torch.tensor(alpha))
        self.eps = eps

    @property
    def alpha(self):
        return self._alpha

    def forward(self, v_norm, data, stats):
        """
        v_norm : (N_total, 3) — GNO output normalised [|V|, sinθ, cosθ]
        data   : PyG Data or Batch
        stats  : NormStats
        """
        device = v_norm.device
        sy  = stats.sy.to(device)
        my  = stats.my.to(device)
        sx  = stats.sx.to(device)
        mx  = stats.mx.to(device)
        EPS = stats.EPS
        alpha = self.alpha

        if hasattr(data, 'ptr') and data.ptr is not None:
            ptr = data.ptr.tolist()
        else:
            ptr = [0, v_norm.size(0)]

        n_graphs = len(ptr) - 1
        v_out    = v_norm.clone()

        ldf_starts = [0]
        for g in range(n_graphs):
            N_g  = ptr[g + 1] - ptr[g]
            ns_g = N_g - 1
            ldf_starts.append(ldf_starts[-1] + ns_g * ns_g)

        for g in range(n_graphs):
            s = ptr[g]; e = ptr[g + 1]
            N  = e - s
            ns = N - 1

            if ns <= 0:
                continue

            # Reconstruct LDF matrices
            f_s = ldf_starts[g]; f_e = ldf_starts[g + 1]
            R_ldf = data.R_ldf_flat[f_s:f_e].view(ns, ns).to(device)
            X_ldf = data.X_ldf_flat[f_s:f_e].view(ns, ns).to(device)

            # Get slack index and voltage
            si_tensor = data.slack_idx
            slack_idx_g = int(si_tensor[g].item() if si_tensor.dim() > 0 and si_tensor.shape[0] > 1
                              else si_tensor.item())

            v0_sq_tensor = data.v0_sq
            v0_sq = float(v0_sq_tensor[g].item() if v0_sq_tensor.dim() > 0 and v0_sq_tensor.shape[0] > 1
                          else v0_sq_tensor.item())

            # Non-slack indices
            local_ns = [i for i in range(N) if i != slack_idx_g]
            ns_t     = torch.tensor(local_ns, dtype=torch.long, device=device)

            # Denormalize GNO output
            v_n   = v_norm[s:e]                          # (N, 3) normalised
            V_phy = v_n * (sy + EPS) + my                # (N, 3) physical [|V|, sinθ, cosθ]

            v_mag_gno = V_phy[:, 0]   # (N,)  |V| from GNO
            sin_th    = V_phy[:, 1]   # (N,)  sin(θ) — preserved
            cos_th    = V_phy[:, 2]   # (N,)  cos(θ) — preserved

            # Compute v_sq from GNO
            v_sq_gno = v_mag_gno ** 2   # (N,)

            # Denormalize P, Q
            x_n   = data.x[s:e]
            p_pu  = x_n[:, 2] * (sx[2] + EPS) + mx[2]
            q_pu  = x_n[:, 3] * (sx[3] + EPS) + mx[3]
            p_inj = -p_pu;  q_inj = -q_pu

            # LinDistFlow prediction
            with torch.no_grad():
                p_ns = p_inj[ns_t]; q_ns = q_inj[ns_t]
                v_sq_ldf = (v0_sq
                            + 2.0 * (R_ldf @ p_ns)
                            + 2.0 * (X_ldf @ q_ns))
                v_sq_ldf = v_sq_ldf.clamp(min=0.64, max=1.21)

            v_sq_c_ns = (v_sq_ns + alpha * (v_sq_ldf - v_sq_ns)).clamp(min=self.eps)

            # Correct |V| magnitude, preserve angles
            v_mag_c = v_mag_gno.clone()
            v_mag_c[ns_t] = v_sq_c_ns.sqrt()

            V_phy_c = torch.stack([v_mag_c, sin_th, cos_th], dim=-1)  # (N, 3)

            # Renormalize
            v_out[s:e] = (V_phy_c - my) / (sy + EPS)

        return v_out

    def extra_repr(self):
        return f"alpha={self.alpha.item():.4f} (fixed, inference-only)"
