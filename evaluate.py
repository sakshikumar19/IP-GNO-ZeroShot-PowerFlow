"""
evaluate.py - Evaluation script for trained models

Evaluates Vanilla GNO, IP-GNO, and baseline models on test data.
Computes physical metrics, Kirchhoff violations, and DistFlow violations.
"""

import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import NormStats, V0_SQ
from models.gno    import VanillaGNO
from models.ip_gno import IPGNO, kirchhoff_residual
from models.baselines import build_baseline, BASELINE_REGISTRY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDIR = Path("outputs")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_ckpt(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt.get("hparams", {})
    return ckpt, {}


def load_models(stats: NormStats):
    v_path = OUTDIR / "vanilla_gno.pt"
    i_path = OUTDIR / "ip_gno.pt"

    v_sd, v_hp = _load_ckpt(v_path) if v_path.exists() else ({}, {})
    vanilla = VanillaGNO(
        in_dim=v_hp.get("in_dim", 10),
        hidden_dim=v_hp.get("hidden_dim", 64),
        T=v_hp.get("T", 5),
    ).to(DEVICE)
    if v_sd:
        vanilla.load_state_dict(v_sd)
        print(f"[evaluate] VanillaGNO loaded  (hidden={v_hp.get('hidden_dim',64)}, T={v_hp.get('T',5)})")
    else:
        print("[evaluate] WARNING: vanilla_gno.pt not found.")

    i_sd, i_hp = _load_ckpt(i_path) if i_path.exists() else ({}, {})
    ip = IPGNO(
        in_dim=i_hp.get("in_dim", 10),
        hidden_dim=i_hp.get("hidden_dim", 64),
        T=i_hp.get("T", 5),
    ).to(DEVICE)
    if i_sd:
        ip.load_state_dict(i_sd, strict=False)
        alpha = i_hp.get("alpha", ip.correction.alpha.item())
        print(f"[evaluate] IP-GNO loaded  (hidden={i_hp.get('hidden_dim',64)}, "
              f"T={i_hp.get('T',5)}, α={alpha:.4f})")
    else:
        print("[evaluate] WARNING: ip_gno.pt not found.")

    return vanilla, ip


def load_baseline_models(stats: NormStats) -> dict:
    """
    Load all baseline checkpoints found in OUTDIR.
    Returns a dict: {arch_name: model}.  Missing checkpoints are skipped
    with a warning — so evaluate.py works even if only some baselines were
    trained (e.g. `python train.py --model gat`).
    """
    models = {}
    for name in BASELINE_REGISTRY:
        path = OUTDIR / f"{name}.pt"
        if not path.exists():
            print(f"[evaluate] {name.upper()} checkpoint not found, skipping.")
            continue
        sd, hp = _load_ckpt(path)
        model = build_baseline(name, T=hp.get("T", 5)).to(DEVICE)
        model.load_state_dict(sd, strict=False)
        print(f"[evaluate] {name.upper()} loaded  "
              f"(params={model.count_params():,}, T={hp.get('T',5)})")
        models[name] = model
    return models


# ---------------------------------------------------------------------------
# Physical metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_physical_metrics(model, graphs, stats, model_tag="",
                              use_correction=False) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(graphs, batch_size=16, shuffle=False, exclude_keys=["ybus"])

    abs_v, abs_th, sq_v, sq_th = [], [], [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        if use_correction:
            pred_n = model(batch, stats=stats)
        else:
            pred_n = model(batch)

        pred_polar = stats.to_polar(pred_n)
        true_polar = stats.to_polar(batch.y)

        dv  = pred_polar[:, 0] - true_polar[:, 0]
        dth = pred_polar[:, 1] - true_polar[:, 1]

        abs_v.append(dv.abs().cpu()); abs_th.append(dth.abs().cpu())
        sq_v.append((dv**2).cpu());   sq_th.append((dth**2).cpu())

    abs_v  = torch.cat(abs_v);  abs_th = torch.cat(abs_th)
    sq_v   = torch.cat(sq_v);   sq_th  = torch.cat(sq_th)

    metrics = {
        "mae_v_pu":    abs_v.mean().item(),
        "rmse_v_pu":   sq_v.mean().sqrt().item(),
        "mae_th_deg":  abs_th.mean().item(),
        "rmse_th_deg": sq_th.mean().sqrt().item(),
        "max_v_pu":    abs_v.max().item(),
        "max_th_deg":  abs_th.max().item(),
    }

    if model_tag:
        print(f"\n[{model_tag}]")
        print(f"  MAE  |V|  = {metrics['mae_v_pu']:.4e} pu   "
              f"RMSE |V|  = {metrics['rmse_v_pu']:.4e} pu")
        print(f"  MAE  θ    = {metrics['mae_th_deg']:.4e} deg  "
              f"RMSE θ    = {metrics['rmse_th_deg']:.4e} deg")
        print(f"  MAX  |V|  = {metrics['max_v_pu']:.4e} pu   "
              f"MAX  θ    = {metrics['max_th_deg']:.4e} deg")

    return metrics


# ---------------------------------------------------------------------------
# Kirchhoff violation analysis  (extended to N models)
# ---------------------------------------------------------------------------

@torch.no_grad()
def kirchhoff_violation_analysis(vanilla, ip, graphs, stats,
                                  extra_models: dict = None):
    """
    Compute mean KCL residual for VanillaGNO, IP-GNO, and any extra baseline
    models (pass as extra_models={name: model}).
    """
    all_models = {"VanillaGNO": (vanilla, False), "IP-GNO": (ip, True)}
    if extra_models:
        for n, m in extra_models.items():
            all_models[n.upper()] = (m, False)   # baselines: no DistFlow correction

    for m in all_models.values():
        m[0].eval()

    results = {name: [] for name in all_models}
    for g in graphs:
        g = g.to(DEVICE)
        for name, (model, use_corr) in all_models.items():
            pred = model(g, stats=stats) if use_corr else model(g)
            results[name].append(kirchhoff_residual(pred, g, stats).item())

    print("\n" + "=" * 60)
    print("Kirchhoff Violation Analysis")
    print("=" * 60)
    import numpy as np
    van_mean = float(np.mean(results["VanillaGNO"]))
    for name, vals in results.items():
        mean_r = float(np.mean(vals)); std_r = float(np.std(vals))
        pct    = 100.0 * (van_mean - mean_r) / (van_mean + 1e-12) if name != "VanillaGNO" else 0.0
        suffix = f"  ({pct:+.1f}% vs Vanilla)" if name != "VanillaGNO" else ""
        print(f"  {name:<14}: mean residual = {mean_r:.4e}  ±{std_r:.4e}{suffix}")
    print("  (Lower = better KCL satisfaction)")
    return {n: float(np.mean(v)) for n, v in results.items()}


# ---------------------------------------------------------------------------
# DistFlow violation analysis  (Proposition 2, extended to N models)
# ---------------------------------------------------------------------------

@torch.no_grad()
def distflow_violation_analysis(vanilla, ip, graphs, stats,
                                 extra_models: dict = None):
    """
    For each graph measure mean |v_sq_pred − v_sq_ldf| over non-slack nodes.
    Baseline models (no correction) are expected to score higher than IP-GNO.
    """
    all_models = {"VanillaGNO": (vanilla, False), "IP-GNO": (ip, True)}
    if extra_models:
        for n, m in extra_models.items():
            all_models[n.upper()] = (m, False)

    for m in all_models.values():
        m[0].eval()

    results = {name: [] for name in all_models}

    for g in graphs:
        g  = g.to(DEVICE)
        N  = g.num_nodes
        ns = N - 1
        si = int(g.slack_idx.item())

        R = g.R_ldf_flat.view(ns, ns).to(DEVICE)
        X = g.X_ldf_flat.view(ns, ns).to(DEVICE)
        v0_sq = float(g.v0_sq.item())

        non_slack = [i for i in range(N) if i != si]
        ns_t = torch.tensor(non_slack, device=DEVICE)

        p_pu = stats.denormalise_x_col(g.x[:, 2], 2)
        q_pu = stats.denormalise_x_col(g.x[:, 3], 3)
        p_inj = -p_pu[ns_t]; q_inj = -q_pu[ns_t]
        v_sq_ldf = v0_sq + 2.0 * (R @ p_inj) + 2.0 * (X @ q_inj)

        def v_sq_from_pred(v_norm):
            v_phys = stats.denormalise_y(v_norm)
            return (v_phys[:, 0]**2)[ns_t]

        for name, (model, use_corr) in all_models.items():
            pred = model(g, stats=stats) if use_corr else model(g)
            viol = (v_sq_from_pred(pred) - v_sq_ldf).abs().mean().item()
            results[name].append(viol)

    print("\n" + "=" * 60)
    print("LinDistFlow Violation Analysis  (Proposition 2)")
    print("=" * 60)
    van_mean = float(np.mean(results["VanillaGNO"]))
    for name, vals in results.items():
        mean_v = float(np.mean(vals)); std_v = float(np.std(vals))
        pct    = 100.0 * (van_mean - mean_v) / (van_mean + 1e-12) if name != "VanillaGNO" else 0.0
        suffix = f"  ({pct:+.1f}% vs Vanilla)" if name != "VanillaGNO" else ""
        print(f"  {name:<14}: mean |Δv²| = {mean_v:.4e}  ±{std_v:.4e}{suffix}")
    print("  (Lower = closer to LinDistFlow manifold)")
    return {n: float(np.mean(v)) for n, v in results.items()}


# ---------------------------------------------------------------------------
# Comparison table  (variable number of models)
# ---------------------------------------------------------------------------

def print_comparison_table(metrics_dict: dict, grid_size: str = "69"):
    """Print comparison table with percentage improvements vs VanillaGNO."""
    keys = [
        ("mae_v_pu",    "|V| MAE (pu)"),
        ("rmse_v_pu",   "|V| RMSE (pu)"),
        ("mae_th_deg",  "θ MAE (deg)"),
        ("rmse_th_deg", "θ RMSE (deg)"),
        ("max_v_pu",    "|V| Max AE (pu)"),
        ("max_th_deg",  "θ Max AE (deg)"),
    ]
    model_names = list(metrics_dict.keys())
    ref_name    = "VanillaGNO"
    col_w       = 15

    header = f"{'Metric':<22}"
    for n in model_names:
        header += f" | {n:>{col_w}}"
    # One Δ% column per non-reference model would be too wide; instead show
    # IP-GNO (full) vs Vanilla as the primary comparison column.
    ip_name = "IP-GNO" if "IP-GNO" in model_names else model_names[-1]
    header += f" | {'Δ% (IP-GNO vs Van)':>20}"

    sep = "=" * len(header)
    print("\n" + sep)
    print(f"Physical Error — Full {grid_size}-Bus Unseen Configurations")
    print(sep)
    print(header)
    print("-" * len(header))

    for key, label in keys:
        ref_val = metrics_dict[ref_name][key]
        row     = f"{label:<22}"
        for n in model_names:
            row += f" | {metrics_dict[n][key]:>{col_w}.4e}"
        ip_val  = metrics_dict[ip_name][key]
        pct     = 100.0 * (ref_val - ip_val) / (ref_val + 1e-12)
        arrow   = "↓" if pct > 0 else "↑"
        row    += f" | {pct:>+18.1f}% {arrow}"
        print(row)

    print("-" * len(header))
    # Per-model Δ% summary line
    print(f"\n  Δ% vs VanillaGNO  (positive = better than VanillaGNO):")
    for name in model_names:
        if name == ref_name:
            continue
        v_mae = metrics_dict[ref_name]["mae_v_pu"]
        m_mae = metrics_dict[name]["mae_v_pu"]
        pct   = 100.0 * (v_mae - m_mae) / (v_mae + 1e-12)
        print(f"    {name:<22}: |V| MAE {pct:>+6.1f}%")


# ---------------------------------------------------------------------------
# Zero-shot evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def zero_shot_evaluation(vanilla, ip, full_test_g, stats, extra_models=None):
    """
    Zero-shot topological generalisation (Proposition 3).
    All models were trained on 20-25 bus subgraphs; tested on the full grid.
    Includes all available models for a complete comparison.
    """
    print("\n" + "=" * 60)
    print("Zero-Shot Topological Generalisation  (Proposition 3)")
    print("=" * 60)
    print("Train: subgraphs 20–25 buses  │  Test: full grid graphs\n")
    n_bus = full_test_g[0].num_nodes if full_test_g else "N"

    all_models = [
        ("VanillaGNO",       vanilla, False),
        ("IP-GNO (no corr)", ip,      False),
        ("IP-GNO",           ip,      True),
    ]
    if extra_models:
        for n, m in extra_models.items():
            all_models.append((n.upper(), m, False))

    for name, model, use_corr in all_models:
        compute_physical_metrics(model, full_test_g, stats,
                                 model_tag=f"{name} on full {n_bus}-bus",
                                 use_correction=use_corr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    splits_path = OUTDIR / "data_splits.pt"
    stats_path  = OUTDIR / "norm_stats.pt"

    if not splits_path.exists() or not stats_path.exists():
        print("ERROR: Run train.py first.")
        sys.exit(1)

    splits = torch.load(splits_path, map_location="cpu", weights_only=False)
    stats  = NormStats.load(str(stats_path))

    test_g      = splits["test_g"]
    full_test_g = splits["full_test_g"]

    vanilla, ip      = load_models(stats)
    baseline_models  = load_baseline_models(stats)

    n_bus = full_test_g[0].num_nodes if full_test_g else "N"

    # ------------------------------------------------------------------
    # 1. Subgraph test  (no correction for any model)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Subgraph Test Set (Unseen Configurations, 20–25 buses)")
    print("=" * 70)
    sub_metrics = {}
    sub_metrics["VanillaGNO"]       = compute_physical_metrics(vanilla, test_g, stats, "VanillaGNO",        use_correction=False)
    sub_metrics["IP-GNO (no corr)"] = compute_physical_metrics(ip,      test_g, stats, "IP-GNO (no corr)", use_correction=False)
    for bname, bmodel in baseline_models.items():
        sub_metrics[bname.upper()] = compute_physical_metrics(bmodel, test_g, stats, bname.upper(), use_correction=False)

    # ------------------------------------------------------------------
    # 2. Full-graph test
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Full {n_bus}-Bus Test Graphs (Unseen Configurations)")
    print("=" * 70)
    full_metrics = {}
    full_metrics["VanillaGNO"]       = compute_physical_metrics(vanilla, full_test_g, stats, "VanillaGNO",        use_correction=False)
    full_metrics["IP-GNO (no corr)"] = compute_physical_metrics(ip,      full_test_g, stats, "IP-GNO (no corr)", use_correction=False)
    full_metrics["IP-GNO"]           = compute_physical_metrics(ip,      full_test_g, stats, "IP-GNO",            use_correction=True)
    for bname, bmodel in baseline_models.items():
        full_metrics[bname.upper()] = compute_physical_metrics(bmodel, full_test_g, stats, bname.upper(), use_correction=False)

    # ------------------------------------------------------------------
    # 3. Comparison table + ranking
    # ------------------------------------------------------------------
    print_comparison_table(full_metrics, grid_size=str(n_bus))

    print("\n" + "=" * 60)
    print("RANKING  (full-graph test, |V| MAE pu — lower is better)")
    print("=" * 60)
    ranked = sorted(full_metrics.items(), key=lambda kv: kv[1]["mae_v_pu"])
    ref_mae = full_metrics["VanillaGNO"]["mae_v_pu"]
    for rank, (name, m) in enumerate(ranked, 1):
        pct = 100.0 * (ref_mae - m["mae_v_pu"]) / (ref_mae + 1e-12)
        sign = f"({pct:+.1f}% vs Vanilla)" if name != "VanillaGNO" else ""
        print(f"  #{rank}  {name:<20}  MAE |V| = {m['mae_v_pu']:.4e} pu   RMSE |V| = {m['rmse_v_pu']:.4e} pu  {sign}")

    # ------------------------------------------------------------------
    # 4. Kirchhoff violation  (all models)
    # ------------------------------------------------------------------
    kirchhoff_violation_analysis(vanilla, ip, full_test_g, stats,
                                  extra_models=baseline_models)

    # ------------------------------------------------------------------
    # 5. DistFlow violation  (Proposition 2)
    # ------------------------------------------------------------------
    distflow_violation_analysis(vanilla, ip, full_test_g, stats,
                                 extra_models=baseline_models)

    # ------------------------------------------------------------------
    # 6. Zero-shot generalisation  (Proposition 3) — all models
    # ------------------------------------------------------------------
    zero_shot_evaluation(vanilla, ip, full_test_g, stats,
                         extra_models=baseline_models)

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    summary = {
        "subgraph":  sub_metrics,
        "full_test": full_metrics,
    }
    torch.save(summary, OUTDIR / "eval_summary.pt")
    print("\nFull evaluation summary saved to outputs/eval_summary.pt")


if __name__ == "__main__":
    main()
