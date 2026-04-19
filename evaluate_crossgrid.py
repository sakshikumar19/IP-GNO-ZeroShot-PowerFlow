"""
evaluate_crossgrid.py - Cross-grid zero-shot evaluation

Tests models trained on one grid (e.g., 33-bus) on a different unseen grid (e.g., 69-bus).
No fine-tuning or access to target grid data during training.
"""

import argparse
import sys
import random
from pathlib import Path

import torch
import numpy as np
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import (
    NormStats, generate_dataset, normalise_splits,
    NODE_DIM, OUT_DIM,
)
from models.gno      import VanillaGNO
from models.ip_gno   import IPGNO, kirchhoff_residual
from models.baselines import build_baseline, BASELINE_REGISTRY
from evaluate import (
    compute_physical_metrics,
    kirchhoff_violation_analysis,
    distflow_violation_analysis,
    print_comparison_table,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Load models from training outdir
# ---------------------------------------------------------------------------

def _load_ckpt(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt.get("hparams", {})
    return ckpt, {}


def load_all_models(outdir: Path):
    """Load VanillaGNO, IP-GNO, and any baseline checkpoints from outdir."""
    v_path = outdir / "vanilla_gno.pt"
    i_path = outdir / "ip_gno.pt"

    v_sd, v_hp = _load_ckpt(v_path) if v_path.exists() else ({}, {})
    vanilla = VanillaGNO(
        in_dim=v_hp.get("in_dim", NODE_DIM),
        hidden_dim=v_hp.get("hidden_dim", 64),
        T=v_hp.get("T", 5),
    ).to(DEVICE)
    if v_sd:
        vanilla.load_state_dict(v_sd)
        print(f"[cross-grid] VanillaGNO loaded from {v_path}")
    else:
        print(f"[cross-grid] WARNING: {v_path} not found")

    i_sd, i_hp = _load_ckpt(i_path) if i_path.exists() else ({}, {})
    ip = IPGNO(
        in_dim=i_hp.get("in_dim", NODE_DIM),
        hidden_dim=i_hp.get("hidden_dim", 64),
        T=i_hp.get("T", 5),
    ).to(DEVICE)
    if i_sd:
        ip.load_state_dict(i_sd, strict=False)
        print(f"[cross-grid] IP-GNO loaded from {i_path}  "
              f"(α={ip.correction.alpha.item():.3f})")
    else:
        print(f"[cross-grid] WARNING: {i_path} not found")

    baselines = {}
    for name in BASELINE_REGISTRY:
        path = outdir / f"{name}.pt"
        if not path.exists():
            continue
        sd, hp = _load_ckpt(path)
        model = build_baseline(name, T=hp.get("T", 5)).to(DEVICE)
        model.load_state_dict(sd, strict=False)
        baselines[name] = model
        print(f"[cross-grid] {name.upper()} loaded from {path}")

    return vanilla, ip, baselines


# ---------------------------------------------------------------------------
# Generate test graphs for the *target* grid
# ---------------------------------------------------------------------------

def build_target_test_graphs(test_grid: str, num_configs: int, scenarios: int,
                              seed: int = 99):
    """
    Generate full-graph test data from the target grid.

    Uses a different seed from training (default 99 vs training default 42)
    to ensure topology configurations are genuinely unseen.

    Returns a list of full-graph Data objects (NOT subgraphs).
    """
    print(f"\n[cross-grid] Generating {test_grid}-bus test graphs "
          f"({num_configs} configs × {scenarios} scenarios, seed={seed}) ...")

    _, _, _, full_test, _ = generate_dataset(
        grid=test_grid,
        num_configs=num_configs,
        scenarios_per_cfg=scenarios,
        subgraphs_per_scen=1,    # we only want full graphs
        train_frac=0.0,          # no train split needed
        val_frac=0.0,
        seed=seed,
        include_ybus=False,
    )

    # If generate_dataset puts everything in train split when fracs are 0,
    # fall back to generating and taking all full graphs directly
    if len(full_test) == 0:  # should not happen with train_frac=0
        print("[cross-grid] Retrying with full split allocation ...")
        tr, _, _, ft, _ = generate_dataset(
            grid=test_grid,
            num_configs=num_configs,
            scenarios_per_cfg=scenarios,
            subgraphs_per_scen=1,
            train_frac=1.0,
            val_frac=0.0,
            seed=seed,
            include_ybus=False,
        )
        # Use train_full — which is the full graphs from the "training" configs
        # In this context we just want full graphs from the target grid
        full_test = tr   # repurpose: these are the target-grid full graphs

    print(f"[cross-grid] Target grid full graphs: {len(full_test)}")
    return full_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-grid zero-shot evaluation for IP-GNO v4"
    )
    p.add_argument("--train_grid",   default="33", choices=["33", "69"],
                   help="Grid the models were trained on")
    p.add_argument("--test_grid",    default="69", choices=["33", "69"],
                   help="Grid to evaluate on (must differ from train_grid)")
    p.add_argument("--train_outdir", default="outputs_33",
                   help="Directory containing trained model checkpoints and norm_stats.pt")
    p.add_argument("--num_configs",  type=int, default=20,
                   help="Number of topology configurations to generate for target grid")
    p.add_argument("--scenarios",    type=int, default=5,
                   help="Load scenarios per topology configuration")
    p.add_argument("--seed",         type=int, default=99,
                   help="RNG seed for target grid generation (use different from train seed)")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.train_outdir)

    if args.train_grid == args.test_grid:
        print("WARNING: train_grid == test_grid.  Use evaluate.py for within-grid evaluation.")

    print(f"\n{'='*70}")
    print(f"Cross-Grid Zero-Shot: Train={args.train_grid}-bus  →  Test={args.test_grid}-bus")
    print(f"Models from: {outdir}")
    print(f"{'='*70}")

    # ── 1. Load training norm stats (no 69-bus data used) ────────────────
    stats_path = outdir / "norm_stats.pt"
    if not stats_path.exists():
        print(f"ERROR: {stats_path} not found. Run train.py --grid {args.train_grid} "
              f"--outdir {args.train_outdir} first.")
        sys.exit(1)
    stats = NormStats.load(str(stats_path))
    print(f"\n[cross-grid] Norm stats loaded from {stats_path}")
    print(f"  my (target mean) = {stats.my.tolist()}")
    print(f"  sy (target std)  = {stats.sy.tolist()}")

    # ── 2. Load models ────────────────────────────────────────────────────
    vanilla, ip, baseline_models = load_all_models(outdir)

    # ── 3. Generate target-grid test graphs ───────────────────────────────
    raw_test = build_target_test_graphs(
        args.test_grid, args.num_configs, args.scenarios, seed=args.seed
    )

    # ── 4. Normalise using SOURCE grid stats ──────────────────────────────
    # Key zero-shot step: use trained stats from source grid to normalize target grid
    [test_g] = normalise_splits(stats, raw_test)
    print(f"\n[cross-grid] Normalised {len(test_g)} target-grid graphs "
          f"using {args.train_grid}-bus statistics")

    n_bus = test_g[0].num_nodes if test_g else "?"

    # ── 5. Physical metrics ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Cross-Grid Results: {args.train_grid}-bus trained → {n_bus}-bus tested")
    print(f"(compare with within-grid results from evaluate.py)")
    print(f"{'='*70}")

    full_metrics = {}
    full_metrics["VanillaGNO"]       = compute_physical_metrics(
        vanilla, test_g, stats, f"VanillaGNO", use_correction=False)
    full_metrics["IP-GNO (no corr)"] = compute_physical_metrics(
        ip, test_g, stats, f"IP-GNO (no corr)", use_correction=False)
    full_metrics["IP-GNO"]           = compute_physical_metrics(
        ip, test_g, stats, f"IP-GNO", use_correction=True)
    for bname, bmodel in baseline_models.items():
        full_metrics[bname.upper()] = compute_physical_metrics(
            bmodel, test_g, stats, bname.upper(), use_correction=False)

    # ── 6. Comparison table ───────────────────────────────────────────────
    print_comparison_table(full_metrics, grid_size=f"{n_bus} (cross-grid {args.train_grid}→{args.test_grid})")

    print(f"\n{'='*60}")
    print(f"RANKING  (cross-grid, |V| MAE — lower is better)")
    print(f"{'='*60}")
    ranked  = sorted(full_metrics.items(), key=lambda kv: kv[1]["mae_v_pu"])
    ref_mae = full_metrics["VanillaGNO"]["mae_v_pu"]
    for rank, (name, m) in enumerate(ranked, 1):
        pct  = 100.0 * (ref_mae - m["mae_v_pu"]) / (ref_mae + 1e-12)
        sign = f"({pct:+.1f}% vs Vanilla)" if name != "VanillaGNO" else ""
        print(f"  #{rank}  {name:<22}  MAE |V| = {m['mae_v_pu']:.4e} pu  "
              f"RMSE = {m['rmse_v_pu']:.4e} pu  {sign}")

    # ── 7. KCL + DistFlow violations ─────────────────────────────────────
    kirchhoff_violation_analysis(vanilla, ip, test_g, stats,
                                 extra_models=baseline_models)
    distflow_violation_analysis(vanilla, ip, test_g, stats,
                                extra_models=baseline_models)

    # ── 8. Save ───────────────────────────────────────────────────────────
    out_path = outdir / f"crossgrid_{args.train_grid}to{args.test_grid}_results.pt"
    torch.save({
        "train_grid":  args.train_grid,
        "test_grid":   args.test_grid,
        "full_metrics": full_metrics,
        "n_test_graphs": len(test_g),
    }, out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
