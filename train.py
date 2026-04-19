"""
train.py - Training script for GNO models

Trains Vanilla GNO, IP-GNO, and baseline models on power grid datasets.
Supports training on IEEE 33-bus or 69-bus systems.
"""

import argparse, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset   import fit_normalisation, generate_dataset, normalise_splits, NormStats
from models.gno     import VanillaGNO
from models.ip_gno  import IPGNO
from models.baselines import build_baseline, BASELINE_REGISTRY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)


def voltage_mse(pred, target):
    """Mean squared error for voltage predictions."""
    return F.mse_loss(pred, target)


# Vanilla GNO training functions

def train_epoch_vanilla(model, loader, opt):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        loss = voltage_mse(model(batch), batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        total += loss.item() * batch.num_graphs
        n     += batch.num_graphs
    return total / n


@torch.no_grad()
def eval_epoch_vanilla(model, loader):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        total += voltage_mse(model(batch), batch.y).item() * batch.num_graphs
        n     += batch.num_graphs
    return total / n


def train_vanilla(train_g, val_g, hidden_dim=64, T=5, epochs=300,
                  lr=1e-3, batch_size=16, patience=40):
    """Train Vanilla GNO model."""
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    model = VanillaGNO(hidden_dim=hidden_dim, T=T).to(DEVICE)
    print(f"\n[VanillaGNO] params = {model.count_params():,}")
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    tr_ld = DataLoader(train_g, batch_size=batch_size, shuffle=True,  exclude_keys=["ybus"])
    va_ld = DataLoader(val_g,   batch_size=batch_size, shuffle=False, exclude_keys=["ybus"])

    best_val, no_improv, best_sd = float("inf"), 0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch_vanilla(model, tr_ld, opt)
        va = eval_epoch_vanilla(model, va_ld)
        sched.step()
        if va < best_val:
            best_val, no_improv = va, 0
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improv += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:4d}/{epochs}  train={tr:.4e}  val={va:.4e}"
                  f"  lr={opt.param_groups[0]['lr']:.2e}  t={time.time()-t0:.1f}s")
        if no_improv >= patience:
            print(f"  Early stop at epoch {ep}  (best val={best_val:.4e})")
            break

    model.load_state_dict(best_sd)
    torch.save({
        "state_dict": best_sd,
        "hparams": {"in_dim": 10, "hidden_dim": hidden_dim, "T": T},
    }, OUTDIR / "vanilla_gno.pt")
    print(f"[VanillaGNO] best val loss = {best_val:.4e}  (saved)")
    return model


# IP-GNO training functions

def train_epoch_ipgno(model, loader, opt, stats):
    # During training, we use the raw GNO output without DistFlow correction
    # to keep gradients stable. Correction is only applied at inference time.
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        pred = model.train_forward(batch)   # raw GNO output, no correction
        loss = voltage_mse(pred, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        total += loss.item() * batch.num_graphs
        n     += batch.num_graphs
    return total / n


@torch.no_grad()
def eval_epoch_ipgno(model, loader, stats):
    # Validation also uses raw GNO output for consistent early stopping.
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        pred  = model.train_forward(batch)   # raw GNO, no correction
        total += voltage_mse(pred, batch.y).item() * batch.num_graphs
        n     += batch.num_graphs
    return total / n


def train_ipgno(train_g, val_g, stats, hidden_dim=64, T=5,
                epochs=300, lr=1e-3, batch_size=16, patience=40):
    """Train IP-GNO model."""
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    model = IPGNO(hidden_dim=hidden_dim, T=T).to(DEVICE)
    print(f"\n[IP-GNO] params = {model.count_params():,}")
    print(f"  DistFlow correction: α = {model.correction.alpha.item():.4f} (fixed, inference-only)")

    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    tr_ld = DataLoader(train_g, batch_size=batch_size, shuffle=True,  exclude_keys=["ybus"])
    va_ld = DataLoader(val_g,   batch_size=batch_size, shuffle=False, exclude_keys=["ybus"])

    best_val, no_improv, best_sd = float("inf"), 0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch_ipgno(model, tr_ld, opt, stats)
        va = eval_epoch_ipgno(model, va_ld, stats)
        sched.step()
        if va < best_val:
            best_val, no_improv = va, 0
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improv += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:4d}/{epochs}  train={tr:.4e}  val={va:.4e}"
                  f"  lr={opt.param_groups[0]['lr']:.2e}  t={time.time()-t0:.1f}s")
        if no_improv >= patience:
            print(f"  Early stop at epoch {ep}  (best val={best_val:.4e})")
            break

    model.load_state_dict(best_sd)
    torch.save({
        "state_dict": best_sd,
        "hparams": {
            "in_dim": 10, "hidden_dim": hidden_dim, "T": T,
            "alpha": float(model.correction.alpha.item()),
        },
    }, OUTDIR / "ip_gno.pt")
    print(f"[IP-GNO] best val = {best_val:.4e}  (saved)")
    return model


# Baseline GNN training functions

def train_epoch_baseline(model, loader, opt):
    """Generic training epoch for baseline models."""
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        loss = voltage_mse(model(batch), batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        total += loss.item() * batch.num_graphs
        n     += batch.num_graphs
    return total / n


@torch.no_grad()
def eval_epoch_baseline(model, loader):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        total += voltage_mse(model(batch), batch.y).item() * batch.num_graphs
        n     += batch.num_graphs
    return total / n


def train_baseline(name, train_g, val_g, hidden_dim=None, T=5,
                   epochs=300, lr=1e-3, batch_size=16, patience=40):
    """
    Train a baseline GNN model.
    If hidden_dim is None, uses default parameters for that model.
    """
    torch.manual_seed(0); torch.cuda.manual_seed(0)

    kwargs = {"T": T}
    if hidden_dim is not None:
        kwargs["hidden_dim"] = hidden_dim

    model = build_baseline(name, **kwargs).to(DEVICE)
    print(f"\n[{name.upper()}] params = {model.count_params():,}")

    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    tr_ld = DataLoader(train_g, batch_size=batch_size, shuffle=True,  exclude_keys=["ybus"])
    va_ld = DataLoader(val_g,   batch_size=batch_size, shuffle=False, exclude_keys=["ybus"])

    best_val, no_improv, best_sd = float("inf"), 0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch_baseline(model, tr_ld, opt)
        va = eval_epoch_baseline(model, va_ld)
        sched.step()
        if va < best_val:
            best_val, no_improv = va, 0
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improv += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:4d}/{epochs}  train={tr:.4e}  val={va:.4e}"
                  f"  lr={opt.param_groups[0]['lr']:.2e}  t={time.time()-t0:.1f}s")
        if no_improv >= patience:
            print(f"  Early stop at epoch {ep}  (best val={best_val:.4e})")
            break

    model.load_state_dict(best_sd)
    ckpt = {
        "state_dict": best_sd,
        "hparams": {
            "in_dim": 10, "hidden_dim": model.convs[0].in_channels
            if hasattr(model, "convs") and hasattr(model.convs[0], "in_channels")
            else kwargs.get("hidden_dim", 64),
            "T": T,
            "arch": name,
        },
    }
    torch.save(ckpt, OUTDIR / f"{name}.pt")
    print(f"[{name.upper()}] best val = {best_val:.4e}  (saved to {OUTDIR / f'{name}.pt'})")
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="both",
                    choices=["vanilla", "ipgno", "both",
                             "gat", "gine", "sage",
                             "baselines", "all"])
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--hidden",      type=int,   default=64)
    p.add_argument("--T",           type=int,   default=5)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--patience",    type=int,   default=40)
    p.add_argument("--num_configs", type=int,   default=120)
    p.add_argument("--scenarios",   type=int,   default=8)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--grid",        type=str,   default="69", choices=["33", "69"])
    p.add_argument("--outdir",      type=str,   default="outputs",
                   help="Output directory (use separate dirs per seed/grid)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device: {DEVICE}")
    print(f"Args: {vars(args)}")
    print(f"Grid: IEEE {args.grid}-bus\n")

    global OUTDIR
    OUTDIR = Path(args.outdir)
    OUTDIR.mkdir(exist_ok=True)

    print("Generating dataset …")
    train_raw, val_raw, test_raw, full_test_raw, train_full_raw = generate_dataset(
        grid=args.grid, num_configs=args.num_configs,
        scenarios_per_cfg=args.scenarios, seed=args.seed, include_ybus=True,
    )
    # Fit normalization stats on full graphs to capture the full voltage range
    stats = fit_normalisation(train_raw, full_train_graphs=train_full_raw)
    stats.save(str(OUTDIR / "norm_stats.pt"))
    train_g, val_g, test_g, full_test_g = normalise_splits(
        stats, train_raw, val_raw, test_raw, full_test_raw)

    print(f"Train: {len(train_g)}  Val: {len(val_g)}  "
          f"Test: {len(test_g)}  Full-test: {len(full_test_g)}\n")

    van_model, ip_model = None, None
    baseline_models: dict = {}   # {name: model}

    # Determine which models to train
    run_vanilla   = args.model in ("vanilla", "both", "all")
    run_ipgno     = args.model in ("ipgno",   "both", "all")
    run_baselines = args.model in ("baselines", "all") or args.model in BASELINE_REGISTRY

    if args.model in BASELINE_REGISTRY:
        # User asked for a single specific baseline
        baselines_to_run = [args.model]
    elif run_baselines:
        baselines_to_run = list(BASELINE_REGISTRY.keys())   # gat, gine, sage
    else:
        baselines_to_run = []

    if run_vanilla:
        van_model = train_vanilla(
            train_g, val_g, hidden_dim=args.hidden, T=args.T,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, patience=args.patience,
        )

    if run_ipgno:
        ip_model = train_ipgno(
            train_g, val_g, stats, hidden_dim=args.hidden, T=args.T,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, patience=args.patience,
        )

    for bname in baselines_to_run:
        baseline_models[bname] = train_baseline(
            bname, train_g, val_g,
            hidden_dim=None,          # use architecture defaults (param-matched)
            T=args.T,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, patience=args.patience,
        )

    print("\n" + "=" * 70)
    print("TEST RESULTS (normalised MSE)  |  sub-graph  |  full-graph")
    print("=" * 70)

    tl = DataLoader(test_g,      batch_size=16, shuffle=False, exclude_keys=["ybus"])
    fl = DataLoader(full_test_g, batch_size=8,  shuffle=False, exclude_keys=["ybus"])

    if van_model:
        s = eval_epoch_vanilla(van_model, tl)
        f = eval_epoch_vanilla(van_model, fl)
        print(f"VanillaGNO     | {s:.4e}      | {f:.4e}")

    if ip_model:
        s = eval_epoch_ipgno(ip_model, tl, stats)
        f = eval_epoch_ipgno(ip_model, fl, stats)
        print(f"IP-GNO         | {s:.4e}      | {f:.4e}")

    for bname, bmodel in baseline_models.items():
        s = eval_epoch_baseline(bmodel, tl)
        f = eval_epoch_baseline(bmodel, fl)
        print(f"{bname.upper():<15}| {s:.4e}      | {f:.4e}")

    torch.save({"train_g": train_g, "val_g": val_g,
                "test_g": test_g, "full_test_g": full_test_g},
               OUTDIR / "data_splits.pt")
    print("Data splits saved to outputs/data_splits.pt")


if __name__ == "__main__":
    main()
