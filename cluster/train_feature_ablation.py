"""
Full feature ablation: retrain with each of the 13 features dropped individually.

Modes (CLI):
  --task-id N           Train config N (0=baseline, 1-13=drop feature N-1)
  --merge               Load all result JSONs, plot bar chart sorted by impact
  (no args)             Run all 14 configs sequentially — for local testing only

Requires features_cache.npz — run train_arch_search.py --preprocess first.
SLURM array: 0-13
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ============================================================
# FIXED HYPERPARAMETERS  (same winning config as train.py)
# ============================================================
WINNING_WEIGHT = 2.0
WINNING_LR     = 1e-4
EPOCHS         = 100
BATCH_SIZE     = 32768
FULL_IN_DIM    = 13
HIDDEN_DIM     = 128
DEPTH          = 4

# Feature names match the column order in extract_triangle_features()
FEATURE_NAMES = [
    "area",        # 0  — normalized triangle area
    "bias",        # 1  — constant 1.0 input
    "normal_x",    # 2  — face normal X
    "normal_y",    # 3  — face normal Y
    "normal_z",    # 4  — face normal Z
    "dist_center", # 5  — 3-D distance to part centroid
    "radial_dist", # 6  — XY radial distance to centroid
    "rel_height",  # 7  — relative Z height
    "edge_a",      # 8  — edge A length
    "edge_b",      # 9  — edge B length
    "edge_c",      # 10 — edge C length
    "outerness",   # 11 — proximity to convex hull surface
    "opp_quality", # 12 — opposite-face ray-cast score
]

# task-id 0 = baseline (all 13), task-id 1-13 = drop feature index 0-12
ABLATION_CONFIGS = [None] + list(range(FULL_IN_DIM))  # 14 configs

CACHE_PATH   = "features_cache.npz"
RESULTS_DIR  = "ablation_results"
PLOT_PATH    = "feature_ablation.png"
SUMMARY_PATH = "feature_ablation_results.json"


# ============================================================
# HELPERS
# ============================================================
def _rot_cols(drop_feat):
    """Return (nx_col, ny_col) for rotation augmentation after dropping drop_feat.
    Returns None when normal_x or normal_y was the dropped feature."""
    if drop_feat in (2, 3):
        return None
    # Dropping any column before index 2 or 3 shifts those indices down
    nx = 2 - (1 if drop_feat is not None and drop_feat < 2 else 0)
    ny = 3 - (1 if drop_feat is not None and drop_feat < 3 else 0)
    return (nx, ny)


def _config_label(drop_feat):
    return "baseline" if drop_feat is None else f"drop_{FEATURE_NAMES[drop_feat]}"


# ============================================================
# DATASET
# ============================================================
class AblationDataset(Dataset):
    def __init__(self, X, Y, augment_rot=False, rot_cols=(2, 3)):
        self.X = X
        self.Y = Y
        self.augment_rot = augment_rot
        self.rot_cols = rot_cols  # (nx_col, ny_col) in the current (possibly reduced) X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.Y[idx]
        if self.augment_rot:
            i, j = self.rot_cols
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            x[i], x[j] = x[i] * c - x[j] * s, x[i] * s + x[j] * c
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ============================================================
# MODEL
# ============================================================
class ClampSupportNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        layers = [nn.Linear(in_dim, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2)]
        for _ in range(DEPTH - 1):
            layers += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2)]
        layers.append(nn.Linear(HIDDEN_DIM, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


# ============================================================
# EVALUATION
# ============================================================
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = (torch.sigmoid(model(X)) > 0.5).float()
            y_c, p_c = y[:, 0], preds[:, 0]
            tp += ((p_c == 1) & (y_c == 1)).sum().item()
            fp += ((p_c == 1) & (y_c == 0)).sum().item()
            fn += ((p_c == 0) & (y_c == 1)).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return f1, precision, recall


# ============================================================
# SINGLE-TASK MODE  (--task-id N)
# ============================================================
def run_task(task_id):
    drop_feat = ABLATION_CONFIGS[task_id]
    label     = _config_label(drop_feat)
    print(f"Task {task_id}: {label}", flush=True)

    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"Cache not found: {CACHE_PATH}. "
            "Run train_arch_search.py --preprocess first."
        )

    cache   = np.load(CACHE_PATH)
    train_X = cache["train_X"].copy()
    train_Y = cache["train_Y"]
    val_X   = cache["val_X"].copy()
    val_Y   = cache["val_Y"]
    test_X  = cache["test_X"].copy()
    test_Y  = cache["test_Y"]

    if drop_feat is not None:
        train_X = np.delete(train_X, drop_feat, axis=1)
        val_X   = np.delete(val_X,   drop_feat, axis=1)
        test_X  = np.delete(test_X,  drop_feat, axis=1)

    in_dim   = train_X.shape[1]
    rc       = _rot_cols(drop_feat)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | in_dim={in_dim} | rot_aug={'yes' if rc else 'no'}", flush=True)

    loader_kw    = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    train_loader = DataLoader(
        AblationDataset(train_X, train_Y, augment_rot=(rc is not None), rot_cols=rc or (2, 3)),
        shuffle=True, persistent_workers=True, **loader_kw)
    val_loader   = DataLoader(AblationDataset(val_X,   val_Y),  shuffle=False, **loader_kw)
    test_loader  = DataLoader(AblationDataset(test_X,  test_Y), shuffle=False, **loader_kw)

    model     = ClampSupportNet(in_dim=in_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=WINNING_LR)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([WINNING_WEIGHT, WINNING_WEIGHT]).to(device))

    best_f1 = 0.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

        f1, p, r = evaluate(model, val_loader, device)
        print(f"  epoch {epoch+1:3d} | val F1={f1:.4f}  P={p:.4f}  R={r:.4f}", flush=True)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_f1, test_p, test_r = evaluate(model, test_loader, device)
    print(f"\nTest F1={test_f1:.4f}  P={test_p:.4f}  R={test_r:.4f}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "label": label,
        "drop_feature_idx":  drop_feat,
        "drop_feature_name": FEATURE_NAMES[drop_feat] if drop_feat is not None else None,
        "in_dim": in_dim,
        "f1": test_f1, "precision": test_p, "recall": test_r,
    }
    out_path = os.path.join(RESULTS_DIR, f"{label}.json")
    with open(out_path, "w") as fout:
        json.dump(out, fout, indent=2)
    print(f"Saved → {out_path}", flush=True)


# ============================================================
# MERGE + PLOT  (--merge)
# ============================================================
def run_merge():
    results = {}
    for drop_feat in ABLATION_CONFIGS:
        label = _config_label(drop_feat)
        path  = os.path.join(RESULTS_DIR, f"{label}.json")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        with open(path) as f:
            results[label] = json.load(f)

    with open(SUMMARY_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved → {SUMMARY_PATH}")

    baseline_f1 = results.get("baseline", {}).get("f1", None)

    rows = []
    for label, v in results.items():
        drop = (baseline_f1 - v["f1"]) if (baseline_f1 is not None and label != "baseline") else 0.0
        rows.append((label, v["f1"], v["precision"], v["recall"], drop))
    rows.sort(key=lambda x: x[4], reverse=True)

    labels = [r[0] for r in rows]
    f1s    = [r[1] for r in rows]
    colors = ["steelblue" if r[0] == "baseline" else
              ("tomato"        if r[4] > 0.02 else
               "lightsalmon"   if r[4] > 0.005 else
               "lightsteelblue")
              for r in rows]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(labels, f1s, color=colors)
    if baseline_f1 is not None:
        ax.axvline(baseline_f1, color="black", linestyle="--", linewidth=1.5,
                   label=f"Baseline F1 = {baseline_f1:.3f}")
    ax.set_xlabel("Test F1", fontsize=12)
    ax.set_title("Feature Ablation Study — Test F1 When Each Feature Is Dropped",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.1)
    for bar, (_, f1, _, _, drop) in zip(bars, rows):
        sign = "−" if drop >= 0 else "+"
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{f1:.3f}  ({sign}{abs(drop):.3f})",
                va="center", fontsize=9)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Plot saved → {PLOT_PATH}")

    print("\n" + "=" * 62)
    print(f"{'Config':<25} {'F1':>8} {'Prec':>8} {'Rec':>8} {'F1 Drop':>10}")
    print("-" * 62)
    for label, f1, p, r, drop in rows:
        sign = "−" if drop >= 0 else "+"
        print(f"{label:<25} {f1:>8.4f} {p:>8.4f} {r:>8.4f}  {sign}{abs(drop):>8.4f}")
    print("=" * 62)


# ============================================================
# SEQUENTIAL LOCAL MODE  (no args)
# ============================================================
def run_local():
    print("Local sequential mode (all 14 ablation configs).")
    for task_id in range(len(ABLATION_CONFIGS)):
        run_task(task_id)
    run_merge()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--task-id", type=int, metavar="N",
                       help="Train config N (0-13) from cache (for SLURM array).")
    group.add_argument("--merge", action="store_true",
                       help="Merge result JSONs and plot bar chart.")
    args = parser.parse_args()

    if args.task_id is not None:
        if not (0 <= args.task_id < len(ABLATION_CONFIGS)):
            raise ValueError(f"--task-id must be 0–{len(ABLATION_CONFIGS)-1}")
        run_task(args.task_id)
    elif args.merge:
        run_merge()
    else:
        run_local()
