"""
Architecture search: depth x width grid.

Modes (driven by CLI args):
  --preprocess          Extract all features once, save features_cache.npz
  --task-id N           Load cache, train config N, save results/dD_wW.json
  --merge               Load all result JSONs, plot and save heatmap (no training)
  (no args)             Run all 9 configs sequentially — for local testing only
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
import matplotlib.ticker as mticker
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
import random

import features

# ============================================================
# FIXED HYPERPARAMETERS
# ============================================================
WINNING_WEIGHT = 2.0
WINNING_LR     = 1e-4
EPOCHS         = 100
BATCH_SIZE     = 32768
PATIENCE       = 15
IN_DIM         = 13

DEPTHS  = [2, 3, 4]
WIDTHS  = [64, 128, 256]
CONFIGS = [(d, w) for d in DEPTHS for w in WIDTHS]   # 9 entries, index = task-id

LABELS_PATH = "training_set/all_part_labels.json"
CACHE_PATH  = "features_cache.npz"
RESULTS_DIR = "arch_results"
HEATMAP_PATH = "arch_search_heatmap.png"
SUMMARY_PATH = "arch_search_results.json"


# ============================================================
# DATA LOADING (used only in --preprocess mode)
# ============================================================
def _load_part(args):
    path, label_dict = args
    if not os.path.exists(path):
        return None
    try:
        feats, _, _ = features.extract_triangle_features(path)
        y = np.zeros((feats.shape[0], 2), dtype=np.float32)
        clamp_indices = [int(i) for i, lbl in label_dict.items() if lbl == "clamp"]
        if clamp_indices:
            y[clamp_indices, 0] = 1.0
        return {"tri_features": feats, "labels": y}
    except Exception as e:
        print(f"  Error: {os.path.basename(path)}: {e}")
        return None


def run_preprocess(num_cores=16):
    print(f"Loading labels from {LABELS_PATH} ...")
    with open(LABELS_PATH) as f:
        all_labels = json.load(f)

    tasks = list(all_labels.items())
    print(f"Extracting features for {len(tasks)} parts on {num_cores} cores ...")
    with Pool(processes=num_cores) as pool:
        results = pool.map(_load_part, tasks)
    parts = [r for r in results if r is not None]
    print(f"Loaded {len(parts)} / {len(tasks)} parts successfully.")

    random.seed(42)
    random.shuffle(parts)
    n = len(parts)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    def stack(split):
        X = np.vstack([p["tri_features"] for p in split])
        Y = np.vstack([p["labels"]       for p in split])
        return X.astype(np.float32), Y.astype(np.float32)

    train_X, train_Y = stack(parts[:train_end])
    val_X,   val_Y   = stack(parts[train_end:val_end])
    test_X,  test_Y  = stack(parts[val_end:])

    np.savez_compressed(CACHE_PATH,
        train_X=train_X, train_Y=train_Y,
        val_X=val_X,     val_Y=val_Y,
        test_X=test_X,   test_Y=test_Y)

    print(f"Cache saved → {CACHE_PATH}")
    print(f"  train: {len(train_X):,} triangles")
    print(f"  val:   {len(val_X):,} triangles")
    print(f"  test:  {len(test_X):,} triangles")


# ============================================================
# DATASET (works on pre-stacked arrays from cache)
# ============================================================
class CachedDataset(Dataset):
    def __init__(self, X, Y, augment_rot=False):
        self.X = X
        self.Y = Y
        self.augment_rot = augment_rot

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.Y[idx]
        if self.augment_rot:
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            x[2], x[3] = x[2] * c - x[3] * s, x[2] * s + x[3] * c
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ============================================================
# MODEL
# ============================================================
class ClampSupportNet(nn.Module):
    def __init__(self, hidden_dim=128, depth=4):
        super().__init__()
        layers = [nn.Linear(IN_DIM, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
        layers.append(nn.Linear(hidden_dim, 2))
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
# TRAIN ONE CONFIG
# ============================================================
def train_config(depth, width, train_loader, val_loader, test_loader, device, early_stop=True):
    model     = ClampSupportNet(hidden_dim=width, depth=depth).to(device)
    optimizer = optim.Adam(model.parameters(), lr=WINNING_LR)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([WINNING_WEIGHT, WINNING_WEIGHT]).to(device))

    best_f1 = 0.0
    patience_counter = 0
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
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if early_stop and patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}.")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_f1, test_p, test_r = evaluate(model, test_loader, device)
    return test_f1, test_p, test_r


# ============================================================
# SINGLE-TASK MODE  (--task-id N)
# ============================================================
def run_task(task_id, early_stop=True):
    depth, width = CONFIGS[task_id]
    print(f"Task {task_id}: depth={depth}, width={width}", flush=True)

    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"Cache not found: {CACHE_PATH}. Run --preprocess first.")

    cache = np.load(CACHE_PATH)
    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    train_loader = DataLoader(CachedDataset(cache["train_X"], cache["train_Y"], augment_rot=True),
                              shuffle=True, persistent_workers=True, **loader_kw)
    val_loader   = DataLoader(CachedDataset(cache["val_X"],   cache["val_Y"]),   shuffle=False, **loader_kw)
    test_loader  = DataLoader(CachedDataset(cache["test_X"],  cache["test_Y"]),  shuffle=False, **loader_kw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    f1, p, r = train_config(depth, width, train_loader, val_loader, test_loader, device, early_stop=early_stop)
    print(f"\nTest F1={f1:.4f}  P={p:.4f}  R={r:.4f}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {"depth": depth, "width": width, "f1": f1, "precision": p, "recall": r}
    out_path = os.path.join(RESULTS_DIR, f"d{depth}_w{width}.json")
    with open(out_path, "w") as f_out:
        json.dump(out, f_out, indent=2)
    print(f"Saved → {out_path}", flush=True)


# ============================================================
# MERGE + PLOT  (--merge)
# ============================================================
def run_merge():
    f1_grid        = np.full((len(DEPTHS), len(WIDTHS)), np.nan)
    precision_grid = np.full((len(DEPTHS), len(WIDTHS)), np.nan)
    recall_grid    = np.full((len(DEPTHS), len(WIDTHS)), np.nan)
    all_results = {}

    for i, depth in enumerate(DEPTHS):
        for j, width in enumerate(WIDTHS):
            path = os.path.join(RESULTS_DIR, f"d{depth}_w{width}.json")
            if not os.path.exists(path):
                print(f"WARNING: missing result {path}")
                continue
            with open(path) as f:
                v = json.load(f)
            f1_grid[i, j]        = v["f1"]
            precision_grid[i, j] = v["precision"]
            recall_grid[i, j]    = v["recall"]
            all_results[f"d{depth}_w{width}"] = v

    with open(SUMMARY_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved → {SUMMARY_PATH}")

    # --- Heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        (f1_grid,        "Test F1",        "Blues"),
        (precision_grid, "Test Precision", "Greens"),
        (recall_grid,    "Test Recall",    "Oranges"),
    ]

    for ax, (data, title, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(WIDTHS)))
        ax.set_xticklabels([str(w) for w in WIDTHS])
        ax.set_yticks(range(len(DEPTHS)))
        ax.set_yticklabels([str(d) for d in DEPTHS])
        ax.set_xlabel("Hidden Width", fontsize=11)
        ax.set_ylabel("Depth (hidden layers)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for ii in range(len(DEPTHS)):
            for jj in range(len(WIDTHS)):
                val = data[ii, jj]
                txt = f"{val:.3f}" if not np.isnan(val) else "N/A"
                ax.text(jj, ii, txt, ha="center", va="center",
                        color="black", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, format=mticker.FormatStrFormatter("%.2f"))

    plt.suptitle("Architecture Search — Depth × Width  (LR=1e-4, pos_weight=2.0)", fontsize=13)
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=150)
    print(f"Heatmap saved → {HEATMAP_PATH}")

    # Print ranked table
    print("\n" + "=" * 52)
    print(f"{'Depth':>6} {'Width':>6} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    print("-" * 52)
    for v in sorted(all_results.values(), key=lambda x: -x["f1"]):
        print(f"{v['depth']:>6} {v['width']:>6} {v['f1']:>8.4f} {v['precision']:>8.4f} {v['recall']:>8.4f}")
    print("=" * 52)


# ============================================================
# SEQUENTIAL LOCAL MODE  (no args)
# ============================================================
def run_local():
    print("Local sequential mode (all 9 configs).")
    if not os.path.exists(CACHE_PATH):
        print("Cache not found — running preprocess first.")
        run_preprocess()

    cache = np.load(CACHE_PATH)
    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    train_loader = DataLoader(CachedDataset(cache["train_X"], cache["train_Y"], augment_rot=True),
                              shuffle=True, persistent_workers=True, **loader_kw)
    val_loader   = DataLoader(CachedDataset(cache["val_X"],   cache["val_Y"]),   shuffle=False, **loader_kw)
    test_loader  = DataLoader(CachedDataset(cache["test_X"],  cache["test_Y"]),  shuffle=False, **loader_kw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for task_id, (depth, width) in enumerate(CONFIGS):
        print(f"\n[{task_id+1}/9] depth={depth}, width={width}")
        f1, p, r = train_config(depth, width, train_loader, val_loader, test_loader, device)
        print(f"  Test F1={f1:.4f}  P={p:.4f}  R={r:.4f}")
        with open(os.path.join(RESULTS_DIR, f"d{depth}_w{width}.json"), "w") as fj:
            json.dump({"depth": depth, "width": width, "f1": f1, "precision": p, "recall": r}, fj)

    run_merge()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preprocess", action="store_true",
                       help="Extract features from all STLs and save cache.")
    group.add_argument("--task-id", type=int, metavar="N",
                       help="Train config N (0-8) from cache (for SLURM array).")
    group.add_argument("--merge", action="store_true",
                       help="Merge result JSONs and plot heatmap.")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping (run all EPOCHS).")
    args = parser.parse_args()

    if args.preprocess:
        run_preprocess(num_cores=16)
    elif args.task_id is not None:
        if not (0 <= args.task_id < len(CONFIGS)):
            raise ValueError(f"--task-id must be 0–{len(CONFIGS)-1}")
        run_task(args.task_id, early_stop=not args.no_early_stop)
    elif args.merge:
        run_merge()
    else:
        run_local()
