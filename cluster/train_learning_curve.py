"""
Learning curve: how does test F1 scale with number of training parts?

Modes (CLI):
  --preprocess          Extract features, save lc_cache.npz with per-part boundaries
  --task-id N           Load cache, train config N (0-20), save result JSON
  --merge               Load all result JSONs, plot learning curve with error bars
  (no args)             Run all 21 configs sequentially — for local testing only

SLURM array: 0-20  (7 part counts x 3 seeds)
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
from multiprocessing import Pool
import random

import features

# ============================================================
# FIXED HYPERPARAMETERS  (same winning config as train.py)
# ============================================================
WINNING_WEIGHT = 2.0
WINNING_LR     = 1e-4
EPOCHS         = 100
BATCH_SIZE     = 32768
IN_DIM         = 13
HIDDEN_DIM     = 128
DEPTH          = 4

PART_COUNTS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200]
N_SEEDS     = 3
CONFIGS     = [(n, s) for n in PART_COUNTS for s in range(N_SEEDS)]  # 45 tasks

LABELS_PATH  = "training_set/all_part_labels.json"
CACHE_PATH   = "lc_cache.npz"
RESULTS_DIR  = "lc_results"
PLOT_PATH    = "learning_curve.png"
SUMMARY_PATH = "learning_curve_results.json"


# ============================================================
# DATA LOADING
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

    train_parts = parts[:train_end]
    val_parts   = parts[train_end:val_end]
    test_parts  = parts[val_end:]

    def stack(split):
        X = np.vstack([p["tri_features"] for p in split])
        Y = np.vstack([p["labels"]       for p in split])
        return X.astype(np.float32), Y.astype(np.float32)

    val_X,  val_Y  = stack(val_parts)
    test_X, test_Y = stack(test_parts)

    train_X_all       = np.vstack([p["tri_features"] for p in train_parts]).astype(np.float32)
    train_Y_all       = np.vstack([p["labels"]       for p in train_parts]).astype(np.float32)
    train_part_sizes  = np.array([len(p["tri_features"]) for p in train_parts], dtype=np.int64)

    np.savez_compressed(
        CACHE_PATH,
        train_X=train_X_all,
        train_Y=train_Y_all,
        train_part_sizes=train_part_sizes,
        val_X=val_X,   val_Y=val_Y,
        test_X=test_X, test_Y=test_Y,
    )
    print(f"Cache saved → {CACHE_PATH}")
    print(f"  train parts:     {len(train_parts)}")
    print(f"  train triangles: {len(train_X_all):,}")
    print(f"  val triangles:   {len(val_X):,}")
    print(f"  test triangles:  {len(test_X):,}")


# ============================================================
# DATASET
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
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(IN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2)]
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
    n_parts, seed = CONFIGS[task_id]
    print(f"Task {task_id}: n_parts={n_parts}, seed={seed}", flush=True)

    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"Cache not found: {CACHE_PATH}. Run --preprocess first.")

    cache      = np.load(CACHE_PATH)
    part_sizes = cache["train_part_sizes"]
    n_avail    = len(part_sizes)

    actual_n = min(n_parts, n_avail)
    if actual_n < n_parts:
        print(f"WARNING: requested {n_parts} parts but only {n_avail} available. Using {actual_n}.", flush=True)

    rng     = np.random.default_rng(seed * 1000 + n_parts)
    chosen  = rng.choice(n_avail, size=actual_n, replace=False)
    ends    = np.cumsum(part_sizes)
    starts  = np.concatenate([[0], ends[:-1]])
    mask    = np.zeros(len(cache["train_X"]), dtype=bool)
    for i in chosen:
        mask[starts[i]:ends[i]] = True

    train_X = cache["train_X"][mask]
    train_Y = cache["train_Y"][mask]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | train triangles: {len(train_X):,}", flush=True)

    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    train_loader = DataLoader(CachedDataset(train_X, train_Y, augment_rot=True),
                              shuffle=True, persistent_workers=True, **loader_kw)
    val_loader   = DataLoader(CachedDataset(cache["val_X"], cache["val_Y"]),   shuffle=False, **loader_kw)
    test_loader  = DataLoader(CachedDataset(cache["test_X"], cache["test_Y"]), shuffle=False, **loader_kw)

    model     = ClampSupportNet().to(device)
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
        "n_parts": actual_n, "seed": seed,
        "f1": test_f1, "precision": test_p, "recall": test_r,
        "n_triangles": int(mask.sum()),
    }
    out_path = os.path.join(RESULTS_DIR, f"n{n_parts}_s{seed}.json")
    with open(out_path, "w") as fout:
        json.dump(out, fout, indent=2)
    print(f"Saved → {out_path}", flush=True)


# ============================================================
# MERGE + PLOT  (--merge)
# ============================================================
def run_merge():
    all_results = {}
    for n_parts, seed in CONFIGS:
        path = os.path.join(RESULTS_DIR, f"n{n_parts}_s{seed}.json")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        with open(path) as f:
            all_results[f"n{n_parts}_s{seed}"] = json.load(f)

    with open(SUMMARY_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved → {SUMMARY_PATH}")

    agg = {}
    for n in PART_COUNTS:
        f1s = [all_results[f"n{n}_s{s}"]["f1"]
               for s in range(N_SEEDS)
               if f"n{n}_s{s}" in all_results]
        if f1s:
            agg[n] = (np.mean(f1s), np.std(f1s))

    ns    = sorted(agg.keys())
    means = [agg[n][0] for n in ns]
    stds  = [agg[n][1] for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ns, means, yerr=stds, marker="o", linewidth=2,
                capsize=5, color="steelblue", ecolor="gray", label="Mean ± std (3 seeds)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Training Parts", fontsize=12)
    ax.set_ylabel("Test F1", fontsize=12)
    ax.set_title("Learning Curve — Test F1 vs Training Set Size", fontsize=13, fontweight="bold")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Plot saved → {PLOT_PATH}")

    print(f"\n{'Parts':>8} {'Mean F1':>10} {'Std F1':>10}")
    print("-" * 32)
    for n in ns:
        m, s = agg[n]
        print(f"{n:>8} {m:>10.4f} {s:>10.4f}")


# ============================================================
# SEQUENTIAL LOCAL MODE  (no args)
# ============================================================
def run_local():
    print("Local sequential mode (all 21 configs).")
    if not os.path.exists(CACHE_PATH):
        run_preprocess()
    for task_id in range(len(CONFIGS)):
        run_task(task_id)
    run_merge()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--preprocess", action="store_true",
                       help="Extract features and save lc_cache.npz with part boundaries.")
    group.add_argument("--task-id", type=int, metavar="N",
                       help="Train config N (0-20) from cache (for SLURM array).")
    group.add_argument("--merge", action="store_true",
                       help="Merge result JSONs and plot learning curve.")
    args = parser.parse_args()

    if args.preprocess:
        run_preprocess(num_cores=16)
    elif args.task_id is not None:
        if not (0 <= args.task_id < len(CONFIGS)):
            raise ValueError(f"--task-id must be 0–{len(CONFIGS)-1}")
        run_task(args.task_id)
    elif args.merge:
        run_merge()
    else:
        run_local()
