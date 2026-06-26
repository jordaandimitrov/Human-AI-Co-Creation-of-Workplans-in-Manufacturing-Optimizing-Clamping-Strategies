import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
import random

import features

WINNING_WEIGHT = 4.0
WINNING_LR     = 1e-4
EPOCHS         = 200
BATCH_SIZE     = 32768


# ==========================================
# 1. PARALLEL DATA LOADING
# ==========================================
def load_single_part_wrapper(args):
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
        print(f"Error processing {os.path.basename(path)}: {e}")
        return None


# ==========================================
# 2. DATASET
# ==========================================
class FlatTriangleDataset(Dataset):
    def __init__(self, parts, augment_rot=True):
        self.augment_rot = augment_rot
        all_feats, all_labels = [], []
        for p in parts:
            all_feats.append(p["tri_features"])
            all_labels.append(p["labels"])

        if all_feats:
            self.X = np.vstack(all_feats)
            self.Y = np.vstack(all_labels)
        else:
            self.X = np.zeros((0, 15), dtype=np.float32)
            self.Y = np.zeros((0, 2),  dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.Y[idx]
        if self.augment_rot:
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            x[2], x[3] = x[2] * c - x[3] * s, x[2] * s + x[3] * c
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ==========================================
# 3. MODEL (15 dims)
# ==========================================
class ClampSupportNet(nn.Module):
    def __init__(self, in_dim=15, hidden_dim=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, X):
        return self.net(X)


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


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    labels_path     = "training_set_cylinders/all_part_labels.json"
    model_save_path = "cylinder_model_15dim.pth"

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cores = 16
    print(f"Running on {device} with {num_cores} cores.")

    if not os.path.exists(labels_path):
        print(f"Labels file not found: {labels_path}")
        exit()

    with open(labels_path) as f:
        all_labels = json.load(f)

    tasks = [(path, lbl) for path, lbl in all_labels.items()
             if not path.startswith("_")]

    print(f"Loading {len(tasks)} parts in parallel...")
    with Pool(processes=num_cores) as pool:
        results = pool.map(load_single_part_wrapper, tasks)

    parts = [r for r in results if r is not None]
    print(f"Loaded {len(parts)} parts successfully.")

    random.seed(42)
    random.shuffle(parts)
    n         = len(parts)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    train_ds = FlatTriangleDataset(parts[:train_end],        augment_rot=True)
    val_ds   = FlatTriangleDataset(parts[train_end:val_end], augment_rot=False)
    test_ds  = FlatTriangleDataset(parts[val_end:],          augment_rot=False)

    print(f"Split → train: {train_end}, val: {val_end - train_end}, test: {n - val_end} parts")

    loader_kw    = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  persistent_workers=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    model     = ClampSupportNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=WINNING_LR)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([WINNING_WEIGHT, WINNING_WEIGHT]).to(device))

    best_f1    = 0.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("Training...")
    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

        f1, p, r = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1:3d} | Val F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), model_save_path)

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    f1, p, r = evaluate(model, test_loader, device)
    print(f"\n{'='*40}")
    print(f"TEST RESULTS")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"{'='*40}")
    print(f"Model saved to {model_save_path}")
