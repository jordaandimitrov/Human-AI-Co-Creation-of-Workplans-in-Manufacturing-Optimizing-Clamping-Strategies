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

# CONFIG
WINNING_WEIGHT = 2.0
WINNING_LR = 1e-4
EPOCHS = 100
BATCH_SIZE = 32768
PATIENCE = 15


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
            y[clamp_indices, 0] = 1.0  # Class 0 = Clamp

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
        all_feats = []
        all_labels = []
        for p in parts:
            all_feats.append(p["tri_features"])
            all_labels.append(p["labels"])

        if len(all_feats) > 0:
            self.X = np.vstack(all_feats)
            self.Y = np.vstack(all_labels)
        else:
            self.X = np.zeros((0, 13), dtype=np.float32)
            self.Y = np.zeros((0, 2), dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.Y[idx]
        if self.augment_rot:
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            nx = x[2] * c - x[3] * s
            ny = x[2] * s + x[3] * c
            x[2], x[3] = nx, ny
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ==========================================
# 3. MODEL (13 Dims)
# ==========================================
class ClampSupportNet(nn.Module):
    def __init__(self, in_dim=13, hidden_dim=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, X):
        return self.net(X)


def evaluate(model, loader, device):
    model.eval()
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            probs = torch.sigmoid(model(X))
            preds = (probs > 0.5).float()
            y_c = y[:, 0]
            p_c = preds[:, 0]
            tp += ((p_c == 1) & (y_c == 1)).sum().item()
            fp += ((p_c == 1) & (y_c == 0)).sum().item()
            fn += ((p_c == 0) & (y_c == 1)).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-9)
    return f1, precision, recall


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    labels_path = "training_set/all_part_labels.json"
    model_save_path = "separation_model_13dim.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cores = 16
    print(f"Running on {device} with {num_cores} CPU cores for loading.")

    if not os.path.exists(labels_path):
        print("Labels file not found.")
        exit()

    with open(labels_path) as f:
        all_labels = json.load(f)

    # 1. PARALLEL LOAD
    tasks = [(path, lbl) for path, lbl in all_labels.items()]
    print(f"Starting parallel feature extraction on {len(tasks)} files...")
    with Pool(processes=num_cores) as pool:
        results = pool.map(load_single_part_wrapper, tasks)

    parts = [r for r in results if r is not None]
    print(f"Successfully loaded {len(parts)} parts.")

    # 2. DATA SPLIT (70 / 15 / 15)
    random.seed(42)
    random.shuffle(parts)
    n = len(parts)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    train_ds = FlatTriangleDataset(parts[:train_end],        augment_rot=True)
    val_ds   = FlatTriangleDataset(parts[train_end:val_end], augment_rot=False)
    test_ds  = FlatTriangleDataset(parts[val_end:],          augment_rot=False)

    print(f"Split -> train: {train_end}, val: {val_end - train_end}, test: {n - val_end} parts")

    # 3. LOADERS
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # 4. TRAIN
    model = ClampSupportNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=WINNING_LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([WINNING_WEIGHT, WINNING_WEIGHT]).to(device))

    best_f1 = 0.0
    patience_counter = 0

    print("Starting Training Loop...")

    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        f1, precision, recall = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1} | Val F1: {f1:.4f} (P: {precision:.2f}, R: {recall:.2f})")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early Stopping")
                break

    print(f"Model saved to {model_save_path}")

    # 5. FINAL TEST EVALUATION (run once on held-out set)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    f1, precision, recall = evaluate(model, test_loader, device)
    print(f"\n{'='*40}")
    print(f"TEST RESULTS (held-out, never seen during training)")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"{'='*40}")
