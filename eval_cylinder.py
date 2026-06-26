import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
import random

import features

BATCH_SIZE = 32768


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


class FlatTriangleDataset(Dataset):
    def __init__(self, parts):
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
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.Y[idx], dtype=torch.float32))


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


if __name__ == "__main__":
    labels_path = "training_set_cylinders/all_part_labels.json"
    model_path  = "cylinder_model_15dim.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    with open(labels_path) as f:
        all_labels = json.load(f)

    tasks = [(path, lbl) for path, lbl in all_labels.items()
             if not path.startswith("_")]

    print(f"Loading {len(tasks)} parts...")
    with Pool(processes=8) as pool:
        results = pool.map(load_single_part_wrapper, tasks)

    parts = [r for r in results if r is not None]
    print(f"Loaded {len(parts)} parts.")

    # Reproduce the exact same split as training
    random.seed(42)
    random.shuffle(parts)
    n       = len(parts)
    val_end = int(0.85 * n)

    test_ds     = FlatTriangleDataset(parts[val_end:])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Test set: {n - val_end} parts, {len(test_ds)} triangles")

    model = ClampSupportNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    f1, precision, recall = evaluate(model, test_loader, device)

    print(f"\n{'='*40}")
    print(f"TEST RESULTS  —  cylinder_model_15dim")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"{'='*40}")
