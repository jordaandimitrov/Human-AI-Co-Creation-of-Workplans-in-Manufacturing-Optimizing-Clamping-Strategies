import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count
import random
import argparse

# IMPORT THE SHARED FEATURES MODULE
import features

# ==========================================
# 0. ARGPARSE CONFIG
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train clamp classifier")

    parser.add_argument("--winning_lr", type=float, default=1e-2,
                        help="Learning rate for optimizer")
    parser.add_argument("--winning_weight", type=float, default=5.0,
                        help="Positive class weight for BCEWithLogitsLoss")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=100,
                        help="Early stopping patience")

    return parser.parse_args()


# ==========================================
# 1. PARALLEL DATA LOADING
# ==========================================
def load_single_part_wrapper(args):
    """Helper to unwrap arguments for multiprocessing"""
    path, label_dict = args
    if not os.path.exists(path):
        return None
    try:
        feats, _, _ = features.extract_triangle_features(path)

        # Parse labels
        y = np.zeros((feats.shape[0], 2), dtype=np.float32) #mss naar 1 veranderen
        clamp_indices = [int(i) for i, lbl in label_dict.items() if lbl == "clamp"]
        if clamp_indices:
            y[clamp_indices, 0] = 1.0  # Class 0 = Clamp

        return {"tri_features": feats, "labels": y}
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(path)}: {e}")
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
            self.X = np.zeros((0, 15), dtype=np.float32)
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
# 3. MODEL (15 Dims)
# ==========================================
class ClampSupportNet(nn.Module):
    def __init__(self, in_dim=15, hidden_dim=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, X):
        return self.net(X)


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    args = parse_args()

    WINNING_LR = args.winning_lr
    WINNING_WEIGHT = args.winning_weight
    EPOCHS = args.epochs
    PATIENCE = args.patience

    labels_path = "training_set/all_part_labels.json"
    model_save_path = "separation_model_15dim.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cores = 8
    print(f"🚀 Running on {device} with {num_cores} CPU cores for loading.")
    print(f"⚙️ Params → LR={WINNING_LR}, Weight={WINNING_WEIGHT}, Epochs={EPOCHS}, Patience={PATIENCE}")

    # Load JSON
    if not os.path.exists(labels_path):
        print("❌ Labels file not found.")
        exit()

    with open(labels_path) as f:
        all_labels = json.load(f)

    # 1. PARALLEL LOAD
    tasks = [(path, lbl) for path, lbl in all_labels.items()]

    print(f"Starting parallel feature extraction on {len(tasks)} files...")
    with Pool(processes=num_cores) as pool:
        results = pool.map(load_single_part_wrapper, tasks)

    parts = [r for r in results if r is not None]
    print(f"✅ Successfully loaded {len(parts)} parts.")

    # 2. DATA SPLIT
    random.seed(42)
    random.shuffle(parts)
    split = int(0.8 * len(parts))

    train_ds = FlatTriangleDataset(parts[:split], augment_rot=False) #set to false to test memorization
    val_ds = FlatTriangleDataset(parts[split:], augment_rot=False)

    # 3. LOADERS
    BATCH_SIZE = 8192
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 4. TRAIN
    model = ClampSupportNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=WINNING_LR)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([WINNING_WEIGHT, WINNING_WEIGHT]).to(device)
    )

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

        # Validation
        model.eval()
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                probs = torch.sigmoid(model(X))
                preds = (probs > 0.3).float() #0.5

                y_c = y[:, 0]
                p_c = preds[:, 0]
                tp += ((p_c == 1) & (y_c == 1)).sum().item()
                fp += ((p_c == 1) & (y_c == 0)).sum().item()
                fn += ((p_c == 0) & (y_c == 1)).sum().item()

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        print(f"Epoch {epoch + 1} | Val F1: {f1:.4f} (P: {precision:.2f}, R: {recall:.2f})")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("🛑 Early Stopping")
                break

    print(f"✅ Model saved to {model_save_path}")
