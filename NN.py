import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import copy
import argparse

# --- Vedo (Headless Mode) ---
from vedo import Mesh, settings

# FIX: Remove the invalid setting.
# We only need parallel projection off/on if we render, but for math it doesn't matter.
settings.use_parallel_projection = True

# ============================================================
# 1. CLUSTER CONFIGURATION (Separation Strategy)
# ============================================================
# We use the aggressive settings to force the separation of probabilities
WINNING_WEIGHT = 5.0
WINNING_LR = 1e-4  # 0.0001
EPOCHS = 60
BATCH_SIZE = 8192
PATIENCE = 15


# ============================================================
# 2. ORIGINAL FEATURE EXTRACTION (11 Features)
# ============================================================
def extract_triangle_features_stl(stl_file_path):
    print(f"   Processing: {os.path.basename(stl_file_path)}...")

    # Load mesh - this does NOT open a window
    mesh = Mesh(stl_file_path)
    mesh.triangulate()

    points = mesh.points
    tris = np.array(mesh.cells).astype(np.int64)
    normals = mesh.cell_normals
    centroids = mesh.cell_centers().points
    areas = mesh.area()
    part_center = np.array(mesh.center_of_mass(), dtype=np.float32)
    dist_to_center = np.linalg.norm(centroids - part_center, axis=1)

    # Relative Position & Radial Dist
    rel_pos = centroids - part_center
    radial_dist = np.linalg.norm(rel_pos[:, 0:2], axis=1)

    # Z-height relative
    z_values = centroids[:, 2]
    min_z, max_z = np.min(z_values), np.max(z_values)
    height_range = max_z - min_z if (max_z - min_z) > 1e-6 else 1.0
    rel_height = (z_values - min_z) / height_range

    # Edge lengths
    v0 = points[tris[:, 0]];
    v1 = points[tris[:, 1]];
    v2 = points[tris[:, 2]]
    edge_a = np.linalg.norm(v1 - v0, axis=1)
    edge_b = np.linalg.norm(v2 - v1, axis=1)
    edge_c = np.linalg.norm(v0 - v2, axis=1)

    # Normalization
    max_area = np.max(areas) if np.max(areas) > 1e-6 else 1.0
    max_dist = np.max(dist_to_center) if np.max(dist_to_center) > 1e-6 else 1.0
    max_rad = np.max(radial_dist) if np.max(radial_dist) > 1e-6 else 1.0
    max_edge = np.max([np.max(edge_a), np.max(edge_b), np.max(edge_c)])
    max_edge = max_edge if max_edge > 1e-6 else 1.0

    # Build Original 11-Dim Feature Matrix
    feats = np.zeros((len(tris), 11), dtype=np.float32)
    feats[:, 0] = areas / max_area
    feats[:, 1] = 1.0  # Bias
    feats[:, 2:5] = normals
    feats[:, 5] = dist_to_center / max_dist
    feats[:, 6] = radial_dist / max_rad
    feats[:, 7] = rel_height
    feats[:, 8] = edge_a / max_edge
    feats[:, 9] = edge_b / max_edge
    feats[:, 10] = edge_c / max_edge

    return feats, tris, points


# ============================================================
# 3. FAST DATASET
# ============================================================
class FlatTriangleDataset(Dataset):
    def __init__(self, parts, augment_rot=True):
        self.augment_rot = augment_rot
        all_feats = []
        all_labels = []
        for p in parts:
            X = p["tri_features"]
            y = np.zeros((X.shape[0], 2), dtype=np.float32)
            clamp_indices = [int(i) for i, label in p["labels"].items() if label == "clamp"]
            if clamp_indices: y[clamp_indices, 0] = 1.0
            all_feats.append(X);
            all_labels.append(y)

        if len(all_feats) > 0:
            self.X = np.vstack(all_feats)
            self.Y = np.vstack(all_labels)
        else:
            self.X = np.zeros((0, 11), dtype=np.float32)
            self.Y = np.zeros((0, 2), dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.Y[idx]
        if self.augment_rot:
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            n_x = x[2] * c - x[3] * s
            n_y = x[2] * s + x[3] * c
            x[2] = n_x;
            x[3] = n_y
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ============================================================
# 4. MODEL (Original 11 Dim)
# ============================================================
class ClampSupportNet(nn.Module):
    def __init__(self, in_dim=11, hidden_dim=64, out_dim=2):
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
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, X):
        return self.net(X)


# ============================================================
# 5. TRAINING LOOP
# ============================================================
def train_model(model, train_loader, val_loader, epochs, weight_val, patience_limit, device):
    optimizer = optim.Adam(model.parameters(), lr=WINNING_LR)
    weights = torch.tensor([weight_val, weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

    best_val_f1 = 0.0
    early_stop_counter = 0
    best_model_weights = None

    print(f"\n🚀 Training Started on {device}")
    print(f"   Config: Weight={weight_val}, LR={WINNING_LR}, Epochs={epochs}")

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                y_clamp = y[:, 0]
                pred_clamp = preds[:, 0]
                tp += ((pred_clamp == 1) & (y_clamp == 1)).sum().item()
                fp += ((pred_clamp == 1) & (y_clamp == 0)).sum().item()
                fn += ((pred_clamp == 0) & (y_clamp == 1)).sum().item()

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        print(f"Epoch {epoch + 1}/{epochs} | Val F1: {f1:.4f} (Prec: {precision:.2f}, Rec: {recall:.2f})")

        if f1 > best_val_f1:
            best_val_f1 = f1
            early_stop_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"   ⭐ New Best F1!")
        else:
            early_stop_counter += 1
            print(f"   ⚠️ No improvement ({early_stop_counter}/{patience_limit})")

        if early_stop_counter >= patience_limit:
            print("🛑 Early stopping triggered!")
            break

    if best_model_weights:
        model.load_state_dict(best_model_weights)
        print(f"↺ Restored best model (F1: {best_val_f1:.4f})")

    return model


# ============================================================
# 6. MAIN (HEADLESS)
# ============================================================
if __name__ == "__main__":
    model_path = "separation_model_11dim.pth"
    labels_path = "training_set/all_part_labels.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"📂 Loading Data from {labels_path}...")

    if not os.path.exists(labels_path):
        print(f"❌ ERROR: Cannot find {labels_path}")
        print("   Make sure you are running this script from the correct folder.")
        exit(1)

    with open(labels_path) as f:
        all_labels = json.load(f)

    parts = []
    for path, lbl in all_labels.items():
        # Handle potential Windows/Linux path issues
        path = path.replace("\\", "/")
        if os.path.exists(path):
            feats, _, _ = extract_triangle_features_stl(path)
            parts.append({"tri_features": feats, "labels": lbl})
        else:
            print(f"⚠️ Warning: STL file not found: {path}")

    if len(parts) == 0:
        print("❌ No parts loaded. Exiting.")
        exit(1)

    print(f"✅ Loaded {len(parts)} parts.")

    # Shuffle and Split
    random.seed(42)
    random.shuffle(parts)
    split = int(0.8 * len(parts))

    train_ds = FlatTriangleDataset(parts[:split], augment_rot=True)
    val_ds = FlatTriangleDataset(parts[split:], augment_rot=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("\n=== STARTING HEADLESS TRAINING ===")

    # Using ORIGINAL 11 DIMENSIONS
    model = ClampSupportNet(in_dim=11, hidden_dim=64, out_dim=2).to(device)

    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        weight_val=WINNING_WEIGHT,
        patience_limit=PATIENCE,
        device=device
    )

    torch.save(model.state_dict(), model_path)
    print("=" * 60)
    print(f"✅ SUCCESS! Model saved to {model_path}")
    print("   Download this file to your local computer to run the visualization.")
    print("=" * 60)