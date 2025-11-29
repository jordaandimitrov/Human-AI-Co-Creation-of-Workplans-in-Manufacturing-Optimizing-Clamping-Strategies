import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import copy  # Used to save the best weights in memory

# --- Vedo (Used only for mathematical feature extraction, not plotting) ---
from vedo import Mesh


# ============================================================
# STEP 1: STL Feature Extraction (Unchanged)
# ============================================================
def extract_triangle_features_stl(stl_file_path):
    # 1. Load Mesh
    mesh = Mesh(stl_file_path)
    mesh.triangulate()

    points = mesh.points
    tris = np.array(mesh.cells).astype(np.int64)

    # 2. Get Basic Vectors
    normals = mesh.cell_normals
    centroids = mesh.cell_centers().points
    areas = mesh.area()

    # 3. Calculate Relative Features
    part_center = np.array(mesh.center_of_mass(), dtype=np.float32)
    dist_to_center = np.linalg.norm(centroids - part_center, axis=1)

    rel_pos = centroids - part_center
    radial_dist = np.linalg.norm(rel_pos[:, 0:2], axis=1)

    z_values = centroids[:, 2]
    min_z = np.min(z_values)
    max_z = np.max(z_values)
    height_range = max_z - min_z if (max_z - min_z) > 1e-6 else 1.0
    rel_height = (z_values - min_z) / height_range

    v0 = points[tris[:, 0]]
    v1 = points[tris[:, 1]]
    v2 = points[tris[:, 2]]
    edge_a = np.linalg.norm(v1 - v0, axis=1)
    edge_b = np.linalg.norm(v2 - v1, axis=1)
    edge_c = np.linalg.norm(v0 - v2, axis=1)

    # 4. Normalize Scalar Features
    max_area = np.max(areas) if np.max(areas) > 1e-6 else 1.0
    areas_norm = areas / max_area

    max_dist = np.max(dist_to_center) if np.max(dist_to_center) > 1e-6 else 1.0
    dist_norm = dist_to_center / max_dist

    max_rad = np.max(radial_dist) if np.max(radial_dist) > 1e-6 else 1.0
    rad_norm = radial_dist / max_rad

    max_edge = np.max([np.max(edge_a), np.max(edge_b), np.max(edge_c)])
    max_edge = max_edge if max_edge > 1e-6 else 1.0

    edge_a_norm = edge_a / max_edge
    edge_b_norm = edge_b / max_edge
    edge_c_norm = edge_c / max_edge

    # 5. Construct Feature Matrix
    N_tris = len(tris)
    feats = np.zeros((N_tris, 11), dtype=np.float32)

    feats[:, 0] = areas_norm
    feats[:, 1] = 1.0  # Bias
    feats[:, 2:5] = normals
    feats[:, 5] = dist_norm
    feats[:, 6] = rad_norm
    feats[:, 7] = rel_height
    feats[:, 8] = edge_a_norm
    feats[:, 9] = edge_b_norm
    feats[:, 10] = edge_c_norm

    return feats, tris, points


# ============================================================
# STEP 2: Dataset (Unchanged)
# ============================================================
class TriangleDataset(Dataset):
    def __init__(self, parts, augment_rot=True):
        self.parts = parts
        self.augment_rot = augment_rot

    def __len__(self):
        return sum(p["tri_features"].shape[0] for p in self.parts)

    def __getitem__(self, idx):
        # Locate the part and triangle index
        for p in self.parts:
            n_tri = p["tri_features"].shape[0]
            if idx < n_tri:
                X = p["tri_features"].copy()
                labels = p["labels"]
                break
            idx -= n_tri
        else:
            raise IndexError("Triangle index out of range")

        n_tri = X.shape[0]
        tri_labels = np.zeros((n_tri, 2), dtype=np.float32)
        clamp_indices = [int(i) for i, label in labels.items() if label == "clamp"]
        if clamp_indices:
            tri_labels[clamp_indices, 0] = 1.0

        # Rotation Augmentation
        if self.augment_rot:
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0],
                          [s, c, 0],
                          [0, 0, 1]], dtype=np.float32)
            # Rotate only normals (cols 2,3,4)
            X[:, 2:5] = X[:, 2:5] @ R.T

        return torch.tensor(X, dtype=torch.float32), torch.tensor(tri_labels, dtype=torch.float32)


# ============================================================
# STEP 3: Model Architecture (Unchanged)
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
# STEP 4: Training Loop (UPDATED WITH EARLY STOPPER)
# ============================================================
def train_model_triangles(model, train_loader, val_loader, epochs, pos_weight_val, patience_limit, lr=1e-3,
                          lambda_normal=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    weights = torch.tensor([pos_weight_val, pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

    # --- Early Stopping Variables ---
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_weights = None

    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        train_batches = 0

        for X, y in train_loader:
            if X.dim() == 3 and X.shape[0] == 1:
                X = X.squeeze(0)
                y = y.squeeze(0)
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            probs = torch.sigmoid(logits)

            # Normal constraint
            normals = X[:, 2:5]
            normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-9)
            areas = X[:, 0]
            weights_prob = probs[:, 0]
            total_normal = torch.sum(areas.unsqueeze(1) * normals * weights_prob.unsqueeze(1), dim=0)
            loss_normal = torch.norm(total_normal, p=2)

            loss_bce = criterion(logits, y)
            loss = loss_bce + lambda_normal * loss_normal

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(1, train_batches)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for X, y in val_loader:
                if X.dim() == 3 and X.shape[0] == 1:
                    X = X.squeeze(0)
                    y = y.squeeze(0)
                X, y = X.to(device), y.to(device)

                logits = model(X)
                # Note: No backprop here
                loss_bce = criterion(logits, y)
                val_loss += loss_bce.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)

        # --- DDP LOGGING & EARLY STOPPING ---
        # We need to ensure only Rank 0 makes the decision,
        # but all processes must stop together.

        stop_signal = torch.zeros(1).to(device)  # 0=Continue, 1=Stop

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                # Save best weights in memory
                if isinstance(model, DDP):
                    best_model_weights = copy.deepcopy(model.module.state_dict())
                else:
                    best_model_weights = copy.deepcopy(model.state_dict())
                print(f"   ⭐ Validation Improved. Patience reset.")
            else:
                early_stop_counter += 1
                print(f"   ⚠️ No improvement ({early_stop_counter}/{patience_limit})")

            # Check if we should stop
            if early_stop_counter >= patience_limit:
                print("🛑 Early stopping triggered!")
                stop_signal[0] = 1.0

        # Sync the decision across all GPUs
        if dist.is_initialized():
            dist.broadcast(stop_signal, src=0)

        if stop_signal.item() == 1.0:
            break

    # Restore best weights before returning
    if best_model_weights is not None:
        if dist.is_initialized() and dist.get_rank() == 0:
            print("↺ Restoring weights from best epoch...")

        # We load the best weights back into the model so the saved file is the best one,
        # not the last (overfitted) one.
        if isinstance(model, DDP):
            model.module.load_state_dict(best_model_weights)
        else:
            model.load_state_dict(best_model_weights)

    return model


# ============================================================
# STEP 5: Data Loading Helper (Unchanged)
# ============================================================
def load_labeled_dataset(labels_file):
    with open(labels_file) as f:
        all_labels_by_path = json.load(f)
    all_parts = []

    for stl_file_path, labels_dict in all_labels_by_path.items():
        stl_file_path = os.path.normpath(stl_file_path).replace('\\', '/')
        if not os.path.exists(stl_file_path):
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"⚠️ {stl_file_path} not found, skipping")
            continue

        try:
            tri_feats, tri_verts, points = extract_triangle_features_stl(stl_file_path)
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"✅ Processed {stl_file_path} with {len(tri_feats)} triangles.")

            all_parts.append({
                "tri_features": tri_feats,
                "tri_verts": tri_verts,
                "points": points,
                "labels": labels_dict,
                "file_path": stl_file_path
            })
        except Exception as e:
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"⚠️ Error processing {stl_file_path}: {e}")

    return all_parts


# ============================================================
# STEP 6: Main Training Script (UPDATED with Patience Input)
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # --- DDP INIT ---
    if 'RANK' in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Fallback
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    # Paths
    model_path = "clamp_support_model.pth"
    labels_path = r"training_set/all_part_labels.json"

    # =======================================================
    # NEW: Ask for User Input (Epochs, Weight, Patience)
    # =======================================================
    # Tensor size increased to 3 to hold [epochs, weight, patience]
    config_tensor = torch.zeros(3).to(device)

    # Only Rank 0 asks
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("\n=== Training Configuration ===")
        try:
            u_epochs = int(input("Enter max Epochs (e.g., 50): ").strip())
            u_weight = float(input("Enter Positive Class Weight (e.g., 20.0): ").strip())
            u_patience = int(input("Enter Early Stopping Patience (e.g., 3): ").strip())
        except ValueError:
            print("Invalid input. Using defaults: Epochs=50, Weight=20.0, Patience=3")
            u_epochs = 50
            u_weight = 20.0
            u_patience = 3

        config_tensor[0] = u_epochs
        config_tensor[1] = u_weight
        config_tensor[2] = u_patience

    # Broadcast to all GPUs
    if dist.is_initialized():
        dist.broadcast(config_tensor, src=0)

    user_epochs = int(config_tensor[0].item())
    user_weight = float(config_tensor[1].item())
    user_patience = int(config_tensor[2].item())

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"⚙️ Running: Max Epochs={user_epochs}, Weight={user_weight}, Patience={user_patience}")
    # =======================================================

    # Model Setup
    model = ClampSupportNet(in_dim=11, hidden_dim=64, out_dim=2).to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("🧠 Loading labeled dataset...")

    all_parts = load_labeled_dataset(labels_path)

    if not all_parts:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("No labeled parts found!")
        if dist.is_initialized(): dist.destroy_process_group()
        exit()

    # Split dataset
    random.seed(42)
    random.shuffle(all_parts)

    split_idx = int(0.8 * len(all_parts))
    parts_train = all_parts[:split_idx]
    parts_val = all_parts[split_idx:]

    # --- Samplers & Loaders ---
    if dist.is_initialized():
        train_sampler = DistributedSampler(parts_train, shuffle=True)
        val_sampler = DistributedSampler(parts_val, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    train_loader = DataLoader(train_dataset := TriangleDataset(parts_train, augment_rot=True),
                              batch_size=1, sampler=train_sampler, shuffle=shuffle_train, num_workers=4)
    val_loader = DataLoader(val_dataset := TriangleDataset(parts_val, augment_rot=False),
                            batch_size=1, sampler=val_sampler, shuffle=False, num_workers=4)

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("🚀 Starting Training with Early Stopping...")

    # Run Training
    trained_model = train_model_triangles(
        model, train_loader, val_loader,
        epochs=user_epochs,
        pos_weight_val=user_weight,
        patience_limit=user_patience,  # <--- Pass patience here
        lr=1e-3, device=device
    )

    # Save only on rank 0
    if not dist.is_initialized() or dist.get_rank() == 0:
        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save(state_dict, model_path)
        print(f"✅ Best model (from early stopping) saved to {model_path}")

    if dist.is_initialized():
        dist.destroy_process_group()