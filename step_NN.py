# ============================================================
# TRIANGLE-LEVEL CLAMP & SUPPORT CLASSIFIER (STL/Vedo Based)
# ============================================================

import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# --- Vedo ---
from vedo import Mesh, Plotter, color_map, Text2D


# --- OCC/STEP dependencies REMOVED ---

# Global variable to store the JSON data structure for clarity
# The new format is: {full_path: {tri_index: label_string, ...}}
# Labels are "clamp" or "unselected" (now "none" in the script)

# ============================================================
# STEP 1: STL file & triangle feature extraction (Vedo)
# ============================================================

def extract_triangle_features_stl(stl_file_path):
    """
    Extracts triangle features directly from a loaded STL mesh using Vedo/NumPy.
    """

    # 1. Load Mesh
    mesh = Mesh(stl_file_path)
    mesh.triangulate()

    points = mesh.points

    # Mesh Cell/Triangle extraction
    tris = np.array(mesh.cells).astype(np.int64)  # Triangles (faces)

    # 2. Compute basic features
    normals = mesh.cell_normals
    centroids = mesh.cell_centers().points
    areas = mesh.area()

    # --- FIX APPLIED HERE: Use the direct mesh.center_of_mass() method ---
    # This method returns a list/array of coordinates, bypassing the dictionary access error.
    part_center = np.array(mesh.center_of_mass(), dtype=np.float32)
    # ---------------------------------------------------------------------

    dist_center = np.linalg.norm(centroids - part_center, axis=1)  # N x 1

    # Initialize feature matrix
    N_tris = len(tris)
    feats = np.zeros((N_tris, 23), dtype=np.float32)

    # 3. Populate features (23-dimension vector)

    # --- Core Geometrical Features ---
    feats[:, 0] = areas  # Area (1)
    feats[:, 1] = 1.0  # Perimeter (Placeholder)
    feats[:, 2] = 1.0  # Aspect Ratio (Placeholder)

    # Normal Vector
    feats[:, 3:6] = normals

    # Centroid
    feats[:, 6:9] = centroids

    # Distance to Part Center
    feats[:, 9] = dist_center

    # --- Vertex Coordinates ---
    feats[:, 14:17] = points[tris[:, 0]]  # Vertex A
    feats[:, 17:20] = points[tris[:, 1]]  # Vertex B
    feats[:, 20:23] = points[tris[:, 2]]  # Vertex C

    # --- Neighbor/Topology Info (Placeholders) ---
    feats[:, 10:14] = 0.0

    return feats, tris, points

# ============================================================
# STEP 2: Dataset
# ============================================================

class TriangleDataset(Dataset):
    def __init__(self, parts, augment_rot=True):
        self.parts = parts
        self.augment_rot = augment_rot

    def __len__(self):
        return sum(p["tri_features"].shape[0] for p in self.parts)

    def __getitem__(self, idx):
        # Locate the part and triangle index corresponding to idx
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
        # Initialize labels: [clamp, support] (support is always zero now)
        tri_labels = np.zeros((n_tri, 2), dtype=np.float32)

        # New JSON format uses {str(index): label_str}
        clamp_indices = [int(idx) for idx, label in labels.items() if label == "clamp"]

        if clamp_indices:
            tri_labels[clamp_indices, 0] = 1.0  # Set clamp label

        if self.augment_rot and X.shape[1] >= 9:
            # Augment rotation around Z-axis (cols 3:6=normal, cols 6:9=centroid)
            angle = np.random.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
            X[:, 3:6] = X[:, 3:6] @ R.T
            X[:, 6:9] = X[:, 6:9] @ R.T

        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            tri_labels, dtype=torch.float32
        )


# ============================================================
# STEP 3: Model (Unchanged)
# ============================================================

class ClampSupportNet(nn.Module):
    def __init__(self, in_dim=23, hidden_dim=64, out_dim=2):
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
# STEP 4: Training (Unchanged)
# ============================================================

def train_model_triangles(model, dataloader, epochs=3, lr=1e-3, lambda_normal=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for X, y in dataloader:
            if X.dim() == 3 and X.shape[0] == 1:
                X = X.squeeze(0)
                y = y.squeeze(0)
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(X)
            probs = torch.sigmoid(logits)

            # --- Normal constraint calculation ---
            normals = X[:, 3:6]
            normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-9)
            areas = X[:, 0]

            weights = probs[:, 0]
            total_normal = torch.sum(areas.unsqueeze(1) * normals * weights.unsqueeze(1), dim=0)
            loss_normal = torch.norm(total_normal, p=2)

            # --- Combine with BCE loss ---
            loss_bce = criterion(logits, y)
            loss = loss_bce + lambda_normal * loss_normal

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    return model


# ============================================================
# STEP 5: Prediction (Unchanged)
# ============================================================

def predict_best_triangles(model, features, top_k=5, zero_support=False):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(features, dtype=torch.float32)
        logits = model(X)
        probs = torch.sigmoid(logits).numpy()
    if zero_support: probs[:, 1] = 0.0
    clamp_probs = probs[:, 0]
    support_probs = probs[:, 1]
    top_clamp = np.argsort(-clamp_probs)[:top_k]
    top_support = np.argsort(-support_probs)[:top_k]
    return {"clamp_top": top_clamp, "support_top": top_support}, probs


# ============================================================
# STEP 6: Visualization (Unchanged)
# ============================================================

# ============================================================
# STEP 6: Visualization (FIXED: Robust Picking)
# ============================================================

# ============================================================
# STEP 6: Visualization (FIXED: Reliable Picking)
# ============================================================

# ============================================================
# STEP 6: Visualization (FIXED: Robust Individual Triangle Actors)
# ============================================================

def visualize_triangles(points, tris, tri_probs):
    """
    Visualizes the mesh by creating a separate actor for every triangle,
    allowing for reliable index-based picking via actor properties.
    """

    # 1. Create a list of individual Mesh actors (one per triangle)
    meshes = []

    # tri_probs[:, 0] is Clamp probability
    # tri_probs[:, 1] is Support probability (often 0.0)

    for i, t_indices in enumerate(tris):
        # Create a new mesh actor for this single triangle
        # Points: only the 3 vertices of this triangle
        # Cells: [[0, 1, 2]] - indices of the 3 points
        mesh = Mesh([points[t_indices], [[0, 1, 2]]])

        # CRITICAL: Store the original triangle index on the mesh actor
        mesh.tri_idx = i

        meshes.append(mesh)

    # 2. Setup Plotter and Controls
    plt = Plotter(title="Clamp/Support Triangles", bg="gray", axes=1)

    # State tracking: 0=Clamp, 1=Support
    view_mode = {"current": 0}

    overlay = Text2D("Mode: Clamp", s=1.5)
    face_info = Text2D("", pos="top-right", c="white", s=1.2)

    plt.add(overlay, face_info)  # Add text overlays

    # 3. Define Callback Functions

    def update_colors():
        """Updates the color of all triangles based on the current view_mode."""
        for i, m in enumerate(meshes):
            # Get the probability for the current mode (Clamp or Support)
            prob = tri_probs[i, view_mode["current"]]

            # Color the individual mesh actor
            m.c(color_map(prob, name="jet", vmin=0, vmax=1))

        plt.render()

    def on_pick(event):
        """Executed when a mesh actor is clicked."""
        # Check if the event picked an actor AND if that actor has the stored index
        if event.actor and hasattr(event.actor, "tri_idx"):
            idx = event.actor.tri_idx
            prob = tri_probs[idx, view_mode["current"]]
            label = "Clamp" if view_mode["current"] == 0 else "Support"

            # Update the text overlay
            face_info.text(f"Triangle {idx}\n{label}: {prob:.4f}")
            plt.render()
        else:
            # Clear text if click missed a triangle
            face_info.text("")
            plt.render()

    def on_key(event):
        """Executed on key press (C or U)."""
        key = event.keypress.lower()
        if key == "c":
            view_mode["current"] = 0
            overlay.text("Mode: Clamp")
            update_colors()
        elif key == "u":
            view_mode["current"] = 1
            overlay.text("Mode: Support")
            update_colors()

    # 4. Final Setup and Display

    # Add all individual mesh actors to the plotter
    for m in meshes:
        plt.add(m)

    # Register callbacks
    plt.add_callback("mouse click", on_pick)
    plt.add_callback("key press", on_key)

    # Initial color update
    update_colors()

    plt.show(interactive=True)# ============================================================

def load_labeled_dataset(labels_file):
    with open(labels_file) as f:
        all_labels_by_path = json.load(f)
    all_parts = []

    for stl_file_path, labels_dict in all_labels_by_path.items():
        # Standardize path for consistency
        stl_file_path = os.path.normpath(stl_file_path).replace('\\', '/')

        if not os.path.exists(stl_file_path):
            print(f"⚠️ {stl_file_path} not found, skipping")
            continue

        try:
            tri_feats, tri_verts, points = extract_triangle_features_stl(stl_file_path)

            print(f"✅ Processed {stl_file_path} with {len(tri_feats)} triangles.")

            # Map features to the expected dictionary format
            all_parts.append({
                "tri_features": tri_feats,
                "tri_verts": tri_verts,
                "points": points,
                "labels": labels_dict,
                "file_path": stl_file_path
            })
        except Exception as e:
            print(f"⚠️ Error processing {stl_file_path}: {e}")
            import traceback
            traceback.print_exc()

    return all_parts


# ============================================================
# STEP 8: Main (ADAPTED FOR STL)
# ============================================================

if __name__ == "__main__":
    #Tk().withdraw()
    model_path = "clamp_support_model.pth"
    labels_path = r"training_set/all_part_labels.json"

    model = ClampSupportNet(in_dim=23, hidden_dim=64, out_dim=2)

    choice = input("Train new model (t) or load existing (l)? ").strip().lower()

    if choice == "t" or not os.path.exists(model_path):
        print("🧠 Loading labeled dataset...")
        all_parts = load_labeled_dataset(labels_path)
        if not all_parts: raise ValueError("No labeled parts found!")
        dataset = TriangleDataset(all_parts, augment_rot=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path));
                print("Loaded existing weights.")
            except:
                pass
        print("🚀 Training model...")
        trained_model = train_model_triangles(model, dataloader, epochs=2, lr=1e-3)
        torch.save(trained_model.state_dict(), model_path)
        torch.save(trained_model.state_dict(), f"clamp_support_model_{datetime.now():%Y%m%d_%H%M}.pth")
        print(f"✅ Model saved to {model_path}")
    else:
        print("⚡ Loading pretrained model...")
        model.load_state_dict(torch.load(model_path, 'cpu'))
        trained_model = model

    while True:
        test_file_path = askopenfilename(title="Select STL file", filetypes=[("STL files", "*.stl")])
        if not test_file_path: break

        tri_feats, tri_verts, points = extract_triangle_features_stl(test_file_path)

        tops, probs = predict_best_triangles(trained_model, tri_feats, top_k=5, zero_support=True)
        print("\n=== Triangle Probabilities (Clamp | Support) ===")
        for i, p in enumerate(probs):
            print(f"Tri {i}: clamp={p[0]:.4f} | support={p[1]:.4f}")
        visualize_triangles(points, tri_verts, probs)
        cont = input("\nPress [t] to test another file, or any other key to quit: ").strip().lower()
        if cont != "t": break