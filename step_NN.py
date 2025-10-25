# ============================================================
# MULTI-OUTPUT CLAMP & SUPPORT FACE CLASSIFIER
# (Prismatic 3-axis clamp/support face detector)
# ============================================================

import json
import os
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ---------- OpenCascade imports ----------
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool

# ---------- Vedo ----------
from vedo import Mesh, Plotter, color_map, Text2D

# ============================================================
# STEP 1: Feature extraction (unchanged, returns (n_faces, 13))
# ============================================================

def read_step_file(filename):
    reader = STEPControl_Reader()
    status = reader.ReadFile(filename)
    if status != IFSelect_RetDone:
        raise ValueError(f"Error reading STEP file: {filename}")
    reader.TransferRoots()
    return reader.OneShape()


def safe_face_normal(face):
    """Compute approximate normal for planar or cylindrical faces."""
    try:
        surf = BRepAdaptor_Surface(face, True)
        surf_type = surf.GetType()
        if surf_type == 0:  # Plane
            dir = surf.Plane().Axis().Direction()
        elif surf_type == 1:  # Cylinder
            dir = surf.Cylinder().Axis().Direction()
        else:
            return (0.0, 0.0, 1.0)
        return dir.X(), dir.Y(), dir.Z()
    except Exception:
        return (0.0, 0.0, 1.0)


import math
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopExp import topexp
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from sklearn.decomposition import PCA

def get_edges(face):
    edges = []
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        edges.append(exp.Current())
        exp.Next()
    return edges

def edge_length(edge):
    curve_adapt = BRepAdaptor_Curve(edge)
    return curve_adapt.LastParameter() - curve_adapt.FirstParameter()

def extract_face_features(shape, apply_local_frame=True):
    """Extended geometric features for each face (13 features as before)."""
    # ensure a mesh exists for triangulation
    BRepMesh_IncrementalMesh(shape, 0.05, True, True)

    features = []
    centers = []

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    all_faces = []
    while exp.More():
        all_faces.append(topods.Face(exp.Current()))
        exp.Next()

    # collect vertices for PCA
    all_vertices = []
    for face in all_faces:
        loc = face.Location()
        triang = BRep_Tool.Triangulation(face, loc)
        if triang:
            pts = np.array([[triang.Node(i).X(),
                             triang.Node(i).Y(),
                             triang.Node(i).Z()]
                            for i in range(1, triang.NbNodes() + 1)])
            all_vertices.append(pts)
    if not all_vertices:
        raise ValueError("No vertices found in shape for PCA normalization.")
    all_vertices = np.vstack(all_vertices)

    if apply_local_frame:
        pca = PCA(n_components=3)
        pca.fit(all_vertices)
        center_pca = pca.mean_
        rot_matrix = pca.components_.T
    else:
        center_pca = np.zeros(3)
        rot_matrix = np.eye(3)

    # precompute part center
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    part_center = np.array([props.CentreOfMass().X(),
                            props.CentreOfMass().Y(),
                            props.CentreOfMass().Z()])

    for face in all_faces:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        c = np.array([props.CentreOfMass().X(),
                      props.CentreOfMass().Y(),
                      props.CentreOfMass().Z()])
        surf = BRepAdaptor_Surface(face, True)
        surf_type_id = surf.GetType()
        nx, ny, nz = safe_face_normal(face)
        normal = np.array([nx, ny, nz])
        normal /= np.linalg.norm(normal) + 1e-8
        z_angle = math.degrees(math.acos(abs(normal[2])))
        x_angle = math.degrees(math.acos(abs(normal[0])))
        edges = get_edges(face)
        lengths = [edge_length(e) for e in edges if edge_length(e) > 1e-6]
        perimeter = sum(lengths)
        aspect_ratio = max(lengths) / (min(lengths) + 1e-8) if lengths else 1.0
        dist_center = np.linalg.norm(c - part_center)
        flatness = 1 if surf_type_id == 0 else 0
        if apply_local_frame:
            c_local = (c - center_pca) @ rot_matrix
            normal_local = normal @ rot_matrix
        else:
            c_local = c
            normal_local = normal

        features.append([
            area, perimeter, aspect_ratio,
            *normal_local,
            z_angle, x_angle,
            *c_local,
            dist_center, flatness
        ])
        centers.append(c)

    features = np.array(features, dtype=np.float32)
    # normalize certain columns carefully (avoid divide by zero)
    if features.shape[0] > 0:
        features[:, 0:3] /= np.max(features[:, 0:3], axis=0, keepdims=True) + 1e-8
        features[:, 8:11] /= np.max(np.abs(features[:, 8:11]), axis=0, keepdims=True) + 1e-8
        # column 11 (dist_center?) exists - ensure non-zero denom
        if np.max(features[:, 11]) > 0:
            features[:, 11] /= np.max(features[:, 11]) + 1e-8

    return features

# ============================================================
# STEP 2: Dataset (now supports two-label targets per face)
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

class ClampSupportDataset(Dataset):
    """Dataset that returns X: (n_faces, n_features) and y: (n_faces, 2)
       where column 0 = clamp label, column 1 = support label."""
    def __init__(self, parts, augment_rot=True):
        # parts: list of dicts {"features": np.array(n_faces, n_feat),
        #                        "labels": {"clamp_labels": [...], "support_labels": [...]}}
        self.parts = parts
        self.augment_rot = augment_rot

    def __len__(self):
        return len(self.parts)

    def __getitem__(self, idx):
        part = self.parts[idx]
        X = part["features"].copy()  # (n_faces, n_feat)
        # labels may be stored as dict (new) or old style list (legacy)
        raw_labels = part["labels"]
        if isinstance(raw_labels, dict):
            clamp = np.array(raw_labels.get("clamp_labels", [0]*len(X)), dtype=np.float32)
            support = np.array(raw_labels.get("support_labels", [0]*len(X)), dtype=np.float32)
        else:
            # legacy single list -> interpret as clamp labels, support zeros
            clamp = np.array(raw_labels, dtype=np.float32)
            support = np.zeros_like(clamp)

        y = np.stack([clamp, support], axis=1)  # shape (n_faces, 2)

        # augmentation: rotate normals and centers around Z
        if self.augment_rot:
            angle = np.random.uniform(0, 2*np.pi)
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                   [np.sin(angle),  np.cos(angle), 0],
                                   [0, 0, 1]])
            # normal_local at cols 3:6, centers at 8:11
            if X.shape[1] >= 11:
                X[:, 3:6] = X[:, 3:6] @ rot_matrix.T
                X[:, 8:11] = X[:, 8:11] @ rot_matrix.T

        X = torch.tensor(X, dtype=torch.float32)  # (n_faces, n_feat)
        y = torch.tensor(y, dtype=torch.float32)  # (n_faces, 2)
        return X, y

def load_labeled_dataset(labels_file):
    """Load labeled parts from JSON. Accepts new dict format or legacy list format."""
    try:
        with open(labels_file) as f:
            all_labels = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{labels_file} not found. Please label some files first.")

    all_parts = []
    for stp_file, labels in all_labels.items():
        if not os.path.exists(stp_file):
            print(f"⚠️  Warning: {stp_file} not found, skipping.")
            continue
        try:
            shape = read_step_file(stp_file)
            features = extract_face_features(shape)
        except Exception as e:
            print(f"⚠️  Error reading/extracting {stp_file}: {e}")
            continue

        # determine number of faces and check label lengths
        nfaces = features.shape[0]
        if isinstance(labels, dict):
            cl = labels.get("clamp_labels", [])
            sp = labels.get("support_labels", [])
            if len(cl) != nfaces or len(sp) != nfaces:
                print(f"⚠️  feature/label mismatch for {stp_file} ({nfaces} faces vs labels {len(cl)}/{len(sp)})")
                continue
            label_obj = {"clamp_labels": cl, "support_labels": sp}
        else:
            # legacy: single list
            if len(labels) != nfaces:
                print(f"⚠️  feature/label mismatch for {stp_file} ({nfaces} faces vs labels {len(labels)})")
                continue
            label_obj = labels  # will be interpreted as clamp labels, support zeros in dataset

        all_parts.append({"features": features, "labels": label_obj})

    return all_parts

# ============================================================
# STEP 3: Model (multi-output)
# ============================================================

class ClampSupportNet(nn.Module):
    def __init__(self, in_dim=13, hidden_dim=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)   # now outputs 2 logits per face
        )

    def forward(self, X):
        # X: (n_faces, in_dim) -> returns (n_faces, out_dim)
        return self.net(X)

# ============================================================
# STEP 4: Training (handles per-face multi-label targets)
# ============================================================

def train_model(model, dataloader, epochs=50, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # works with shape (n,2) targets

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for X, y in dataloader:
            # X: batch_size x n_faces x n_feat  (we used batch_size=1)
            # y: batch_size x n_faces x 2
            # We'll handle variable number of faces by squeezing batch dim (assume batch_size==1)
            if X.dim() == 3 and X.shape[0] == 1:
                X = X.squeeze(0)
                y = y.squeeze(0)
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)            # shape (n_faces, 2)
            loss = criterion(logits, y)  # BCEWithLogitsLoss averages over elements
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model

# ============================================================
# STEP 5: Inference helpers
# ============================================================

def predict_best_faces(model, features, top_k=2):
    """Predict clamp & support probabilities for all faces."""
    model.eval()
    with torch.no_grad():
        X = torch.tensor(features, dtype=torch.float32)
        logits = model(X)  # shape: [n_faces, 2]
        probs = torch.sigmoid(logits).numpy()  # convert to probabilities

    clamp_probs = probs[:, 0]
    support_probs = probs[:, 1]

    clamp_top = np.argsort(-clamp_probs)[:top_k]
    support_top = np.argsort(-support_probs)[:top_k]

    return {"clamp_top": clamp_top, "support_top": support_top}, probs


# ============================================================
# STEP 6: Visualization (choose which class to visualize)
# ============================================================

from vedo import Mesh, Plotter, color_map, Text2D
import numpy as np

def visualize_faces(shape, clamp_probs, support_probs):
    """
    Visualize clamp & support probabilities with:
    - Heatmap toggle (C/U)
    - Click-to-show selected face index and probability
    """

    # --- Mesh faces ---
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods

    BRepMesh_IncrementalMesh(shape, 0.05, True, True)
    exp = TopExp_Explorer(shape, TopAbs_FACE)

    meshes = []
    face_idx = 0

    while exp.More():
        face = topods.Face(exp.Current())
        loc = face.Location()
        triang = BRep_Tool.Triangulation(face, loc)
        if triang:
            pts = np.array([[triang.Node(i).X(),
                             triang.Node(i).Y(),
                             triang.Node(i).Z()]
                            for i in range(1, triang.NbNodes() + 1)])
            tris = np.array([[triang.Triangle(i).Get()[0] - 1,
                              triang.Triangle(i).Get()[1] - 1,
                              triang.Triangle(i).Get()[2] - 1]
                             for i in range(1, triang.NbTriangles() + 1)])
            mesh = Mesh([pts, tris])
            mesh.face_idx = face_idx
            meshes.append(mesh)
            face_idx += 1
        exp.Next()

    # --- Create Plotter ---
    plt = Plotter(title="Clamp/Support Visualization", bg="gray", axes=1)

    # Current view mode: 0 = clamp, 1 = support
    view_mode = {"current": 0}

    def update_colors():
        """Update mesh colors depending on active mode"""
        current_probs = clamp_probs if view_mode["current"] == 0 else support_probs
        for i, m in enumerate(meshes):
            if i < len(current_probs):
                color = color_map(current_probs[i], name="jet", vmin=0, vmax=1)
            else:
                color = "lightgray"
            m.c(color)
        plt.render()

    # Initial color mode: clamp
    update_colors()
    for m in meshes:
        plt.add(m)

    # --- Overlay text in top right ---
    overlay = Text2D("Mode: Clamp", pos="top-right", c="white", s=1.5)
    plt.add(overlay)

    # --- Clicked face info text ---
    face_info = Text2D("", pos="bottom-right", c="white", s=1.2)
    plt.add(face_info)

    # --- Handle face clicks ---
    def on_pick(evt):
        if evt.actor and hasattr(evt.actor, "face_idx"):
            idx = evt.actor.face_idx
            if view_mode["current"] == 0:
                prob = clamp_probs[idx]
                label = "Clamp"
            else:
                prob = support_probs[idx]
                label = "Support"
            face_info.text(f"Selected Face: {idx}\n{label} Probability: {prob:.3f}")
            plt.render()

    plt.add_callback("mouse click", on_pick)

    # --- Keyboard toggle ---
    def on_key(evt):
        key = evt.keypress.lower()
        if key == "c":
            view_mode["current"] = 0
            overlay.text("Mode: Clamp")
            update_colors()
        elif key == "u":
            view_mode["current"] = 1
            overlay.text("Mode: Support")
            update_colors()
        plt.render()

    plt.add_callback("key press", on_key)

    from vedo import Button

    # --- Toggle Button ---
    # Add this flag outside the toggle_mode function
    # Mutable state
    view_mode = {"current": 0}
    last_toggle_time = {"ts": 0}

    # Callback
    def toggle_mode(widget, event=None):
        import time
        now = time.time()
        if now - last_toggle_time["ts"] < 0.3:  # debounce 300ms
            return
        last_toggle_time["ts"] = now

        # Toggle mode
        view_mode["current"] = 1 - view_mode["current"]
        mode_text = "Clamp" if view_mode["current"] == 0 else "Support"

        # Update overlay
        overlay.text(f"Mode: {mode_text}")

        # Update button label
        btn.text(mode_text)

        # Update face colors
        update_colors()
        plt.render()

    # Add button and capture the returned Button object
    btn = plt.add_button(
        toggle_mode,
        pos=(0.05, 0.95),
        states=["Clamp", "Support"],  # required but we override manually
        c=["lightblue", "lightgreen"],
        bc=["black", "black"],
        font="Courier",
        size=24,
        bold=True,
    )

    plt.show(interactive=True)


# ============================================================
# STEP 7: Main Execution (train/load and test with toggle)
# ============================================================
# ============================================================
# STEP 7: Main Execution (train/load and test with toggle)
# ============================================================

if __name__ == "__main__":
    Tk().withdraw()
    model_path = "clamp_support_model.pth"
    labels_path = r"C:\Users\jorda\Desktop\Unif\MA3\Thesis\Git\labels.json"

    model = ClampSupportNet(in_dim=13, hidden_dim=64, out_dim=2)

    choice = input("Train new model (t) or load existing (l)? ").strip().lower()

    if choice == "t" or not os.path.exists(model_path):
        print("🧠 Loading labeled dataset...")
        all_parts = load_labeled_dataset(labels_path)
        if not all_parts:
            raise ValueError("No labeled parts found in labels.json!")

        dataset = ClampSupportDataset(all_parts, augment_rot=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Optionally resume
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path))
                print("Loaded existing weights to continue training.")
            except Exception as e:
                print(f"⚠️ Could not load previous weights: {e}")

        print("🚀 Training model...")
        trained_model = train_model(model, dataloader, epochs=50, lr=1e-3)

        # Save
        torch.save(trained_model.state_dict(), model_path)
        torch.save(trained_model.state_dict(),
                   f"clamp_support_model_{datetime.now():%Y%m%d_%H%M}.pth")
        print(f"✅ Model saved to {model_path}")

    else:
        print("⚡ Loading pretrained model...")
        model.load_state_dict(torch.load(model_path))
        trained_model = model

    # --- Testing loop ---
    while True:
        test_file = askopenfilename(
            title="Select a STEP file to test (Cancel to quit)",
            filetypes=[("STP files", "*.stp")]
        )
        if not test_file:
            print("No file selected. Exiting testing loop.")
            break

        shape = read_step_file(test_file)
        features = extract_face_features(shape)
        tops, probs = predict_best_faces(trained_model, features, top_k=2)

        print("\n=== Face Probabilities (Clamp | Support) ===")
        for i, p in enumerate(probs):
            print(f"Face {i}: clamp={p[0]:.4f} | support={p[1]:.4f}")

        visualize_faces(shape, clamp_probs=probs[:, 0], support_probs=probs[:, 1])

        cont = input("\nPress [t] to test another file, or any other key to quit: ").strip().lower()
        if cont != "t":
            print("Exiting testing loop.")
            break
