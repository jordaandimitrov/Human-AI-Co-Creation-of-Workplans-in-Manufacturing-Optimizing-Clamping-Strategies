# ============================================================
# CLAMP FACE CLASSIFIER (Prismatic 3-axis clamp face detector)
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
from vedo import Mesh, show, color_map


# ============================================================
# STEP 1: Feature extraction
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
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRep import BRep_Tool_Surface
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopExp import topexp
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.gp import gp_Pnt

def face_center(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    p = props.CentreOfMass()
    return np.array([p.X(), p.Y(), p.Z()])

def get_edges(face):
    """Return a list of TopoDS_Edge from a face."""
    edges = []
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        edges.append(exp.Current())
        exp.Next()
    return edges

def edge_length(edge):
    curve_adapt = BRepAdaptor_Curve(edge)
    return curve_adapt.LastParameter() - curve_adapt.FirstParameter()

from sklearn.decomposition import PCA

def extract_face_features(shape, apply_local_frame=True):
    """Extended geometric features for each face, optionally in local part frame."""
    BRepMesh_IncrementalMesh(shape, 0.05, True, True)

    features = []
    centers = []

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    all_faces = []
    while exp.More():
        all_faces.append(topods.Face(exp.Current()))
        exp.Next()

    # Compute all vertices for PCA
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

    # PCA for local coordinate frame
    if apply_local_frame:
        pca = PCA(n_components=3)
        pca.fit(all_vertices)
        center_pca = pca.mean_
        rot_matrix = pca.components_.T  # 3x3 rotation matrix
    else:
        center_pca = np.zeros(3)
        rot_matrix = np.eye(3)

    # Precompute part center
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    part_center = np.array([props.CentreOfMass().X(),
                            props.CentreOfMass().Y(),
                            props.CentreOfMass().Z()])

    for face in all_faces:
        # --- Area ---
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()

        # --- Center ---
        c = np.array([props.CentreOfMass().X(),
                      props.CentreOfMass().Y(),
                      props.CentreOfMass().Z()])

        # --- Surface type ---
        surf = BRepAdaptor_Surface(face, True)
        surf_type_id = surf.GetType()

        # --- Normal ---
        nx, ny, nz = safe_face_normal(face)
        normal = np.array([nx, ny, nz])
        normal /= np.linalg.norm(normal) + 1e-8

        # --- Orientation angles ---
        z_angle = math.degrees(math.acos(abs(normal[2])))  # angle to Z
        x_angle = math.degrees(math.acos(abs(normal[0])))  # angle to X

        # --- Perimeter & aspect ratio ---
        edges = get_edges(face)
        lengths = [edge_length(e) for e in edges if edge_length(e) > 1e-6]
        perimeter = sum(lengths)
        aspect_ratio = max(lengths) / (min(lengths) + 1e-8) if lengths else 1.0

        # --- Distance from part center ---
        dist_center = np.linalg.norm(c - part_center)

        # --- Flatness ---
        flatness = 1 if surf_type_id == 0 else 0

        # --- Transform to part local frame ---
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
    # normalize certain columns
    features[:, 0:3] /= np.max(features[:, 0:3], axis=0, keepdims=True) + 1e-8
    features[:, 8:11] /= np.max(np.abs(features[:, 8:11]), axis=0, keepdims=True) + 1e-8
    features[:, 11] /= np.max(features[:, 11]) + 1e-8

    return features



# ============================================================
# STEP 2: Dataset
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

class ClampDataset(Dataset):
    """Clamp dataset with on-the-fly random rotation augmentation for robustness."""
    def __init__(self, parts, augment_rot=True):
        self.parts = parts
        self.augment_rot = augment_rot

    def __len__(self):
        return len(self.parts)

    def __getitem__(self, idx):
        part = self.parts[idx]
        X = part["features"].copy()
        y = torch.tensor(part["labels"], dtype=torch.float32)

        # --- Random rotation augmentation ---
        if self.augment_rot:
            angle = np.random.uniform(0, 2*np.pi)
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                   [np.sin(angle),  np.cos(angle), 0],
                                   [0, 0, 1]])
            # Apply rotation to normals (cols 3-5) and face centers (cols 8-10)
            X[:, 3:6] = X[:, 3:6] @ rot_matrix.T
            X[:, 8:11] = X[:, 8:11] @ rot_matrix.T

        X = torch.tensor(X, dtype=torch.float32)
        return X, y

def load_labeled_dataset(labels_file):
    """Load labeled parts (features + face labels) from JSON"""
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

        shape = read_step_file(stp_file)
        features = extract_face_features(shape)

        if len(features) != len(labels):
            print(f"⚠️  Warning: feature/label mismatch for {stp_file} ({len(features)} vs {len(labels)})")
            continue

        all_parts.append({"features": features, "labels": labels})

    return all_parts


# ============================================================
# STEP 3: Model
# ============================================================

class ClampNet(nn.Module):
    def __init__(self, in_dim=13, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, X):
        return self.net(X).squeeze(-1)


# ============================================================
# STEP 4: Training
# ============================================================

def train_model(model, dataloader, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0
        for X, y in dataloader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")

    return model


# ============================================================
# STEP 5: Inference
# ============================================================

def predict_best_faces(model, features, top_k=2):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(features, dtype=torch.float32)
        logits = model(X)
        probs = torch.sigmoid(logits)
        top_faces = torch.topk(probs, top_k).indices.numpy()
    return top_faces, probs.numpy()


# ============================================================
# STEP 6: Visualization
# ============================================================

from vedo import Mesh, show, Text3D
import numpy as np



from vedo import Mesh, show, Plotter, color_map

from vedo import Mesh, Plotter, color_map

from vedo import Mesh, Plotter, color_map, Text2D

from vedo import Mesh, Plotter, color_map, Text2D

def visualize_clamp_faces(shape, predicted_faces, probs=None):
    """Show colored faces with picking enabled and display selected face number + probability."""
    BRepMesh_IncrementalMesh(shape, 0.05, True, True)
    exp = TopExp_Explorer(shape, TopAbs_FACE)

    meshes = []
    face_idx = 0  # index for faces

    while exp.More():
        face = topods.Face(exp.Current())
        loc = face.Location()
        triang = BRep_Tool.Triangulation(face, loc)

        if triang:
            nodes = np.array([[triang.Node(i).X(),
                               triang.Node(i).Y(),
                               triang.Node(i).Z()]
                              for i in range(1, triang.NbNodes() + 1)])
            tris = np.array([[triang.Triangle(i).Get()[0] - 1,
                              triang.Triangle(i).Get()[1] - 1,
                              triang.Triangle(i).Get()[2] - 1]
                             for i in range(1, triang.NbTriangles() + 1)])

            # Color mapping
            if probs is not None and face_idx < len(probs):
                color = color_map(probs[face_idx], name="jet", vmin=0, vmax=1)
            else:
                color = "red" if face_idx in predicted_faces else "lightgray"

            mesh = Mesh([nodes, tris], c=color, alpha=1.0)
            mesh.face_idx = face_idx  # store the face index for picking
            meshes.append(mesh)
            face_idx += 1

        exp.Next()

    # Create plotter
    plt = Plotter(title="Predicted Clamping Faces", bg="gray", axes=1)

    # Add meshes
    for m in meshes:
        plt.add(m)

    # Add overlay Text2D in top-right
    overlay_text = Text2D("", pos="top-right", c="white", s=1.5)
    plt.add(overlay_text)

    # Picking callback
    def on_pick(evt):
        if evt.actor and hasattr(evt.actor, "face_idx"):
            idx = evt.actor.face_idx
            prob = probs[idx] if probs is not None and idx < len(probs) else None
            text_str = f"Selected Face: {idx}"
            if prob is not None:
                text_str += f"\nProbability: {prob:.3f}"
            overlay_text.text(text_str)
            print(f"Face clicked: {idx}, Probability: {prob}")
            plt.render()  # update screen

    plt.add_callback("mouse click", on_pick)
    plt.show(interactive=True)







# ============================================================
# STEP 7: Main Execution
# ============================================================

if __name__ == "__main__":
    Tk().withdraw()
    model_path = "clamp_model.pth"
    labels_path = r"C:\Users\jorda\Desktop\Unif\MA3\Thesis\Git\labels.json"

    model = ClampNet(in_dim=13)

    # -------- Train or Load choice --------
    choice = input("Train new model (t) or load existing (l)? ").strip().lower()

    if choice == "t" or not os.path.exists(model_path):
        print("🧠 Loading labeled dataset...")
        all_parts = load_labeled_dataset(labels_path)
        if not all_parts:
            raise ValueError("No labeled parts found in labels.json!")

        dataset = ClampDataset(all_parts)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        if os.path.exists(model_path):
            print("Continuing training from previous model...")
            model.load_state_dict(torch.load(model_path))

        print("🚀 Training model...")
        trained_model = train_model(model, dataloader, epochs=50, lr=1e-3)

        # Save with timestamp backup
        torch.save(trained_model.state_dict(), model_path)
        torch.save(trained_model.state_dict(),
                   f"clamp_model_{datetime.now():%Y%m%d_%H%M}.pth")
        print(f"✅ Model saved to {model_path}")

    else:
        print("⚡ Loading pretrained model...")
        model.load_state_dict(torch.load(model_path))
        trained_model = model

    while True:
        # ---------------------------------------------
        # STEP 4: Testing / Inference on a new STEP file
        # ---------------------------------------------
        test_file = askopenfilename(
            title="Select a STEP file to test (Cancel to quit)",
            filetypes=[("STP files", "*.stp")]
        )
        if not test_file:
            print("No file selected. Exiting testing loop.")
            break

        shape = read_step_file(test_file)
        features = extract_face_features(shape)
        predicted_faces, probs = predict_best_faces(trained_model, features)

        # Print probabilities for each face
        print("\n=== Face Probabilities ===")
        for i, p in enumerate(probs):
            print(f"Face {i}: {p:.4f}")

        visualize_clamp_faces(shape, predicted_faces, probs)

        # ---------------------------------------------
        # Ask user whether to continue testing
        # ---------------------------------------------
        user_in = input("\nPress [t] to test another file, or any other key to quit: ").strip().lower()
        if user_in != "t":
            print("Exiting testing loop.")
            break

