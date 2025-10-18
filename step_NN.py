# ============================================================
# CLAMP FACE CLASSIFIER (with safe normals + vedo visualization)
# ============================================================
import json
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# OpenCascade imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepTools import breptools
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt

# vedo
from vedo import Points, show


# ============================================================
# STEP 1: Feature extraction
# ============================================================

def read_step_file(filename):
    reader = STEPControl_Reader()
    status = reader.ReadFile(filename)
    if status != IFSelect_RetDone:
        raise ValueError("Error reading STEP file")
    reader.TransferRoots()
    return reader.OneShape()


def safe_face_normal(face):
    """
    Compute approximate normal for planar or cylindrical faces.
    Fallback to (0,0,1) for other surfaces.
    """
    try:
        surf = BRepAdaptor_Surface(face, True)
        surf_type = surf.GetType()
        # Plane
        if surf_type == 0:
            plane = surf.Plane()
            dir = plane.Axis().Direction()
            return dir.X(), dir.Y(), dir.Z()
        # Cylinder
        elif surf_type == 1:
            cyl = surf.Cylinder()
            dir = cyl.Axis().Direction()
            return dir.X(), dir.Y(), dir.Z()
        else:
            return 0.0, 0.0, 1.0
    except Exception:
        return 0.0, 0.0, 1.0


def extract_face_features(shape):
    """Extract features per face: area, surface type, normal vector"""
    features = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)

    while exp.More():
        face = exp.Current()
        area = 0.0
        surf_type_id = 0
        # Area
        try:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            area = props.Mass()
        except Exception:
            area = 0.0
        # Surface type
        try:
            surf = BRepAdaptor_Surface(face, True)
            surf_type_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
            surf_type_id = surf_type_map.get(int(surf.GetType()), 0)
        except Exception:
            surf_type_id = 0
        # Normal
        nx, ny, nz = safe_face_normal(face)
        features.append([area, surf_type_id, nx, ny, nz])
        exp.Next()

    features = np.array(features, dtype=np.float32)
    # Normalize normals
    norms = np.linalg.norm(features[:, 2:5], axis=1, keepdims=True) + 1e-8
    features[:, 2:5] /= norms
    features[:, 0] /= np.max(features[:, 0]) + 1e-8  # normalize area

    return features


# ============================================================
# STEP 2: Dataset
# ============================================================
import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
class ClampDataset(Dataset):
    def __init__(self, parts):
        self.parts = parts

    def __len__(self):
        return len(self.parts)

    def __getitem__(self, idx):
        part = self.parts[idx]
        X = torch.tensor(part["features"], dtype=torch.float32)
        y = torch.tensor(part["labels"], dtype=torch.float32)
        return X, y


# ============================================================
# STEP 3: Neural network
# ============================================================

class ClampNet(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, X):
        logits = self.net(X).squeeze(-1)
        return logits







def load_labeled_dataset(labels_file=r"C:\Users\jorda\Desktop\Unif\MA3\Thesis\Git\labels.json"):
    try:
        with open(labels_file) as f:
            all_labels = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{labels_file} not found. Please label some files first.")

    all_parts = []
    for stp_file, labels in all_labels.items():
        if not os.path.exists(stp_file):
            print(f"Warning: {stp_file} does not exist, skipping.")
            continue
        shape = read_step_file(stp_file)
        features = extract_face_features(shape)
        if len(features) != len(labels):
            print(f"Warning: feature/label mismatch for {stp_file}")
            continue
        all_parts.append({"features": features, "labels": labels})
    return all_parts

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
# STEP 6: Visualization with vedo
# ============================================================

from vedo import show, Points
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
import numpy as np

from vedo import show, Mesh
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
import numpy as np

from vedo import show, Mesh
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
import numpy as np

def visualize_clamp_faces(shape, predicted_faces, probs=None):
    """
    Visualize faces with color-coded probabilities.
    Red = high probability, blue = low probability.
    """
    from vedo import color_map

    # Mesh the shape
    BRepMesh_IncrementalMesh(shape, 0.05, True, True)

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    meshes = []
    idx = 0
    while exp.More():
        face = topods.Face(exp.Current())
        loc = face.Location()
        triang = BRep_Tool.Triangulation(face, loc)
        if triang:
            nodes = np.array([[triang.Node(i).X(),
                               triang.Node(i).Y(),
                               triang.Node(i).Z()]
                              for i in range(1, triang.NbNodes() + 1)])
            tris = np.array([[triang.Triangle(i).Get()[0]-1,
                              triang.Triangle(i).Get()[1]-1,
                              triang.Triangle(i).Get()[2]-1]
                             for i in range(1, triang.NbTriangles() + 1)])

            # --- Color based on probability ---
            if probs is not None:
                color = color_map(probs[idx], name="jet", vmin=0, vmax=1)
            else:
                color = "red" if idx in predicted_faces else "lightgray"

            mesh = Mesh([nodes, tris], c=color, alpha=1.0)
            meshes.append(mesh)

        exp.Next()
        idx += 1

    print("Color legend: Blue = low confidence, Red = high confidence")
    show(*meshes, "Predicted Clamping Faces with Probabilities", axes=1, viewup="z", resetcam=True)




# ============================================================
# STEP 7: Example usage d
# ============================================================

if __name__ == "__main__":
    # ---------------------------------------------
    # STEP 0: Pick STEP files (optional, for labeling)
    # ---------------------------------------------
    Tk().withdraw()

    # ---------------------------------------------
    # STEP 1: Load all labeled STEP files
    # ---------------------------------------------
    all_parts = load_labeled_dataset(r"C:\Users\jorda\Desktop\Unif\MA3\Thesis\Git\labels.json")
    if not all_parts:
        raise ValueError("No labeled STEP files found in labels.json.")

    # ---------------------------------------------
    # STEP 2: Create dataset and dataloader
    # ---------------------------------------------
    dataset = ClampDataset(all_parts)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ---------------------------------------------
    # STEP 3: Initialize and train the model
    # ---------------------------------------------
    model = ClampNet(in_dim=5)
    trained_model = train_model(model, dataloader, epochs=50, lr=1e-3)

    # Optional: save model
    torch.save(trained_model.state_dict(), "clamp_model.pth")

    # ---------------------------------------------
    # STEP 4: Testing / Inference on a new STEP file
    # ---------------------------------------------
    test_file = askopenfilename(title="Select a STEP file to test", filetypes=[("STP files", "*.stp")])
    if not test_file:
        raise ValueError("No STEP file selected!")

    shape = read_step_file(test_file)
    features = extract_face_features(shape)

    predicted_faces, probs = predict_best_faces(trained_model, features)

    # Print probabilities for each face
    print("\n=== Face Probabilities ===")
    for i, p in enumerate(probs):
        print(f"Face {i}: {p:.4f}")

    # Visualize with probabilities
    visualize_clamp_faces(shape, predicted_faces, probs)

