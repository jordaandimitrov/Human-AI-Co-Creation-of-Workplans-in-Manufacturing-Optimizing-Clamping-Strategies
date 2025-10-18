import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from vedo import show, Mesh
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
import numpy as np

# -------- Load STEP file ----------
def read_step_file(filename):
    reader = STEPControl_Reader()
    status = reader.ReadFile(filename)
    if status != IFSelect_RetDone:
        raise ValueError("Error reading STEP file")
    reader.TransferRoots()
    return reader.OneShape()

# -------- Mesh faces ----------
def mesh_faces(shape):
    """Return list of Mesh objects and their indices"""
    BRepMesh_IncrementalMesh(shape, 0.05, True, True)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    meshes = []
    face_indices = []
    idx = 0
    while exp.More():
        face = topods.Face(exp.Current())
        loc = face.Location()
        triang = BRep_Tool.Triangulation(face, loc)
        if triang:
            nodes = np.array([[triang.Node(i).X(),
                               triang.Node(i).Y(),
                               triang.Node(i).Z()]
                              for i in range(1, triang.NbNodes()+1)])
            tris = np.array([[triang.Triangle(i).Get()[0]-1,
                              triang.Triangle(i).Get()[1]-1,
                              triang.Triangle(i).Get()[2]-1]
                             for i in range(1, triang.NbTriangles()+1)])
            mesh = Mesh([nodes, tris])
            meshes.append(mesh)
            face_indices.append(idx)
        exp.Next()
        idx += 1
    return meshes, face_indices

# -------- Visualization ----------
def visualize_with_labels(shape, labels_for_file):
    meshes, face_indices = mesh_faces(shape)
    for mesh, idx in zip(meshes, face_indices):
        if idx < len(labels_for_file) and labels_for_file[idx] == 1:
            mesh.c("red")  # clamping
        else:
            mesh.c("lightgray")  # non-clamping
        mesh.alpha(1.0)
    show(*meshes, "Clamping Face Check", axes=1, viewup="z", resetcam=True)

# -------- Main ----------
if __name__ == "__main__":
    # File explorer to pick STEP
    Tk().withdraw()
    step_file = askopenfilename(title="Select an STP file", filetypes=[("STP files", "*.stp")])
    if not step_file:
        raise ValueError("No STEP file selected!")

    # Load labels JSON
    labels_file = "labels.json"
    try:
        with open(labels_file) as f:
            all_labels = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Labels file {labels_file} not found!")

    # Get labels for this STEP file
    labels_for_file = all_labels.get(step_file)
    if labels_for_file is None:
        print(f"No labels found for {step_file} in {labels_file}. Showing gray faces.")
        labels_for_file = []

    # Read STEP
    shape = read_step_file(step_file)

    # Visualize
    visualize_with_labels(shape, labels_for_file)
