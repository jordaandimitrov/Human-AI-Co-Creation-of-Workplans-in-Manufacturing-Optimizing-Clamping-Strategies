from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json
import numpy as np
from vedo import Mesh, Plotter
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

# ---------------- STEP and meshing (same as brush-labeler) ----------------
def read_step_file(filename):
    reader = STEPControl_Reader()
    status = reader.ReadFile(filename)
    if status != IFSelect_RetDone:
        raise ValueError("Error reading STEP file")
    reader.TransferRoots()
    return reader.OneShape()

def sample_face(face, nx=20, ny=20, uv_range=None):
    adaptor = BRepAdaptor_Surface(face)
    umin, umax, vmin, vmax = adaptor.FirstUParameter(), adaptor.LastUParameter(), adaptor.FirstVParameter(), adaptor.LastVParameter()
    if uv_range:
        (umin, umax), (vmin, vmax) = uv_range
    u = np.linspace(umin, umax, nx)
    v = np.linspace(vmin, vmax, ny)
    points = []
    for vi in v:
        for ui in u:
            p = adaptor.Value(ui, vi)
            points.append([p.X(), p.Y(), p.Z()])
    points = np.array(points)
    tris = []
    for i in range(ny-1):
        for j in range(nx-1):
            idx = i*nx + j
            tris.append([idx, idx+1, idx+nx])
            tris.append([idx+1, idx+nx+1, idx+nx])
    tris = np.array(tris)
    return points, tris

def mesh_faces(shape, nx=20, ny=20):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    meshes = []
    face_indices = []
    idx = 0
    while exp.More():
        face = topods.Face(exp.Current())
        nodes, tris = sample_face(face, nx=nx, ny=ny)
        mesh = Mesh([nodes, tris], c="lightgray", alpha=0.7)
        meshes.append(mesh)
        face_indices.append(idx)
        exp.Next()
        idx += 1
    return meshes, face_indices

# ---------------- Visualization ----------------
def visualize_patches(shape, labels_for_file):
    meshes, _ = mesh_faces(shape)

    clamp_labels = labels_for_file.get("clamp_labels", [])
    support_labels = labels_for_file.get("support_labels", [])

    for m_idx, mesh in enumerate(meshes):
        tris = np.array(mesh.cells).reshape(-1,3)
        n_faces = len(tris)
        face_colors = np.full((n_faces,3), 200, dtype=np.uint8)  # default gray

        # clamp coloring
        if m_idx < len(clamp_labels):
            clamp_array = clamp_labels[m_idx]
            for t_idx in range(min(len(clamp_array), n_faces)):
                if clamp_array[t_idx] == 1:
                    face_colors[t_idx] = [255,0,0]
        # support coloring
        if m_idx < len(support_labels):
            support_array = support_labels[m_idx]
            for t_idx in range(min(len(support_array), n_faces)):
                if support_array[t_idx] == 1:
                    face_colors[t_idx] = [0,0,255]

        mesh.cellcolors = face_colors
        mesh.alpha(1.0)

    # Show all meshes
    pl = Plotter(title="Selected Patches Viewer")
    for m in meshes:
        pl.add(m)
    pl.show(interactive=True, axes=1, viewup="z", resetcam=True)

# ---------------- Main ----------------
if __name__ == "__main__":
    Tk().withdraw()
    step_file = askopenfilename(title="Select STEP file", filetypes=[("STEP files","*.stp;*.step")])
    if not step_file:
        raise ValueError("No STEP file selected!")

    # Load JSON labels
    labels_file = "labels.json"
    try:
        with open(labels_file) as f:
            all_labels = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{labels_file} not found!")

    labels_for_file = all_labels.get(step_file)
    if labels_for_file is None:
        print(f"No labels found for {step_file}. Showing all gray.")
        labels_for_file = {"clamp_labels": [], "support_labels": []}

    # Read STEP
    shape = read_step_file(step_file)
    # Visualize
    visualize_patches(shape, labels_for_file)
