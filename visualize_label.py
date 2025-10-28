from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json
import numpy as np
from vedo import Mesh, Plotter
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods

# ---------------- STEP file reading ----------------
def read_step_file(filename):
    reader = STEPControl_Reader()
    status = reader.ReadFile(filename)
    if status != IFSelect_RetDone:
        raise ValueError("Error reading STEP file")
    reader.TransferRoots()
    return reader.OneShape()

# ---------------- Mesh generation ----------------
def mesh_faces(shape, linear_deflection=0.05, angular_deflection=0.5):
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)

    all_points = []
    all_tris = []
    point_map = {}
    current_index = 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        triangulation = BRep_Tool.Triangulation(face, face.Location())
        if triangulation is None:
            exp.Next()
            continue

        n_nodes = triangulation.NbNodes()
        nodes = np.array([[triangulation.Node(i).X(),
                           triangulation.Node(i).Y(),
                           triangulation.Node(i).Z()] for i in range(1, n_nodes+1)])

        n_tris = triangulation.NbTriangles()
        tris = np.array([[triangulation.Triangle(i).Value(1)-1,
                          triangulation.Triangle(i).Value(2)-1,
                          triangulation.Triangle(i).Value(3)-1] for i in range(1, n_tris+1)])

        for i, p in enumerate(nodes):
            key = tuple(np.round(p, 6))
            if key not in point_map:
                point_map[key] = current_index
                all_points.append(p)
                current_index += 1

        for tri in tris:
            all_tris.append([point_map[tuple(np.round(nodes[i], 6))] for i in tri])

        exp.Next()

    if len(all_points) == 0 or len(all_tris) == 0:
        return None

    return Mesh([np.array(all_points), np.array(all_tris)], c="lightgray", alpha=0.7)

# ---------------- Subdivision (same as labeler) ----------------
def subdivide_mesh(mesh, levels=1):
    points = mesh.points
    tris = np.array(mesh.cells).reshape(-1, 3)

    for _ in range(levels):
        new_tris = []
        new_points = points.tolist()
        midpoint_map = {}

        def get_midpoint(i, j):
            key = tuple(sorted([i, j]))
            if key in midpoint_map:
                return midpoint_map[key]
            m = (points[i] + points[j]) / 2
            new_points.append(m)
            idx = len(new_points) - 1
            midpoint_map[key] = idx
            return idx

        for tri in tris:
            v0, v1, v2 = tri
            m01 = get_midpoint(v0, v1)
            m12 = get_midpoint(v1, v2)
            m20 = get_midpoint(v2, v0)
            new_tris.extend([
                [v0, m01, m20],
                [v1, m12, m01],
                [v2, m20, m12],
                [m01, m12, m20]
            ])

        points = np.array(new_points)
        tris = np.array(new_tris)

    return Mesh([points, tris], c=mesh.c(), alpha=mesh.alpha())

# ---------------- Visualization ----------------
def visualize_labels(step_file, labels_file):
    # Load JSON labels
    try:
        with open(labels_file) as f:
            all_labels = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{labels_file} not found!")

    labels_for_file = all_labels.get(step_file)
    if labels_for_file is None:
        print(f"No labels found for {step_file}. Showing all gray.")
        labels_for_file = {"clamp_labels": [], "support_labels": [], "subdivide_levels": 0}

    subdivide_levels = labels_for_file.get("subdivide_levels", 0)
    clamp_points = labels_for_file.get("clamp_labels", [])
    support_points = labels_for_file.get("support_labels", [])

    # Read STEP and generate mesh
    shape = read_step_file(step_file)
    mesh = mesh_faces(shape)
    if mesh is None:
        raise ValueError("Could not generate mesh from STEP file.")

    # ---------------- Apply the same subdivision as labeler ----------------
    if subdivide_levels > 0:
        mesh = subdivide_mesh(mesh, levels=subdivide_levels)

    # ---------------- Apply discrete coloring per triangle ----------------
    tris = np.array(mesh.cells).reshape(-1, 3)
    n_faces = len(tris)
    face_colors = np.full((n_faces, 3), 200, dtype=np.uint8)  # default gray

    for i, tri in enumerate(tris):
        if any(idx < len(clamp_points) and clamp_points[idx] == 1 for idx in tri):
            face_colors[i] = [255, 0, 0]
        elif any(idx < len(support_points) and support_points[idx] == 1 for idx in tri):
            face_colors[i] = [0, 0, 255]

    mesh.cellcolors = face_colors
    mesh.alpha(1.0)

    # ---------------- Show mesh ----------------
    pl = Plotter(title="Selected Patches Viewer")
    pl.add(mesh)
    pl.show(interactive=True, axes=1, viewup="z", resetcam=True)

# ---------------- Main ----------------
if __name__ == "__main__":
    Tk().withdraw()
    step_file = askopenfilename(title="Select STEP file", filetypes=[("STEP files","*.stp;*.step")])
    if not step_file:
        raise ValueError("No STEP file selected!")

    labels_file = "labels.json"
    visualize_labels(step_file, labels_file)
