from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json
import numpy as np
from vedo import Mesh, Plotter, Text2D
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
    """Tessellate entire STEP shape and produce a single connected mesh."""
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

        # Vertices
        n_nodes = triangulation.NbNodes()
        nodes = np.array([[triangulation.Node(i).X(),
                           triangulation.Node(i).Y(),
                           triangulation.Node(i).Z()] for i in range(1, n_nodes+1)])

        # Triangles
        n_tris = triangulation.NbTriangles()
        tris = np.array([[triangulation.Triangle(i).Value(1)-1,
                          triangulation.Triangle(i).Value(2)-1,
                          triangulation.Triangle(i).Value(3)-1] for i in range(1, n_tris+1)])

        # Weld vertices
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

# ---------------- Triangle subdivision ----------------
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

# ---------------- Interactive labeling ----------------
def interactive_labeling(save_json="labels.json", subdivide_levels=1):
    Tk().withdraw()
    file_path = askopenfilename(
        title="Select an STP file",
        filetypes=[("STP files", "*.stp"), ("STEP files", "*.step")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    shape = read_step_file(file_path)
    mesh = mesh_faces(shape, linear_deflection=0.05, angular_deflection=0.5)
    if mesh is None:
        print("No valid mesh generated.")
        return

    # Subdivide triangles for more points
    if subdivide_levels > 0:
        mesh = subdivide_mesh(mesh, levels=subdivide_levels)

    selected_points = set()  # global vertex indices
    mode = ["clamp"]
    BRUSH_RADIUS = 5.0  # adjust to your model units

    pl = Plotter(title="Brush Labeler (C/U=mode, S=save, click to paint)")

    mode_text = Text2D("Mode: CLAMP  [C]=Clamp  [U]=Support  [S]=Save",
                       pos="top-left", c="yellow", bg="black", font="courier")
    pl.add(mode_text)
    pl.add(mesh)

    def update_text():
        mode_text.text(f"Mode: {mode[0].upper()}  [C]=Clamp  [U]=Support  [S]=Save")
        mode_text.c("yellow" if mode[0] == "clamp" else "cyan")
        pl.render()

    def update_colors():
        tris = np.array(mesh.cells).reshape(-1, 3)
        face_colors = np.full((len(tris), 3), 200, dtype=np.uint8)
        for i, tri in enumerate(tris):
            if any(v in selected_points for v in tri):
                face_colors[i] = [255, 0, 0] if mode[0]=="clamp" else [0, 0, 255]
        mesh.cellcolors = face_colors
        mesh.modified()
        pl.render()

    def save_labels():
        clamp_array = [1 if i in selected_points else 0 for i in range(len(mesh.points))]
        support_array = [0]*len(mesh.points)
        try:
            with open(save_json) as f:
                all_labels = json.load(f)
        except:
            all_labels = {}
        all_labels[file_path] = {
            "clamp_labels": clamp_array,
            "support_labels": support_array
        }
        with open(save_json, "w") as f:
            json.dump(all_labels, f, indent=4)
        print(f"💾 Saved labels for {file_path}")

    def on_key(evt):
        key = evt.keypress.lower()
        if key == "c":
            mode[0] = "clamp"
        elif key == "u":
            mode[0] = "support"
        elif key == "s":
            save_labels()
        update_text()
        update_colors()

    def on_click(evt):
        if evt.picked3d is None:
            return
        click_point = np.array(evt.picked3d)
        dists = np.linalg.norm(mesh.points - click_point, axis=1)
        selected = np.where(dists <= BRUSH_RADIUS)[0]
        for idx in selected:
            selected_points.add(idx)
        update_colors()

    pl.add_callback("key press", on_key)
    pl.add_callback("mouse click", on_click)

    update_text()
    pl.show(interactive=True)

# ---------------- Main ----------------
if __name__ == "__main__":
    interactive_labeling("labels.json", subdivide_levels=2)
