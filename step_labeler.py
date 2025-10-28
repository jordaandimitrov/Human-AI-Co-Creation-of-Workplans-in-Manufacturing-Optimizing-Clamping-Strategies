from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json
import numpy as np
from vedo import Mesh, Plotter, Text2D, Spheres
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

# ---------------- STEP and meshing ----------------
def read_step_file(filename):
    reader = STEPControl_Reader()
    status = reader.ReadFile(filename)
    if status != IFSelect_RetDone:
        raise ValueError("Error reading STEP file")
    reader.TransferRoots()
    return reader.OneShape()



from OCC.Core.gp import gp_Pnt2d
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.TopAbs import TopAbs_IN
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepTools import breptools

import numpy as np

def sample_face(face, nx=30, ny=30, uv_range=None):
    adaptor = BRepAdaptor_Surface(face)
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    if uv_range:
        (umin, umax), (vmin, vmax) = uv_range

    u = np.linspace(umin, umax, nx)
    v = np.linspace(vmin, vmax, ny)

    classifier = BRepClass_FaceClassifier()

    # store grid of valid/invalid sample points
    valid = np.zeros((ny, nx), dtype=bool)
    points = np.zeros((ny, nx, 3), dtype=float)

    for i, vi in enumerate(v):
        for j, ui in enumerate(u):
            uv = gp_Pnt2d(ui, vi)
            classifier.Perform(face, uv, 1e-6, False, 1e-6)
            if classifier.State() == TopAbs_IN:
                p = adaptor.Value(ui, vi)
                points[i, j] = [p.X(), p.Y(), p.Z()]
                valid[i, j] = True

    # build a clean list of points and connectivity
    flat_points = []
    index_map = -np.ones((ny, nx), dtype=int)
    idx = 0
    for i in range(ny):
        for j in range(nx):
            if valid[i, j]:
                index_map[i, j] = idx
                flat_points.append(points[i, j])
                idx += 1
    flat_points = np.array(flat_points)

    tris = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            # add triangles only if all corners are valid
            if valid[i, j] and valid[i + 1, j] and valid[i, j + 1]:
                tris.append([index_map[i, j], index_map[i + 1, j], index_map[i, j + 1]])
            if valid[i + 1, j] and valid[i + 1, j + 1] and valid[i, j + 1]:
                tris.append([index_map[i + 1, j], index_map[i + 1, j + 1], index_map[i, j + 1]])

    tris = np.array(tris, dtype=int)

    if len(flat_points) == 0 or len(tris) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return flat_points, tris





from scipy.spatial import cKDTree
from vedo import Mesh
import numpy as np
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods

from scipy.spatial import cKDTree
from vedo import Mesh
import numpy as np
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods

def mesh_faces(shape, nx=20, ny=20, merge_tol=1e-4):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    all_nodes = []
    all_tris = []
    node_offset = 0

    while exp.More():
        face = topods.Face(exp.Current())
        nodes, tris = sample_face(face, nx=nx, ny=ny)
        if len(nodes) == 0 or len(tris) == 0:
            exp.Next()
            continue
        all_nodes.append(nodes)
        all_tris.append(tris + node_offset)
        node_offset += len(nodes)
        exp.Next()

    if not all_nodes:
        return None

    all_nodes = np.vstack(all_nodes)
    all_tris = np.vstack(all_tris)

    # Weld close vertices
    tree = cKDTree(all_nodes)
    groups = tree.query_ball_point(all_nodes, merge_tol)

    mapping = -np.ones(len(all_nodes), dtype=int)
    unique_pts = []
    counter = 0
    for i, g in enumerate(groups):
        if mapping[i] == -1:
            for j in g:
                mapping[j] = counter
            unique_pts.append(all_nodes[i])
            counter += 1

    unique_pts = np.array(unique_pts)
    remapped_tris = mapping[all_tris]

    mesh = Mesh([unique_pts, remapped_tris], c="lightgray", alpha=0.7)
    return mesh




# ---------------- Interactive labeling ----------------
def interactive_labeling(save_json="labels.json"):
    Tk().withdraw()
    file_path = askopenfilename(
        title="Select an STP file",
        filetypes=[("STP files", "*.stp"), ("STEP files", "*.step")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    shape = read_step_file(file_path)
    merged_mesh = mesh_faces(shape, nx=20, ny=20)
    if merged_mesh is None:
        print("No valid mesh generated.")
        return

    meshes = [merged_mesh]
    face_indices = [0]

    selected_points = set()  # (mesh_index, local_vertex_index)
    selected_faces = {m_idx: set() for m_idx in range(len(meshes))}
    mode = ["clamp"]
    BRUSH_RADIUS = 5.0  # adjust to your model units

    # Flatten points for distance computations and store start indices
    all_points = []
    mesh_start_idx = []
    current_idx = 0
    for m_idx, m in enumerate(meshes):
        pts = m.points
        all_points.append(pts)
        mesh_start_idx.append(current_idx)
        current_idx += len(pts)
    all_points = np.vstack(all_points)
    mesh_start_idx = np.array(mesh_start_idx)

    # Map global index -> mesh index and local index
    global_to_mesh_local = {}
    for m_idx, start_idx in enumerate(mesh_start_idx):
        n_pts = len(meshes[m_idx].points)
        for i in range(n_pts):
            global_to_mesh_local[start_idx + i] = (m_idx, i)

    pl = Plotter(title="Brush Labeler (C/U=mode, S=save, click to paint)")

    mode_text = Text2D("Mode: CLAMP  [C]=Clamp  [U]=Support  [S]=Save",
                       pos="top-left", c="yellow", bg="black", font="courier")
    pl.add(mode_text)

    # ---------------- Add meshes ----------------
    for m in meshes:
        pl.add(m)

    # Optional: spheres at all points (can be removed for performance)
    for m_idx, m in enumerate(meshes):
        pts = m.points
        spheres = Spheres(pts, r=0.3, c="orange")
        pl.add(spheres)

    # ---------------- Functions ----------------
    def update_text():
        mode_text.text(f"Mode: {mode[0].upper()}  [C]=Clamp  [U]=Support  [S]=Save")
        mode_text.c("yellow" if mode[0] == "clamp" else "cyan")

    def update_colors_mesh():
        for m_idx, m in enumerate(meshes):
            tris = np.array(m.cells).reshape(-1,3)  # robust for all vedo versions
            n_faces = len(tris)
            face_colors = np.full((n_faces, 3), 200, dtype=np.uint8)
            selected_local = [local for mi, local in selected_points if mi == m_idx]
            selected_set = set(selected_local)
            selected_faces[m_idx].clear()  # reset
            for t_idx, tri in enumerate(tris):
                if any(v in selected_set for v in tri):
                    face_colors[t_idx] = [255, 0, 0] if mode[0] == "clamp" else [0, 0, 255]
                    selected_faces[m_idx].add(t_idx)
            m.cellcolors = face_colors
            m.modified()
        pl.render()

    def save_labels():
        all_clamp = []
        all_support = []
        for m_idx, m in enumerate(meshes):
            n_faces = len(np.array(meshes[m_idx].cells).reshape(-1,3))
            clamp_array = [1 if f in selected_faces[m_idx] and mode[0]=="clamp" else 0 for f in range(n_faces)]
            support_array = [1 if f in selected_faces[m_idx] and mode[0]=="support" else 0 for f in range(n_faces)]
            all_clamp.append(clamp_array)
            all_support.append(support_array)

        try:
            with open(save_json) as f:
                all_labels = json.load(f)
        except:
            all_labels = {}
        all_labels[file_path] = {
            "clamp_labels": all_clamp,
            "support_labels": all_support
        }
        with open(save_json, "w") as f:
            json.dump(all_labels, f, indent=4)
        print(f"💾 Saved labels for {file_path}")

    # ---------------- Callbacks ----------------
    def on_key(evt):
        key = evt.keypress.lower()
        if key == "c":
            mode[0] = "clamp"
        elif key == "u":
            mode[0] = "support"
        elif key == "s":
            save_labels()
        update_text()
        update_colors_mesh()

    def on_click_brush(evt):
        if evt.picked3d is None:
            return
        click_point = np.array(evt.picked3d)
        dists = np.linalg.norm(all_points - click_point, axis=1)
        selected_global = np.where(dists <= BRUSH_RADIUS)[0]
        for g_idx in selected_global:
            m_idx, local_idx = global_to_mesh_local[g_idx]
            selected_points.add((m_idx, local_idx))
        update_colors_mesh()

    # ---------------- Attach callbacks ----------------
    pl.add_callback("key press", on_key)
    pl.add_callback("mouse click", on_click_brush)

    update_text()
    pl.show(interactive=True)

# ---------------- Main ----------------
if __name__ == "__main__":
    interactive_labeling("labels.json")
