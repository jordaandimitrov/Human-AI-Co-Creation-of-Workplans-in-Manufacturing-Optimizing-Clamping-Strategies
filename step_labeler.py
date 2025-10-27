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
    meshes, face_indices = mesh_faces(shape, nx=20, ny=20)

    selected_points = set()  # (mesh_index, local_vertex_index)
    mode = ["clamp"]
    BRUSH_RADIUS = 5  # adjust to your model units

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
    all_spheres = []
    for m_idx, m in enumerate(meshes):
        pts = m.points
        spheres = Spheres(pts, r=0.3, c="orange")
        all_spheres.append(spheres)
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
            for t_idx, tri in enumerate(tris):
                if any(v in selected_set for v in tri):
                    face_colors[t_idx] = [255, 0, 0] if mode[0] == "clamp" else [0, 0, 255]

            # Assign per-face colors and update
            m.cellcolors = face_colors
            m.modified()
        pl.render()

    # ---------------- Callbacks ----------------
    def on_key(evt):
        key = evt.keypress.lower()
        if key == "c":
            mode[0] = "clamp"
        elif key == "u":
            mode[0] = "support"
        elif key == "s":
            clamp_labels = [0]*len(face_indices)
            support_labels = [0]*len(face_indices)
            for m_idx, _ in selected_points:
                if mode[0]=="clamp":
                    clamp_labels[m_idx] = 1
                else:
                    support_labels[m_idx] = 1
            try:
                with open(save_json) as f:
                    all_labels = json.load(f)
            except:
                all_labels = {}
            all_labels[file_path] = {
                "clamp_labels": clamp_labels,
                "support_labels": support_labels
            }
            with open(save_json, "w") as f:
                json.dump(all_labels, f, indent=4)
            print(f"💾 Saved labels for {file_path}")
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
        update_colors_mesh()  # update mesh triangles

    # ---------------- Attach callbacks ----------------
    pl.add_callback("key press", on_key)
    pl.add_callback("mouse click", on_click_brush)

    update_text()
    pl.show(interactive=True)

# ---------------- Main ----------------
if __name__ == "__main__":
    interactive_labeling("labels.json")
