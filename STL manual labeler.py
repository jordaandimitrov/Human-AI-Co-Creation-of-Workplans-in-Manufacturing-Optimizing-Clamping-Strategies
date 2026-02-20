from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json
import numpy as np
from vedo import Mesh, Plotter, Text2D

NORMAL_TOLERANCE = 0.95  # cosine similarity threshold (~18 degrees)
BRUSH_RADIUS = 5.0

# ---------------- Load STL directly ----------------
def load_stl_mesh(filename):
    mesh = Mesh(filename)
    if mesh.npoints == 0:
        raise ValueError("Failed to load STL mesh or file is empty.")
    # ensure triangles
    mesh.triangulate()
    mesh.compute_normals()
    return mesh

# ---------------- Triangle subdivision ----------------
def subdivide_mesh(mesh, levels=1):
    points = mesh.points()
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

    new_mesh = Mesh([points, tris], c=mesh.c(), alpha=mesh.alpha())
    new_mesh.compute_normals()
    return new_mesh

# ---------------- Interactive labeling ----------------
def interactive_labeling(save_json="labels.json", subdivide_levels=0, initial_brush=5.0):
    global BRUSH_RADIUS
    BRUSH_RADIUS = initial_brush

    Tk().withdraw()
    file_path = askopenfilename(
        title="Select an STL file",
        filetypes=[("STL files", "*.stl")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    print("📂 Loading STL ...")
    base_mesh = load_stl_mesh(file_path)

    # Optional subdivision
    if subdivide_levels > 0:
        print("⏳ Subdividing mesh ...")
        mesh = subdivide_mesh(base_mesh, levels=subdivide_levels)
    else:
        mesh = base_mesh

    # Extract geometry
    tris = np.array(mesh.cells).reshape(-1, 3)
    points = mesh.points

    # Triangle normals
    triangle_normals = np.array([
        np.cross(points[t[1]] - points[t[0]], points[t[2]] - points[t[0]])
        for t in tris
    ])
    triangle_normals /= np.linalg.norm(triangle_normals, axis=1)[:, None] + 1e-12

    # Selection storage
    selected_clamp = set()
    undo_stack = []
    mode = ["clamp"]

    # UI setup
    pl = Plotter(title="STL Labeler (C=Clamp, U=Support, D=Delete, S=Save, Y/T=Brush Size)")

    mode_text = Text2D(
        f"Mode: CLAMP | Brush: {BRUSH_RADIUS:.1f}",
        pos="top-left", c="yellow", bg="black", font="courier"
    )
    pl.add(mode_text)
    pl.add(mesh)

    # ---------------- Update UI ----------------
    def update_text():
        mode_text.text(f"Mode: {mode[0].upper()} | Brush: {BRUSH_RADIUS:.1f}")
        mode_text.c("yellow" if mode[0] == "clamp" else "cyan")
        pl.render()

    def update_colors():
        face_colors = np.full((len(tris), 3), 200, dtype=np.uint8)
        for i in selected_clamp:
            face_colors[i] = [255, 0, 0]

        mesh.cellcolors = face_colors
        mesh.modified()
        pl.render()

    # ---------------- Saving ----------------
    def save_labels():
        # Create a dictionary of per-triangle labels
        tri_labels = {}

        for i in range(len(tris)):
            if i in selected_clamp:
                tri_labels[str(i)] = "clamp"
            else:
                tri_labels[str(i)] = "unselected"

        # Load previous JSON if exists
        try:
            with open(save_json) as f:
                all_labels = json.load(f)
        except:
            all_labels = {}

        # Normalize path
        normalized = file_path.replace("\\", "/")
        target_folder = "training_set"

        # Try to extract relative path from training_set onward
        if target_folder in normalized:
            start_idx = normalized.find(target_folder)
            json_key = normalized[start_idx:]
        else:
            # If "training_set" not found, use filename only
            json_key = os.path.basename(normalized)

        all_labels[json_key] = tri_labels

        with open(save_json, "w") as f:
            json.dump(all_labels, f, indent=4)

        print(f"💾 Saved {len(tri_labels)} triangle labels to {save_json}")

    # ---------------- Key events ----------------
    def on_key(evt):
        global BRUSH_RADIUS
        key = evt.keypress.lower()

        if key == "1":  # undo
            if undo_stack:
                selected_clamp.clear()
                selected_clamp.update(undo_stack.pop())
                update_colors()
            return

        if key == "c":
            mode[0] = "clamp"
        elif key == "d":
            mode[0] = "delete"
        elif key == "s":
            save_labels()
        elif key == "y":
            BRUSH_RADIUS += 1.0
        elif key == "t":
            BRUSH_RADIUS = max(0.1, BRUSH_RADIUS - 0.5)

        update_text()

    # ---------------- Mouse click ----------------
    def on_click(evt):
        if evt.picked3d is None:
            return

        click_pt = np.array(evt.picked3d)
        centroids = np.mean(points[tris], axis=1)
        dists = np.linalg.norm(centroids - click_pt, axis=1)

        undo_stack.append(selected_clamp.copy())

        nearest = np.argmin(dists)
        near_normal = triangle_normals[nearest]

        # Brush selection
        if BRUSH_RADIUS <= 1:
            candidates = [nearest]
        else:
            candidates = np.where(dists <= BRUSH_RADIUS)[0]

        # Normal similarity
        candidates = [
            t for t in candidates
            if np.dot(triangle_normals[t], near_normal) >= NORMAL_TOLERANCE
        ]

        if mode[0] == "clamp":
            selected_clamp.update(candidates)
        elif mode[0] == "delete":
            selected_clamp.difference_update(candidates)

        update_colors()

    pl.add_callback("key press", on_key)
    pl.add_callback("mouse click", on_click)

    update_text()
    pl.show(interactive=True)

# ---------------- Main ----------------
if __name__ == "__main__":
    interactive_labeling("labels.json", subdivide_levels=0, initial_brush=5.0)
