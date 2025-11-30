# debug_features.py
import numpy as np
import trimesh
from vedo import Mesh, Plotter, Text2D, Line, Points
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# --- IMPORT FROM YOUR FEATURE SCRIPT ---
# This ensures the heatmap matches your training data exactly.
try:
    from features import compute_outerness
except ImportError:
    print("❌ ERROR: Could not find 'features.py'. Make sure it is in the same folder.")
    exit()


def run_debug_tool():
    Tk().withdraw()
    path = askopenfilename(title="Select STL to Debug", filetypes=[("STL", ".stl")])
    if not path: return

    print(f"Loading {path}...")
    mesh = Mesh(path)
    mesh.triangulate()  # Ensure consistency

    centroids = mesh.cell_centers().points
    normals = mesh.cell_normals

    # ==========================================
    # 1. USE IMPORTED LOGIC (Convex Hull)
    # ==========================================
    print("Computing Outerness (Convex Hull)...")
    outerness_map = compute_outerness(mesh, centroids)
    print("Computing Outerness (Convex Hull)...")
    outerness_map = compute_outerness(mesh, centroids)

    # --- SANITY CHECK ---
    print(f"📊 DEBUG STATS:")
    print(f"   Min Outerness: {np.min(outerness_map):.4f}")
    print(f"   Max Outerness: {np.max(outerness_map):.4f}")
    print(f"   Mean Outerness: {np.mean(outerness_map):.4f}")
    # --------------------
    # ==========================================
    # 2. RE-RUN RAY CASTING FOR VISUALIZATION
    # ==========================================
    # We re-run this locally so we can create the heatmap
    print("Computing Ray Scores...")
    tm = trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells, process=False)
    verticality = 1.0 - np.abs(normals[:, 2])

    # Elite Definition (Must match features.py threshold)
    is_elite = (verticality > 0.8) & (outerness_map > 0.85)

    candidate_indices = np.where(is_elite)[0]
    ray_score_map = np.zeros(mesh.ncells)

    if len(candidate_indices) > 0:
        c_norms = normals[candidate_indices]
        c_cents = centroids[candidate_indices]

        world_z = np.array([0, 0, 1])
        tangent = np.cross(c_norms, world_z)
        norms = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-6
        tangent = tangent / norms
        bitangent = np.cross(c_norms, tangent)
        offset = 3.0

        origins_list = [
            c_cents,
            c_cents + (tangent * offset),
            c_cents - (tangent * offset),
            c_cents + (bitangent * offset),
            c_cents - (bitangent * offset)
        ]

        hits = np.zeros(len(candidate_indices))

        for origins in origins_list:
            start_pts = origins - (c_norms * 0.01)
            try:
                idx_tri, idx_ray, _ = tm.ray.intersects_id(
                    ray_origins=start_pts, ray_directions=-c_norms,
                    multiple_hits=False, max_d=500.0
                )
            except:
                idx_tri, idx_ray = [], []

            if len(idx_ray) > 0:
                s_idx = idx_ray
                t_idx = idx_tri

                # Check Logic
                s_n = c_norms[s_idx]
                t_n = normals[t_idx]
                align = np.einsum('ij,ij->i', s_n, t_n)
                t_vert = 1.0 - np.abs(t_n[:, 2])

                # TARGET ELITE CHECK (Uses imported outerness)
                t_out = outerness_map[t_idx]
                t_is_elite = t_out > 0.85

                valid = (align < -0.85) & (t_vert > 0.85) & t_is_elite
                hits[s_idx[valid]] += 0.2

        ray_score_map[candidate_indices] = hits

    # ==========================================
    # 3. INTERACTIVE PLOTTER
    # ==========================================
    plt = Plotter(title="Convex Hull Debugger", bg="blackboard", axes=1)

    txt_mode = Text2D("Mode: 1 - Convex Hull Heatmap", pos="top-left", s=1.1, c="yellow")
    txt_info = Text2D("Click a face to inspect", pos="bottom-left", s=0.9, c="white")

    state = {"mode": 1, "actors": []}

    def update_view():
        # FIX: Do not use cellcolors = None.
        # Instead, set the mesh to a solid base color to clear previous arrays.
        mesh.c("gold")

        if state["mode"] == 1:
            txt_mode.text("Mode: 1 - Outerness (Red=Outer, Blue=Inner)")
            # Map scalar data to cells
            mesh.cmap("jet", outerness_map, vmin=0.0, vmax=1.0, on='cells')

        elif state["mode"] == 2:
            txt_mode.text("Mode: 2 - Ray Score (Red=Good, Blue=Bad)")
            # Map scalar data to cells
            mesh.cmap("jet", ray_score_map, vmin=0.0, vmax=1.0, on='cells')

        elif state["mode"] == 3:
            txt_mode.text("Mode: 3 - Elite Mask (Green=Candidate)")
            # Manual coloring
            cols = np.full((mesh.ncells, 3), 50, dtype=np.uint8)  # Dark Grey
            cols[is_elite] = [0, 255, 0]  # Green
            mesh.cellcolors = cols

        plt.render()

    def on_key(evt):
        if evt.keypress == "1":
            state["mode"] = 1
        elif evt.keypress == "2":
            state["mode"] = 2
        elif evt.keypress == "3":
            state["mode"] = 3
        update_view()

    def on_click(evt):
        if not evt.actor: return
        plt.remove(state["actors"])
        state["actors"] = []

        # Get Face ID
        pt = evt.picked3d
        res = mesh.closest_point(pt, return_cell_id=True)
        fid = res[-1] if isinstance(res, (list, tuple)) else res

        print(f"\n--- Face {fid} ---")
        val = outerness_map[fid]
        print(f"Outerness: {val:.4f}")

        if val < 0.85:
            print("⚠️ REJECTED: Too deep inside (Threshold 0.85)")
            return

        if not is_elite[fid]:
            print("⚠️ REJECTED: Not vertical enough")
            return

        # Visualize Rays for this specific face
        c_cent = centroids[fid]
        c_norm = normals[fid]

        # Jitter
        world_z = np.array([0, 0, 1])
        tan = np.cross(c_norm, world_z)
        tan = tan / (np.linalg.norm(tan) + 1e-6)
        bitan = np.cross(c_norm, tan)
        off = 3.0

        origins = [c_cent, c_cent + tan * off, c_cent - tan * off, c_cent + bitan * off, c_cent - bitan * off]
        names = ["Center", "Right", "Left", "Up", "Down"]

        for i, start in enumerate(origins):
            direction = -c_norm
            # Shoot 1 ray
            idx_tri, _, locs = tm.ray.intersects_id(
                ray_origins=[start - c_norm * 0.01], ray_directions=[direction],
                multiple_hits=False, max_d=500.0, return_locations=True
            )

            color = "red"
            if len(idx_tri) > 0:
                t_idx = idx_tri[0]
                t_out = outerness_map[t_idx]  # IMPORTED CHECK

                t_norm = normals[t_idx]
                align = np.dot(c_norm, t_norm)
                t_vert = 1.0 - abs(t_norm[2])

                if t_out > 0.85 and align < -0.85 and t_vert > 0.85:
                    color = "green"
                    print(f"Ray {i}: VALID Hit (Target Outerness {t_out:.2f})")
                else:
                    color = "orange"
                    print(f"Ray {i}: INVALID Hit (Target Outerness {t_out:.2f})")

                ln = Line(start, locs[0], c=color, lw=3)
                pt = Points([locs[0]], r=8, c=color)
                state["actors"].extend([ln, pt])
            else:
                print(f"Ray {i}: MISS (Slot?)")
                ln = Line(start, start + direction * 50, c="red", lw=1, alpha=0.5)
                state["actors"].append(ln)

        plt.add(state["actors"])
        plt.render()

    plt.add_callback("key press", on_key)
    plt.add_callback("mouse click", on_click)
    update_view()
    plt.show(mesh, interactive=True)


if __name__ == "__main__":
    run_debug_tool()