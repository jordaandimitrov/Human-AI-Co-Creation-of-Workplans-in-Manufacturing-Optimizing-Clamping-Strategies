import numpy as np
import trimesh
from vedo import Mesh, Plotter, Text2D, Line, Points, Arrow
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# --- IMPORT SHARED LOGIC ---
try:
    from features import compute_outerness
except ImportError:
    print("❌ ERROR: Could not find 'features.py'. Ensure it's in the same folder.")
    exit()


def run_debug_tool():
    # 1. Load File
    Tk().withdraw()
    path = askopenfilename(title="Select STL to Debug", filetypes=[("STL", ".stl")])
    if not path: return

    print(f"Processing {path}...")
    mesh = Mesh(path)
    mesh.triangulate()

    centroids = mesh.cell_centers().points
    normals = mesh.cell_normals

    # 2. Pre-Calculate Features
    print("   1/3 Computing Outerness (Convex Hull)...")
    outerness_map = compute_outerness(mesh, centroids)

    print("   2/3 Computing Elite Mask...")
    verticality = 1.0 - np.abs(normals[:, 2])
    is_elite = (verticality > 0.8) & (outerness_map > 0.85)



    print("   3/3 Computing Full Ray Scores (X-Ray Mode)...")
    tm = trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells, process=False)
    ray_score_map = np.zeros(mesh.ncells)

    candidate_indices = np.where(is_elite)[0]

    if len(candidate_indices) > 0:
        c_norms = normals[candidate_indices]
        c_cents = centroids[candidate_indices]

        # Jitter Setup
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

            # --- ROBUST CROSS-PLATFORM RAY CAST ---
            try:
                # We ask for locations explicitly so we generally get 3 values
                # But we handle the tuple size just in case
                res = tm.ray.intersects_id(
                    ray_origins=start_pts, ray_directions=-c_norms,
                    multiple_hits=True, max_d=500.0, return_locations=True
                )

                # Dynamic Unpacking
                if len(res) == 3:
                    idx_tri, idx_ray, _ = res
                elif len(res) == 2:
                    idx_tri, idx_ray = res
                else:
                    idx_tri, idx_ray = [], []

            except Exception as e:
                print(f"   ⚠️ Ray cast failed: {e}")
                idx_tri, idx_ray = [], []
            # --------------------------------------

            if len(idx_ray) > 0:
                s_idx = idx_ray
                t_idx = idx_tri

                s_n = c_norms[s_idx]
                t_n = normals[t_idx]
                t_out = outerness_map[t_idx]

                align = np.einsum('ij,ij->i', s_n, t_n)
                t_vert = 1.0 - np.abs(t_n[:, 2])

                # Check
                valid = (align < -0.85) & (t_vert > 0.85) & (t_out > 0.85)

                good_rays = s_idx[valid]
                unique_good = np.unique(good_rays)
                hits[unique_good] += 0.2

        ray_score_map[candidate_indices] = hits

    # ==========================================
    # 3. GUI SETUP
    # ==========================================
    plt = Plotter(title="Feature Debugger", bg="blackboard", axes=1)

    state = {
        "mode": "outerness",
        "debug_actors": []
    }

    info_txt = Text2D(
        "Click a face to fire laser beams.\nCheck console for hit details.",
        pos="bottom-left", s=0.9, c="gray"
    )
    plt.add(info_txt)

    def set_mode(mode_name):
        state["mode"] = mode_name
        mesh.c("gold")  # Clear previous colors

        if mode_name == "outerness":
            mesh.cmap("jet", outerness_map, vmin=0, vmax=1, on='cells')
            # Safe scalarbar removal
            try:
                plt.remove("Legend")
            except:
                pass
            mesh.add_scalarbar(title="Outerness")

        elif mode_name == "rays":
            mesh.cmap("jet", ray_score_map, vmin=0, vmax=1, on='cells')
            try:
                plt.remove("Legend")
            except:
                pass
            mesh.add_scalarbar(title="Ray Score")

        elif mode_name == "mask":
            cols = np.full((mesh.ncells, 3), 80, dtype=np.uint8)
            cols[is_elite] = [0, 255, 0]
            mesh.cellcolors = cols
            try:
                plt.remove("Legend")
            except:
                pass

        plt.render()

    # --- FIX 1: ACCEPT ARGUMENTS (*args) ---
    def btn_outer(*args):
        set_mode("outerness")

    def btn_rays(*args):
        set_mode("rays")

    def btn_mask(*args):
        set_mode("mask")

    plt.add_button(btn_outer, states=[" Show Outerness "], c=["w"], bc=["r"], pos=(0.2, 0.05), size=25, font="courier")
    plt.add_button(btn_rays, states=[" Show Ray Scores "], c=["w"], bc=["b"], pos=(0.5, 0.05), size=25, font="courier")
    plt.add_button(btn_mask, states=[" Show Elite Mask "], c=["black"], bc=["g"], pos=(0.8, 0.05), size=25,
                   font="courier")

    # --- FIX 2: ROBUST CLICK HANDLER ---
    def on_click(evt):
        # Check if we actually clicked the mesh
        if not evt.actor: return

        # Check if a 3D point was actually picked
        pt = evt.picked3d
        if pt is None: return

        plt.remove(state["debug_actors"])
        state["debug_actors"] = []

        res = mesh.closest_point(pt, return_cell_id=True)
        fid = res[-1] if isinstance(res, (list, tuple)) else res

        print(f"\n{'=' * 40}")
        print(f"🔍 INSPECTING FACE {fid}")
        print(f"   Outerness: {outerness_map[fid]:.4f}")

        if not is_elite[fid]:
            print("   ⚠️ Cannot shoot rays: Face is not an Elite Candidate.")
            return

        c_cent = centroids[fid]
        c_norm = normals[fid]

        world_z = np.array([0, 0, 1])
        tan = np.cross(c_norm, world_z)
        tan = tan / (np.linalg.norm(tan) + 1e-6)
        bitan = np.cross(c_norm, tan)
        off = 3.0

        origins = [c_cent, c_cent + tan * off, c_cent - tan * off, c_cent + bitan * off, c_cent - bitan * off]
        labels = ["Center", "Right ", "Left  ", "Up    ", "Down  "]

        for i, start in enumerate(origins):
            direction = -c_norm

            # X-RAY MODE DEBUGGING
            idx_tri, _, locs = tm.ray.intersects_id(
                ray_origins=[start - c_norm * 0.01],
                ray_directions=[direction],
                multiple_hits=True,
                max_d=500.0,
                return_locations=True
            )

            if len(idx_tri) > 0:
                dists = np.linalg.norm(locs - start, axis=1)
                sorted_indices = np.argsort(dists)
                idx_tri = idx_tri[sorted_indices]
                locs = locs[sorted_indices]

                found_valid = False

                for k, t_idx in enumerate(idx_tri):
                    t_loc = locs[k]
                    t_out = outerness_map[t_idx]
                    t_norm = normals[t_idx]

                    align = np.dot(c_norm, t_norm)
                    t_vert = 1.0 - abs(t_norm[2])

                    if (align < -0.85) and (t_vert > 0.85) and (t_out > 0.85):
                        color = "green"
                        ln = Line(start, t_loc, c=color, lw=4)
                        pt = Points([t_loc], r=12, c=color)
                        state["debug_actors"].extend([ln, pt])
                        print(f"   [{labels[i]}] -> ✅ VALID HIT (Passed {k} inner walls)")
                        found_valid = True
                        break
                    else:
                        # Visualize ignored inner wall hits
                        pt = Points([t_loc], r=5, c="orange")
                        state["debug_actors"].append(pt)

                if not found_valid:
                    print(f"   [{labels[i]}] -> ❌ HIT INVALID (Only Recesses found)")
                    ln = Line(start, locs[-1], c="orange", lw=2)
                    state["debug_actors"].append(ln)

            else:
                vis_end = start + (direction * 80.0)
                ln = Line(start, vis_end, c="red", lw=1, alpha=0.3)
                state["debug_actors"].append(ln)
                print(f"   [{labels[i]}] -> 💨 MISS")

        plt.add(state["debug_actors"])
        plt.render()

    plt.add_callback("mouse click", on_click)
    set_mode("outerness")
    plt.show(mesh, interactive=True)


if __name__ == "__main__":
    run_debug_tool()