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

    # ============================================================
    # 2. PRE-CALCULATE FEATURES
    # ============================================================
    print("   1/4 Computing Outerness (Convex Hull)...")
    outerness_map = compute_outerness(mesh, centroids)

    print("   2/4 Computing Elite Mask...")
    verticality = 1.0 - np.abs(normals[:, 2])
    is_elite = (outerness_map > 0.85)

    print("   3/4 Computing Cylindricity Feature...")

    # -----------------------------
    # RELAXED + OUTER-MASKED + CURVATURE-AWARE CYLINDRICITY
    # -----------------------------
    part_center = np.mean(centroids, axis=0)

    # radial vector & alignment
    radial_vec = np.column_stack([
        centroids[:, 0] - part_center[0],
        centroids[:, 1] - part_center[1],
        np.zeros(len(centroids))
    ])
    radial_norm = np.linalg.norm(radial_vec, axis=1) + 1e-6
    radial_unit = radial_vec / radial_norm[:, None]
    base_cyl = np.abs(np.einsum("ij,ij->i", radial_unit, normals)).astype(np.float32)

    # radius smoothness
    radius = radial_norm
    neighbor_radius = np.zeros_like(radius)
    for i in range(len(radius)):
        d = np.sum((centroids - centroids[i]) ** 2, axis=1)
        nn = np.argpartition(d, 6)[:6]
        neighbor_radius[i] = np.mean(radius[nn])

    smooth = np.exp(-3 * np.abs(radius - neighbor_radius))  # relaxed smoothing
    alpha = 0.7

    # compute rough curvature to suppress flat triangles
    curvature = np.zeros(len(normals), dtype=np.float32)
    for i in range(len(normals)):
        dists = np.sum((centroids - centroids[i]) ** 2, axis=1)
        nn = np.argpartition(dists, 6)[:6]
        neighbor_norms = normals[nn]
        curvature[i] = np.mean(np.linalg.norm(neighbor_norms - normals[i], axis=1))

    # only apply cylindricity to triangles with outerness > 0.8
    outer_mask = outerness_map > 0.8
    cylindricity_map = np.zeros_like(base_cyl)

    # compute curvature
    curvature = np.zeros(len(normals), dtype=np.float32)
    for i in range(len(normals)):
        dists = np.sum((centroids - centroids[i]) ** 2, axis=1)
        nn = np.argpartition(dists, 12)[:12]  # more neighbors
        neighbor_norms = normals[nn]
        curvature[i] = np.mean(np.linalg.norm(neighbor_norms - normals[i], axis=1))

    # relax curvature
    k = 5.0
    curvature_relaxed = 1 - np.exp(-k * curvature)  # exponential relaxation
    alpha_curv = 0.7
    curvature_final = alpha_curv * curvature_relaxed + (1 - alpha_curv)

    # multiply by cylindricity
    cylindricity_map[outer_mask] = base_cyl[outer_mask] * (alpha * smooth[outer_mask] + (1 - alpha)) * curvature_final[
        outer_mask]



    # normalize
    min_val, max_val = cylindricity_map.min(), cylindricity_map.max()
    cylindricity_map = (cylindricity_map - min_val) / (max_val - min_val + 1e-6)
    # -----------------------------

    print("   4/4 Computing Full Ray Scores (X-Ray Mode)...")
    tm = trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells, process=False)
    ray_score_map = np.zeros(mesh.ncells)

    candidate_indices = np.where(is_elite)[0]

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
            c_cents + tangent * offset,
            c_cents - tangent * offset,
            c_cents + bitangent * offset,
            c_cents - bitangent * offset,
        ]

        hits = np.zeros(len(candidate_indices))

        for origins in origins_list:
            start_pts = origins - c_norms * 0.01

            try:
                res = tm.ray.intersects_id(
                    ray_origins=start_pts,
                    ray_directions=-c_norms,
                    multiple_hits=True,
                    max_d=500.0,
                    return_locations=True
                )

                if len(res) == 3:
                    idx_tri, idx_ray, _ = res
                elif len(res) == 2:
                    idx_tri, idx_ray = res
                else:
                    idx_tri, idx_ray = [], []

            except Exception as e:
                print(f"   ⚠️ Ray cast failed: {e}")
                idx_tri, idx_ray = [], []

            if len(idx_ray) > 0:
                s_idx = idx_ray
                t_idx = idx_tri
                s_n = c_norms[s_idx]
                t_n = normals[t_idx]
                t_out = outerness_map[t_idx]

                align = np.einsum("ij,ij->i", s_n, t_n)
                t_vert = 1.0 - np.abs(t_n[:, 2])
                valid = (align < -0.85) & (t_vert > 0.85) & (t_out > 0.85)

                good_rays = s_idx[valid]
                hits[np.unique(good_rays)] += 0.2

        ray_score_map[candidate_indices] = hits

    # ==========================================
    # 3. GUI SETUP
    # ==========================================
    plt = Plotter(title="Feature Debugger", bg="blackboard", axes=1)

    state = {"mode": "outerness", "debug_actors": []}

    info_txt = Text2D(
        "Click a face to fire laser beams.\nCheck console for hit details.",
        pos="bottom-left", s=0.9, c="gray"
    )
    plt.add(info_txt)

    # ======================================================
    # Rendering modes
    # ======================================================
    def set_mode(mode_name):
        state["mode"] = mode_name
        mesh.c("gold")

        try: plt.remove("Legend")
        except: pass

        if mode_name == "outerness":
            mesh.cmap("jet", outerness_map, vmin=0, vmax=1, on="cells")
            mesh.add_scalarbar(title="Outerness")

        elif mode_name == "rays":
            mesh.cmap("jet", ray_score_map, vmin=0, vmax=1, on="cells")
            mesh.add_scalarbar(title="Ray Score")

        elif mode_name == "mask":
            cols = np.full((mesh.ncells, 3), 80, dtype=np.uint8)
            cols[is_elite] = [0, 255, 0]
            mesh.cellcolors = cols

        elif mode_name == "cyl":
            mesh.cmap("jet", cylindricity_map, vmin=0, vmax=1, on="cells")
            mesh.add_scalarbar(title="Cylindricity")

        plt.render()

    # buttons
    def btn_outer(*a): set_mode("outerness")
    def btn_rays(*a): set_mode("rays")
    def btn_mask(*a): set_mode("mask")
    def btn_cyl(*a): set_mode("cyl")

    plt.add_button(btn_outer, states=[" Outerness "], c=["w"], bc=["r"], pos=(0.18, 0.05), size=25)
    plt.add_button(btn_rays,  states=[" Ray Scores "], c=["w"], bc=["b"], pos=(0.42, 0.05), size=25)
    plt.add_button(btn_mask,  states=[" Elite Mask "], c=["black"], bc=["g"], pos=(0.66, 0.05), size=25)
    plt.add_button(btn_cyl,   states=[" Cylindricity "], c=["black"], bc=["yellow"], pos=(0.90, 0.05), size=25)

    # Click handler unchanged
    def on_click(evt):
        if not evt.actor: return
        pt = evt.picked3d
        if pt is None: return

        plt.remove(state["debug_actors"])
        state["debug_actors"] = []

        res = mesh.closest_point(pt, return_cell_id=True)
        fid = res[-1] if isinstance(res, (list, tuple)) else res

        print("\n========================================")
        print(f"🔍 Face {fid}")
        print(f"   Outerness:     {outerness_map[fid]:.4f}")
        print(f"   Cylindricity:  {cylindricity_map[fid]:.4f}")
        print(f"   Elite:         {is_elite[fid]}")

        plt.render()

    plt.add_callback("mouse click", on_click)
    set_mode("outerness")
    plt.show(mesh, interactive=True)


if __name__ == "__main__":
    run_debug_tool()
