import numpy as np
import trimesh
from vedo import Mesh, Plotter, Text2D, Points, Line
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ----------------- IMPORT SHARED FEATURE EXTRACTOR -----------------
from features import (
    compute_outerness,
    compute_edge_ratio,
    compute_curvature,
    compute_slenderness,
    compute_thickness_ratio
)


def run_debug_tool():
    # 1. Select file
    Tk().withdraw()
    path = askopenfilename(title="Select STL to Debug", filetypes=[("STL", ".stl")])
    if not path:
        return

    print(f"Processing {path}...")
    mesh = Mesh(path)
    mesh.triangulate()

    centroids = mesh.cell_centers().points
    normals = mesh.cell_normals

    # ----------------- COMPUTE ALL FEATURES -----------------
    print("✔ Computing outerness…")
    outerness = compute_outerness(mesh, centroids)

    print("✔ Computing edge ratio…")
    edge_ratio = compute_edge_ratio(mesh)

    print("✔ Computing curvature…")
    curvature = compute_curvature(mesh)

    print("✔ Computing slenderness…")
    slenderness = compute_slenderness(mesh)

    print("✔ Computing thickness ratio…")
    thickness = compute_thickness_ratio(mesh)

    # Elite-class candidate mask (same logic as training)
    verticality = 1.0 - np.abs(normals[:, 2])
    is_elite = (verticality > 0.8) & (outerness > 0.85)

    # --------------------------------------------------------------------
    # UI + Visualization
    # --------------------------------------------------------------------
    plt = Plotter(title="Fragility Feature Debugger", bg="blackboard", axes=1)

    info_txt = Text2D(
        "Click a face to inspect rays.\nUse buttons to view feature maps.",
        pos="bottom-left",
        s=0.9,
        c="gray"
    )
    plt.add(info_txt)

    state = {"mode": "outerness"}

    # --------------------------------------------------------------------
    # Helper to change visual mode
    # --------------------------------------------------------------------
    def set_mode(mode):
        state["mode"] = mode
        mesh.c("white")  # reset base color

        try:
            plt.remove("Legend")
        except:
            pass

        if mode == "outerness":
            mesh.cmap("jet", outerness, vmin=0, vmax=1, on='cells')
            mesh.add_scalarbar(title="Outerness")

        elif mode == "edge":
            mesh.cmap("jet", edge_ratio, on='cells')
            mesh.add_scalarbar(title="Edge Ratio")

        elif mode == "curvature":
            mesh.cmap("jet", curvature, on='cells')
            mesh.add_scalarbar(title="Curvature")

        elif mode == "slenderness":
            mesh.cmap("jet", slenderness, on='cells')
            mesh.add_scalarbar(title="Slenderness")

        elif mode == "thickness":
            mesh.cmap("jet", thickness, on='cells')
            mesh.add_scalarbar(title="Thickness Ratio")

        elif mode == "mask":
            cols = np.full((mesh.ncells, 3), [70, 70, 70], dtype=np.uint8)
            cols[is_elite] = [0, 255, 0]
            mesh.cellcolors = cols

        plt.render()

    # ----------------- BUTTONS -----------------
    plt.add_button(lambda *_: set_mode("outerness"), ["Outerness"], c="white", bc="red", pos=(0.20, 0.05))
    plt.add_button(lambda *_: set_mode("edge"), ["Edge Ratio"], c="white", bc="blue", pos=(0.35, 0.05))
    plt.add_button(lambda *_: set_mode("curvature"), ["Curvature"], c="white", bc="purple", pos=(0.50, 0.05))
    plt.add_button(lambda *_: set_mode("slenderness"), ["Slenderness"], c="white", bc="orange", pos=(0.65, 0.05))
    plt.add_button(lambda *_: set_mode("thickness"), ["Thickness"], c="white", bc="green", pos=(0.80, 0.05))
    plt.add_button(lambda *_: set_mode("mask"), ["Elite Mask"], c="black", bc="lime", pos=(0.95, 0.05))

    # --------------------------------------------------------------------
    # Clicking a face = show ray behavior & print properties
    # --------------------------------------------------------------------
    tm = trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells, process=False)

    def on_click(evt):
        """
        Robust click handler for debugging.
        Click a mesh face to inspect rays and print feature values.
        """
        # Ensure debug_actors exists
        if "debug_actors" not in state:
            state["debug_actors"] = []

        # 1. Make sure we clicked a valid actor
        if not evt.actor:
            return

        # 2. Check if a 3D point was picked
        if evt.picked3d is None:
            print("⚠ Click missed mesh (picked3d=None).")
            return

        # 3. Get closest face
        res = mesh.closest_point(evt.picked3d, return_cell_id=True)

        # 4. Robustly extract face index
        fid = None
        if res is None:
            print("⚠ Click missed mesh or no valid face found.")
            return
        elif isinstance(res, int):
            fid = res
        elif isinstance(res, (list, tuple, np.ndarray)):
            if len(res) == 0:
                print("⚠ Click missed mesh or no valid face found.")
                return
            fid = res[-1]
        else:
            print("⚠ Unexpected type returned from closest_point:", type(res))
            return

        # 5. Print all feature values for that face
        print("\n" + "=" * 40)
        print(f"🔎 Inspecting Face {fid}")
        print(f"• Outerness:        {outerness[fid]:.4f}")
        print(f"• Edge Ratio:       {edge_ratio[fid]:.4f}")
        print(f"• Curvature:        {curvature[fid]:.4f}")
        print(f"• Slenderness:      {slenderness[fid]:.4f}")
        print(f"• Thickness Ratio:  {thickness[fid]:.4f}")

        # 6. Skip ray tracing if face is not elite
        if not is_elite[fid]:
            print("⚠ This face is NOT in elite mask → no ray tracing")
            return

        # 7. Ray origins for visualization
        c = centroids[fid]
        n = normals[fid]

        world_z = np.array([0, 0, 1])
        tan = np.cross(n, world_z)
        tan /= np.linalg.norm(tan) + 1e-6
        bitan = np.cross(n, tan)
        offset = 3.0

        origins = [
            c,
            c + tan * offset,
            c - tan * offset,
            c + bitan * offset,
            c - bitan * offset
        ]

        labels = ["Center", "Right", "Left", "Up", "Down"]

        # 8. Remove previous debug actors safely
        if state.get("debug_actors"):
            plt.remove(state["debug_actors"])
        state["debug_actors"] = []

        # 9. Shoot rays and visualize hits
        for i, o in enumerate(origins):
            direction = -n

            try:
                idx_tri, _, locs = tm.ray.intersects_id(
                    ray_origins=[o - n * 0.01],
                    ray_directions=[direction],
                    multiple_hits=True,
                    return_locations=True,
                    max_d=500.0
                )
            except Exception as e:
                print(f"⚠ Ray cast failed for origin {labels[i]}: {e}")
                continue

            if len(idx_tri) == 0:
                vis_end = o + direction * 80.0
                ln = Line(o, vis_end, c="red", lw=1, alpha=0.3)
                state["debug_actors"].append(ln)
                print(f" → [{labels[i]}] MISS")
                continue

            # Take first hit for visualization
            hit = locs[0]
            ln = Line(o, hit, c="green", lw=2)
            pt = Points([hit], r=10, c="yellow")
            state["debug_actors"].extend([ln, pt])
            print(f" → [{labels[i]}] VALID HIT")

        # 10. Render updates
        plt.add(state["debug_actors"])
        plt.render()

    plt.add_callback("mouse click", on_click)

    set_mode("outerness")
    plt.show(mesh, interactive=True)


if __name__ == "__main__":
    run_debug_tool()
