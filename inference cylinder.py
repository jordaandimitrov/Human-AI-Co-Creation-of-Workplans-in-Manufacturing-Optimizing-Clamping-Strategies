import torch
import torch.nn as nn
import numpy as np
import trimesh
import networkx as nx
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from vedo import Mesh, Plotter, Text2D, Box, Cylinder, merge

# IMPORT SHARED FEATURES
try:
    import features
except ImportError:
    print("❌ ERROR: 'features.py' not found.")
    exit()


# ============================================================
# 1. MODEL DEFINITION
# ============================================================
class ClampSupportNet(nn.Module):
    # Ensure in_dim matches exactly what your model was trained with (15 or 16)
    def __init__(self, in_dim=15, hidden_dim=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, X): return self.net(X)


# ============================================================
# 2. V-BLOCK GENERATION & KINEMATICS
# ============================================================
def generate_vblock_actors(mesh, mask, adjacency, face_areas, pull_offset, z_shift, vblock_len, screw_travel,
                           screw_slide, min_area=10.0):
    """
    Finds the clamping strips and generates a V-block, a sliding U-bracket, and a movable screw.
    """
    selected = np.where(mask)[0]
    if len(selected) == 0: return [], []

    subset_mask = mask[adjacency[:, 0]] & mask[adjacency[:, 1]]
    subset_edges = adjacency[subset_mask]

    G = nx.Graph()
    G.add_nodes_from(selected)
    if len(subset_edges) > 0:
        G.add_edges_from(subset_edges)

    comp_stats = []
    for comp in nx.connected_components(G):
        idx_list = list(comp)
        area = np.sum(face_areas[idx_list])
        if area >= min_area:
            comp_stats.append((area, idx_list))

    comp_stats.sort(key=lambda x: x[0], reverse=True)

    if len(comp_stats) < 2:
        return [], []

    patch_A = comp_stats[0][1]
    patch_B = comp_stats[1][1]
    active_indices = np.concatenate([patch_A, patch_B])

    centroids = mesh.cell_centers().points
    normals = mesh.cell_normals

    cA = np.mean(centroids[patch_A], axis=0)
    cB = np.mean(centroids[patch_B], axis=0)
    nA = np.mean(normals[patch_A], axis=0)
    nB = np.mean(normals[patch_B], axis=0)

    n_avg = nA + nB
    n_avg[2] = 0.0
    if np.linalg.norm(n_avg) < 1e-6:
        n_avg = np.array([0.0, 1.0, 0.0])
    else:
        n_avg = n_avg / np.linalg.norm(n_avg)

    approach_vec = -n_avg
    midpoint = (cA + cB) / 2.0

    vy = approach_vec
    vz = np.array([0.0, 0.0, 1.0])
    vx = np.cross(vy, vz)

    strip_dist = np.linalg.norm(cA - cB)

    # Dynamic Sizing Math
    notch_w = max(50.0, strip_dist * 1.5)
    notch_d = notch_w / 2.0
    width = notch_w + 40.0
    thickness = notch_d + 30.0

    geometric_shift = notch_d - (strip_dist / 2.0)
    final_pos = midpoint + (vy * geometric_shift) - (vy * pull_offset) + (vz * z_shift)

    # --- 1. BUILD THE V-BLOCK MESH ---
    pts = [
        [-width / 2, -thickness, -vblock_len / 2], [width / 2, -thickness, -vblock_len / 2],
        [width / 2, 0, -vblock_len / 2], [notch_w / 2, 0, -vblock_len / 2],
        [0, -notch_d, -vblock_len / 2], [-notch_w / 2, 0, -vblock_len / 2],
        [-width / 2, 0, -vblock_len / 2], [-width / 2, -thickness, vblock_len / 2],
        [width / 2, -thickness, vblock_len / 2], [width / 2, 0, vblock_len / 2],
        [notch_w / 2, 0, vblock_len / 2], [0, -notch_d, vblock_len / 2],
        [-notch_w / 2, 0, vblock_len / 2], [-width / 2, 0, vblock_len / 2],
    ]

    faces = [
        [0, 4, 1], [1, 4, 3], [1, 3, 2], [0, 5, 4], [0, 6, 5],
        [7, 8, 11], [8, 10, 11], [8, 9, 10], [7, 11, 12], [7, 12, 13],
        [0, 7, 8, 1], [1, 8, 9, 2], [2, 9, 10, 3], [3, 10, 11, 4],
        [4, 11, 12, 5], [5, 12, 13, 6], [6, 13, 7, 0]
    ]

    vblock = Mesh([pts, faces])
    vblock.compute_normals()
    vblock.c("darkgrey").alpha(0.6).linecolor("black")

    # --- 2. BUILD THE SLIDING U-BRACKET ---
    bracket_h = notch_w + 50.0
    pillar_w = 18.0
    pillar_depth = 30.0
    pillar_h = thickness + bracket_h
    pillar_y_center = (-thickness + bracket_h) / 2.0

    # Apply the screw_slide offset to the local Z coordinates
    lp = Box(pos=(-width / 2 - pillar_w / 2, pillar_y_center, screw_slide), length=pillar_w, width=pillar_h,
             height=pillar_depth)
    rp = Box(pos=(width / 2 + pillar_w / 2, pillar_y_center, screw_slide), length=pillar_w, width=pillar_h,
             height=pillar_depth)
    top_bar = Box(pos=(0, bracket_h - pillar_w / 2, screw_slide), length=width + 2 * pillar_w, width=pillar_w,
                  height=pillar_depth)

    bracket = merge([lp, rp, top_bar])
    bracket.c("#444444").alpha(0.8).linecolor("black")

    # --- 3. BUILD THE CLAMP SCREW ---
    screw_len = bracket_h - 10.0
    tip_y = bracket_h - pillar_w - screw_travel

    # Apply the same screw_slide offset to the screw components
    shaft = Cylinder(pos=(0, tip_y + screw_len / 2, screw_slide), r=4, height=screw_len, axis=(0, 1, 0))
    knob = Cylinder(pos=(0, tip_y + screw_len + 10, screw_slide), r=12, height=20, axis=(0, 1, 0))
    pad = Cylinder(pos=(0, tip_y, screw_slide), r=10, height=4, axis=(0, 1, 0))

    screw = merge([shaft, knob, pad])
    screw.c("silver").linecolor("black")

    # --- 4. ASSEMBLE & ORIENT EVERYTHING ---
    actors = [vblock, bracket, screw]

    R = np.array([vx, vy, vz]).T
    T = np.eye(4)
    T[:3, :3] = R

    for actor in actors:
        actor.apply_transform(T)
        actor.pos(final_pos)

    return actors, active_indices


# ============================================================
# 3. VISUALIZATION CONTROLLER
# ============================================================
def visualize_inference(points, tris, probs, control_state):
    mesh = Mesh([points, tris])
    tm = trimesh.Trimesh(vertices=points, faces=tris, process=False)
    adjacency = tm.face_adjacency
    face_areas = tm.area_faces

    plt = Plotter(title="AUTO CAM - V-Block Visualizer", bg="white", axes=0)

    # UI State Dictionary
    state = {
        "threshold": 0.80,
        "show_clamp": True,
        "pull_offset": 0.0,
        "z_shift": 0.0,
        "vblock_length": 80.0,
        "screw_travel": 35.0,
        "screw_slide": 0.0,  # NEW SLIDER STATE
        "clamp_actors": []
    }

    txt_info = Text2D("", pos="bottom-left")
    plt.add(txt_info)

    def update_view():
        mask = probs[:, 0] > state["threshold"]

        cols = np.full((mesh.ncells, 4), [150, 150, 150, 255], dtype=np.uint8)
        cols[mask] = [255, 0, 0, 255]

        plt.remove(state["clamp_actors"])
        state["clamp_actors"] = []
        active_indices = []
        status_msg = f"Offset: {state['pull_offset']:.1f} | Z-Shift: {state['z_shift']:.1f} | Screw Drop: {state['screw_travel']:.1f} | Bracket Slide: {state['screw_slide']:.1f}"

        if state["show_clamp"]:
            actors, active_indices = generate_vblock_actors(
                mesh, mask, adjacency, face_areas,
                state["pull_offset"], state["z_shift"], state["vblock_length"], state["screw_travel"],
                state["screw_slide"]
            )

            if len(actors) == 0:
                status_msg += " | ⚠️ Found < 2 clamping strips (Adjust Threshold)"
            else:
                state["clamp_actors"] = actors
                plt.add(actors)

        if len(active_indices) > 0:
            cols[active_indices] = [0, 255, 0, 255]

        mesh.cellcolors = cols
        txt_info.text(status_msg)
        plt.render()

    # SLIDER CALLBACKS
    def slide_thresh(w, e):
        state["threshold"] = w.GetRepresentation().GetValue(); update_view()

    def slide_pull(w, e):
        state["pull_offset"] = w.GetRepresentation().GetValue(); update_view()

    def slide_z(w, e):
        state["z_shift"] = w.GetRepresentation().GetValue(); update_view()

    def slide_length(w, e):
        state["vblock_length"] = w.GetRepresentation().GetValue(); update_view()

    def slide_screw(w, e):
        state["screw_travel"] = w.GetRepresentation().GetValue(); update_view()

    def slide_bracket(w, e):
        state["screw_slide"] = w.GetRepresentation().GetValue(); update_view()

    # BUTTON CALLBACKS
    def btn_clamp(*args):
        state["show_clamp"] = not state["show_clamp"]; update_view()

    def btn_load_next(*args):
        control_state["load_next"] = True; plt.close()

    # UI LAYOUT (Arranged into two rows)
    # Row 1: Main Block Positioning
    plt.add_slider(slide_thresh, 0.1, 0.99, value=0.80, pos=[(0.05, 0.05), (0.22, 0.05)], title="Confidence")
    plt.add_slider(slide_pull, 0.0, 50.0, value=0.0, pos=[(0.28, 0.05), (0.45, 0.05)], title="Pull Offset")
    plt.add_slider(slide_z, -100.0, 100.0, value=0.0, pos=[(0.51, 0.05), (0.68, 0.05)], title="Block Z-Shift")
    plt.add_slider(slide_length, 20.0, 200.0, value=80.0, pos=[(0.74, 0.05), (0.95, 0.05)], title="Block Length")

    # Row 2: U-Bracket Kinematics
    plt.add_slider(slide_screw, 0.0, 100.0, value=35.0, pos=[(0.05, 0.12), (0.22, 0.12)], title="Screw Down/Up (mm)")
    plt.add_slider(slide_bracket, -100.0, 100.0, value=0.0, pos=[(0.28, 0.12), (0.45, 0.12)],
                   title="Bracket Slide (mm)")

    plt.add_button(btn_clamp, states=[" Show V-Block ", " Hide V-Block "], c=["w", "w"], bc=["b", "grey"],
                   pos=(0.8, 0.15), size=20)
    plt.add_button(btn_load_next, states=[" LOAD NEW FILE "], c=["white"], bc=["orange"], pos=(0.5, 0.95), size=25,
                   font="courier")

    update_view()
    plt.show(mesh, interactive=True)


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    print("Select your trained PyTorch Model...")
    model_path = askopenfilename(title="Select Model (.pth)", filetypes=[("Model", "*.pth")])
    if not model_path: exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClampSupportNet(in_dim=15).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    while True:
        stl_path = askopenfilename(title="Select Test STL", filetypes=[("STL", ".stl")])
        if not stl_path: break
        print(f"\nProcessing: {stl_path}...")
        try:
            feats, tris, points = features.extract_triangle_features(stl_path)
            input_tensor = torch.tensor(feats, dtype=torch.float32).to(device)

            with torch.no_grad():
                probs = torch.sigmoid(model(input_tensor)).cpu().numpy()

            control_state = {"load_next": False}
            visualize_inference(points, tris, probs, control_state)

            if not control_state["load_next"]:
                break

        except Exception as e:
            print(f"❌ Critical Error: {e}")
            import traceback

            traceback.print_exc()
            break