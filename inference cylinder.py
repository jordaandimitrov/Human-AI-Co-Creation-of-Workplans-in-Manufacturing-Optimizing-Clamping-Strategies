import torch
import torch.nn as nn
import numpy as np
import trimesh
import networkx as nx
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from vedo import Mesh, Plotter, Text2D

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
    # Note: Set out_dim=1 to match your BCEWithLogitsLoss binary training!
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
def generate_vblock_actors(mesh, mask, adjacency, face_areas, pull_offset, z_shift, vblock_len, min_area=10.0):
    """
    Finds the two predicted clamping strips, calculates the bisecting approach vector,
    and generates a custom 90-degree V-block mesh snapped perfectly to the part.
    """
    # 1. Extract Connected Islands (The Strips)
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

    # We need at least 2 strips to place a V-block
    if len(comp_stats) < 2:
        return [], []

    patch_A = comp_stats[0][1]
    patch_B = comp_stats[1][1]
    active_indices = np.concatenate([patch_A, patch_B])

    # 2. Kinematics & Alignment Math
    centroids = mesh.cell_centers().points
    normals = mesh.cell_normals

    cA = np.mean(centroids[patch_A], axis=0)
    cB = np.mean(centroids[patch_B], axis=0)
    nA = np.mean(normals[patch_A], axis=0)
    nB = np.mean(normals[patch_B], axis=0)

    # The V-block pushes INTO the part, so the approach vector is opposite the outward normals
    n_avg = nA + nB
    n_avg[2] = 0.0  # Force it to stay perfectly horizontal

    if np.linalg.norm(n_avg) < 1e-6:
        n_avg = np.array([0.0, 1.0, 0.0])
    else:
        n_avg = n_avg / np.linalg.norm(n_avg)

    approach_vec = -n_avg
    midpoint = (cA + cB) / 2.0

    # Build the Local Coordinate System for the V-block
    vy = approach_vec
    vz = np.array([0.0, 0.0, 1.0])
    vx = np.cross(vy, vz)

    # Calculate exactly how far apart the two clamp strips are
    strip_dist = np.linalg.norm(cA - cB)

    # 3. Construct the Custom 90° V-Block Mesh (Dynamic Sizing)
    # We size the V-block so the notch is 50% wider than the clamp distance
    notch_w = max(50.0, strip_dist * 1.5)
    notch_d = notch_w / 2.0  # Depth must be exactly half the width for a 90° angle
    width = notch_w + 40.0
    thickness = notch_d + 30.0

    # 4. Perfect Alignment Math
    # Since it's a 90° V-block, the shift required to make the faces touch the strips
    # is exactly the notch depth minus half the strip distance.
    geometric_shift = notch_d - (strip_dist / 2.0)

    # Apply Slider Offsets + Geometric Shift
    final_pos = midpoint + (vy * geometric_shift) - (vy * pull_offset) + (vz * z_shift)

    # Vertices of the V-block profile
    pts = [
        [-width / 2, -thickness, -vblock_len / 2],  # 0: Back-Bottom-Left
        [width / 2, -thickness, -vblock_len / 2],  # 1: Back-Bottom-Right
        [width / 2, 0, -vblock_len / 2],  # 2: Back-Top-Right
        [notch_w / 2, 0, -vblock_len / 2],  # 3: Back-Notch-Right
        [0, -notch_d, -vblock_len / 2],  # 4: Back-Notch-Center (The V)
        [-notch_w / 2, 0, -vblock_len / 2],  # 5: Back-Notch-Left
        [-width / 2, 0, -vblock_len / 2],  # 6: Back-Top-Left

        [-width / 2, -thickness, vblock_len / 2],  # 7: Front-Bottom-Left
        [width / 2, -thickness, vblock_len / 2],  # 8: Front-Bottom-Right
        [width / 2, 0, vblock_len / 2],  # 9: Front-Top-Right
        [notch_w / 2, 0, vblock_len / 2],  # 10: Front-Notch-Right
        [0, -notch_d, vblock_len / 2],  # 11: Front-Notch-Center
        [-notch_w / 2, 0, vblock_len / 2],  # 12: Front-Notch-Left
        [-width / 2, 0, vblock_len / 2],  # 13: Front-Top-Left
    ]

    # Polygon mappings (Manually triangulated to fix the "closed off" rendering bug)
    faces = [
        # Back Cap (Triangulated into 5 pieces)
        [0, 4, 1], [1, 4, 3], [1, 3, 2], [0, 5, 4], [0, 6, 5],

        # Front Cap (Triangulated into 5 pieces)
        [7, 8, 11], [8, 10, 11], [8, 9, 10], [7, 11, 12], [7, 12, 13],

        # Outer Walls and V-Notch (Quads)
        [0, 7, 8, 1],  # Bottom Face
        [1, 8, 9, 2],  # Right Outer Face
        [2, 9, 10, 3],  # Top Right Flat
        [3, 10, 11, 4],  # Right V-Notch Face
        [4, 11, 12, 5],  # Left V-Notch Face
        [5, 12, 13, 6],  # Top Left Flat
        [6, 13, 7, 0]  # Left Outer Face
    ]

    vblock = Mesh([pts, faces])
    vblock.compute_normals()

    # 5. Updated V-Block Visuals (Transparent Dark Grey)
    vblock.c("darkgrey").alpha(0.6).linecolor("black")

    # 6. Orient and Place the V-block
    R = np.array([vx, vy, vz]).T
    T = np.eye(4)
    T[:3, :3] = R
    vblock.apply_transform(T)
    vblock.pos(final_pos)

    return [vblock], active_indices


# ============================================================
# 3. VISUALIZATION CONTROLLER
# ============================================================
def visualize_inference(points, tris, probs, control_state):
    mesh = Mesh([points, tris])
    tm = trimesh.Trimesh(vertices=points, faces=tris, process=False)
    adjacency = tm.face_adjacency
    face_areas = tm.area_faces

    # Removed axes
    plt = Plotter(title="AUTO CAM - V-Block Visualizer", bg="white", axes=0)

    # UI State Dictionary
    state = {
        "threshold": 0.80,
        "show_clamp": True,
        "pull_offset": 5.0,
        "z_shift": 0.0,
        "vblock_length": 80.0,
        "clamp_actors": []
    }

    txt_info = Text2D("", pos="bottom-left")
    plt.add(txt_info)

    def update_view():
        mask = probs[:, 0] > state["threshold"]

        # 1. Updated Part Visuals (Solid Industrial Grey)
        cols = np.full((mesh.ncells, 4), [150, 150, 150, 255], dtype=np.uint8)
        cols[mask] = [255, 0, 0, 255]  # Red highlight for clamp zones

        plt.remove(state["clamp_actors"])
        state["clamp_actors"] = []
        active_indices = []
        status_msg = f"V-Block Offset: {state['pull_offset']:.1f}mm | Z-Shift: {state['z_shift']:.1f}mm"

        if state["show_clamp"]:
            vblock_actors, active_indices = generate_vblock_actors(
                mesh, mask, adjacency, face_areas,
                state["pull_offset"], state["z_shift"], state["vblock_length"]
            )

            if len(vblock_actors) == 0:
                status_msg += " | ⚠️ Found < 2 clamping strips (Adjust Threshold)"
            else:
                state["clamp_actors"] = vblock_actors
                plt.add(vblock_actors)

        # Highlight the two strips the V-block is actually touching in Green
        if len(active_indices) > 0:
            cols[active_indices] = [0, 255, 0, 255]

        mesh.cellcolors = cols
        txt_info.text(status_msg)
        plt.render()

    # SLIDER CALLBACKS
    def slide_thresh(w, e):
        state["threshold"] = w.GetRepresentation().GetValue();
        update_view()

    def slide_pull(w, e):
        state["pull_offset"] = w.GetRepresentation().GetValue();
        update_view()

    def slide_z(w, e):
        state["z_shift"] = w.GetRepresentation().GetValue();
        update_view()

    def slide_length(w, e):
        state["vblock_length"] = w.GetRepresentation().GetValue();
        update_view()

    # BUTTON CALLBACKS
    def btn_clamp(*args):
        state["show_clamp"] = not state["show_clamp"];
        update_view()

    def btn_load_next(*args):
        control_state["load_next"] = True;
        plt.close()

    # UI LAYOUT
    plt.add_slider(slide_thresh, 0.1, 0.99, value=0.80, pos=[(0.05, 0.05), (0.25, 0.05)], title="Confidence")
    plt.add_slider(slide_pull, 0.0, 50.0, value=0.0, pos=[(0.3, 0.05), (0.5, 0.05)], title="Pull Offset (mm)")
    plt.add_slider(slide_z, -100.0, 100.0, value=0.0, pos=[(0.55, 0.05), (0.75, 0.05)], title="Z-Shift (Up/Down)")
    plt.add_slider(slide_length, 20.0, 200.0, value=80.0, pos=[(0.8, 0.05), (0.95, 0.05)], title="V-Block Length")

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
    model = ClampSupportNet(in_dim=15).to(device)  # Make sure in_dim matches your features!
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