import torch
import torch.nn as nn
import numpy as np
import trimesh
import networkx as nx
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from vedo import Mesh, Plotter, Text2D, Box

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
    def __init__(self, in_dim=13, hidden_dim=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, X): return self.net(X)


# ============================================================
# 2. POST-PROCESSING (Region Growing)
# ============================================================
def smart_expand_selection(initial_mask, prob_map, adjacency, normals, low_thresh=0.40):
    mask = initial_mask.copy()
    for i in range(10):
        prev_count = np.sum(mask)
        border_A = mask[adjacency[:, 0]] & ~mask[adjacency[:, 1]]
        candidates_B = adjacency[border_A, 1]
        source_A = adjacency[border_A, 0]
        border_B = mask[adjacency[:, 1]] & ~mask[adjacency[:, 0]]
        candidates_A = adjacency[border_B, 0]
        source_B = adjacency[border_B, 1]

        sources = np.concatenate([source_A, source_B])
        candidates = np.concatenate([candidates_B, candidates_A])

        if len(candidates) == 0: break

        prob_condition = prob_map[candidates] > low_thresh
        norm_s = normals[sources]
        norm_c = normals[candidates]
        dots = np.einsum('ij,ij->i', norm_s, norm_c)
        flat_condition = dots > 0.98

        mask[candidates[prob_condition & flat_condition]] = True
        if np.sum(mask) == prev_count: break
    return mask


def filter_small_islands(mask, adjacency, face_areas, min_area=50.0):
    """
    Removes clusters smaller than min_area (mm^2).
    """
    selected = np.where(mask)[0]
    if len(selected) == 0: return mask

    subset_mask = mask[adjacency[:, 0]] & mask[adjacency[:, 1]]
    subset_edges = adjacency[subset_mask]

    if len(subset_edges) == 0:
        clean_mask = np.zeros_like(mask, dtype=bool)
        for idx in selected:
            if face_areas[idx] >= min_area:
                clean_mask[idx] = True
        return clean_mask

    G = nx.from_edgelist(subset_edges)
    G.add_nodes_from(selected)

    clean_mask = np.zeros_like(mask, dtype=bool)
    for comp in nx.connected_components(G):
        comp_indices = list(comp)
        cluster_area = np.sum(face_areas[comp_indices])
        if cluster_area >= min_area:
            clean_mask[comp_indices] = True
    return clean_mask


# ============================================================
# 3. ADVANCED CLAMP GENERATION (Z-Locked & Collinear)
# ============================================================
def get_orientation_matrix(forward_vector):
    """
    Creates a 4x4 transform matrix where X-axis points along forward_vector
    and Z-axis is forced UP (World Z).
    """
    x_axis = forward_vector / (np.linalg.norm(forward_vector) + 1e-6)
    world_z = np.array([0.0, 0.0, 1.0])

    # Sideways (Y) - Width of the jaw
    y_axis = np.cross(world_z, x_axis)
    if np.linalg.norm(y_axis) < 1e-3: y_axis = np.array([0.0, 1.0, 0.0])
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Up (Z) - Recalculated to ensure orthogonality
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    R = np.array([x_axis, y_axis, z_axis]).T
    T = np.eye(4)
    T[:3, :3] = R
    return T


def generate_clamp_actors(mesh, mask, adjacency, face_areas, opening_offset):
    """
    1. Finds Top 2 Patches by Area.
    2. Calculates Squeeze Axis from Patch A Normal (Z-flattened).
    3. Calculates Midpoint between patches.
    4. Projects jaws onto the axis passing through Midpoint (Collinear Fix).
    """
    selected_indices = np.where(mask)[0]
    if len(selected_indices) == 0: return []

    # 1. Identify Clusters
    subset_mask = mask[adjacency[:, 0]] & mask[adjacency[:, 1]]
    subset_edges = adjacency[subset_mask]

    patches = []
    if len(subset_edges) > 0:
        G = nx.from_edgelist(subset_edges)
        G.add_nodes_from(selected_indices)

        # Sort by Area
        components = list(nx.connected_components(G))
        comp_stats = []
        for comp in components:
            idx_list = list(comp)
            total_area = np.sum(face_areas[idx_list])
            comp_stats.append((total_area, idx_list))

        comp_stats.sort(key=lambda x: x[0], reverse=True)
        patches = [x[1] for x in comp_stats]
    else:
        patches = [selected_indices]

    centroids = mesh.cell_centers().points
    normals = mesh.cell_normals

    # 2. Logic Branch (Dual Jaw)
    if len(patches) >= 2:
        patch_A = patches[0]
        patch_B = patches[1]

        # Raw Centers
        raw_center_A = np.mean(centroids[patch_A], axis=0)
        raw_center_B = np.mean(centroids[patch_B], axis=0)

        # --- A. SQUEEZE AXIS (Z-Flattened Normal of A) ---
        norm_A = np.mean(normals[patch_A], axis=0)
        norm_A[2] = 0.0  # Force Horizontal
        mag = np.linalg.norm(norm_A)
        if mag < 1e-6:
            norm_A = np.array([1.0, 0.0, 0.0])
        else:
            norm_A /= mag

        # Squeeze Axis points INTO the part (A -> B)
        squeeze_axis = -norm_A

        # --- B. SHARED RAIL (The Collinearity Fix) ---
        # 1. Find the midpoint of the entire assembly
        midpoint = (raw_center_A + raw_center_B) / 2.0

        # 2. Project A and B onto the line defined by [Midpoint, Squeeze_Axis]
        # This removes any lateral (Y) or vertical (Z) misalignment

        # Vector from Midpoint to A
        vec_mid_to_A = raw_center_A - midpoint
        dist_A = np.dot(vec_mid_to_A, squeeze_axis)

        # Vector from Midpoint to B
        vec_mid_to_B = raw_center_B - midpoint
        dist_B = np.dot(vec_mid_to_B, squeeze_axis)

        # Refined (Perfectly Aligned) Centers
        aligned_center_A = midpoint + (dist_A * squeeze_axis)
        aligned_center_B = midpoint + (dist_B * squeeze_axis)

        # --- C. POSITIONS WITH OFFSET ---
        # Jaw A moves AWAY from center (minus squeeze) to open
        pos_A = aligned_center_A - (squeeze_axis * opening_offset)
        # Jaw B moves AWAY from center (plus squeeze) to open
        pos_B = aligned_center_B + (squeeze_axis * opening_offset)

        # Orientations
        # Jaw A faces B (along squeeze)
        T_matrix_A = get_orientation_matrix(squeeze_axis)
        # Jaw B faces A (against squeeze)
        T_matrix_B = get_orientation_matrix(-squeeze_axis)

        configs = [(pos_A, T_matrix_A), (pos_B, T_matrix_B)]

    else:
        # --- SINGLE JAW FALLBACK ---
        patch_A = patches[0]
        center_A = np.mean(centroids[patch_A], axis=0)

        norm_A = np.mean(normals[patch_A], axis=0)
        norm_A[2] = 0.0
        if np.linalg.norm(norm_A) < 1e-6:
            norm_A = np.array([1.0, 0.0, 0.0])
        else:
            norm_A /= np.linalg.norm(norm_A)

        squeeze_axis = -norm_A
        T_matrix = get_orientation_matrix(squeeze_axis)
        pos_A = center_A - (squeeze_axis * opening_offset)

        configs = [(pos_A, T_matrix)]

    # 3. Create Actors
    actors = []
    jaw_width = 80.0
    jaw_height = 25.0
    jaw_thickness = 10.0

    for pos, transform in configs:
        j = Box(pos=(0, 0, 0), length=jaw_thickness, width=jaw_width, height=jaw_height)
        j.apply_transform(transform)
        j.pos(pos)
        j.c("grey").alpha(0.8).linecolor("black")
        actors.append(j)

    return actors


# ============================================================
# 4. VISUALIZATION
# ============================================================
def visualize_inference(points, tris, probs, control_state):
    print("Building mesh & graph...")
    mesh = Mesh([points, tris])

    tm = trimesh.Trimesh(vertices=points, faces=tris, process=False)
    adjacency = tm.face_adjacency
    normals = tm.face_normals
    face_areas = tm.area_faces

    mesh.celldata["Clamp_Probability"] = probs[:, 0]

    plt = Plotter(title="AUTO CAM - Clamp Visualizer", bg="white", axes=1)

    state = {
        "threshold": 0.90,
        "post_process": True,
        "show_clamp": False,
        "offset": 0.0,
        "clamp_actors": []
    }

    txt = Text2D("", pos="bottom-left")
    plt.add(txt)

    def update_view():
        base_mask = probs[:, 0] > state["threshold"]
        if state["post_process"]:
            mask = smart_expand_selection(base_mask, probs[:, 0], adjacency, normals)
            mask = filter_small_islands(mask, adjacency, face_areas)
        else:
            mask = base_mask

        cols = np.full((mesh.ncells, 4), [220, 220, 220, 50], dtype=np.uint8)
        cols[mask] = [255, 0, 0, 255]
        mesh.cellcolors = cols

        plt.remove(state["clamp_actors"])
        state["clamp_actors"] = []

        if state["show_clamp"]:
            jaws = generate_clamp_actors(mesh, mask, adjacency, face_areas, state["offset"])
            state["clamp_actors"] = jaws
            plt.add(jaws)

        txt.text(f"Offset: {state['offset']:.1f} mm")
        plt.render()

    def slide_thresh(w, e):
        state["threshold"] = w.GetRepresentation().GetValue(); update_view()

    def slide_offset(w, e):
        state["offset"] = w.GetRepresentation().GetValue(); update_view()

    def btn_smart(*args):
        state["post_process"] = not state["post_process"]; update_view()

    def btn_clamp(*args):
        state["show_clamp"] = not state["show_clamp"]; update_view()

    def btn_load_next(*args):
        control_state["load_next"] = True; plt.close()

    plt.add_slider(slide_thresh, 0.5, 0.99, value=0.90, pos=[(0.1, 0.05), (0.3, 0.05)], title="Confidence")
    plt.add_slider(slide_offset, 0.0, 100.0, value=10.0, pos=[(0.4, 0.05), (0.6, 0.05)], title="Jaw Distance (mm)")
    plt.add_button(btn_smart, states=[" Smart Fill: ON ", " Smart Fill: OFF"], c=["w", "w"], bc=["g", "r"],
                   pos=(0.8, 0.09), size=20)
    plt.add_button(btn_clamp, states=[" Show Clamp ", " Hide Clamp "], c=["w", "w"], bc=["b", "grey"], pos=(0.8, 0.04),
                   size=20)
    plt.add_button(btn_load_next, states=[" LOAD NEW FILE "], c=["white"], bc=["orange"], pos=(0.5, 0.95), size=25,
                   font="courier")

    update_view()
    plt.show(mesh, interactive=True)


if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    model_path = askopenfilename(title="Select Model (.pth)", filetypes=[("Model", "*.pth")])
    if not model_path: exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path}...")
    model = ClampSupportNet(in_dim=13).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        exit()
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

            if control_state["load_next"]:
                continue
            else:
                break

        except Exception as e:
            print(f"❌ Critical Error: {e}")
            import traceback

            traceback.print_exc()
            break