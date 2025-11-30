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


def filter_small_islands(mask, adjacency, min_faces=50):
    selected = np.where(mask)[0]
    if len(selected) == 0: return mask

    subset_mask = mask[adjacency[:, 0]] & mask[adjacency[:, 1]]
    subset_edges = adjacency[subset_mask]
    if len(subset_edges) == 0: return np.zeros_like(mask, dtype=bool)

    G = nx.from_edgelist(subset_edges)
    G.add_nodes_from(selected)

    clean_mask = np.zeros_like(mask, dtype=bool)
    for comp in nx.connected_components(G):
        if len(comp) >= min_faces:
            clean_mask[list(comp)] = True
    return clean_mask


# ============================================================
# 3. ADVANCED CLAMP GENERATION (Dual-Patch Sync)
# ============================================================
def get_orientation_matrix(forward_vector):
    """
    Creates a 4x4 transform matrix where X-axis points along forward_vector
    and Z-axis is forced UP (World Z).
    """
    x_axis = forward_vector / (np.linalg.norm(forward_vector) + 1e-6)
    world_z = np.array([0.0, 0.0, 1.0])

    # Sideways (Y)
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


def generate_clamp_actors(mesh, mask, adjacency, opening_offset):
    """
    Identifies the 2 largest patches, glues jaws to them,
    and aligns them along the axis connecting their centers.
    """
    selected_indices = np.where(mask)[0]
    if len(selected_indices) == 0: return []

    # 1. Identify Clusters (Patches)
    subset_mask = mask[adjacency[:, 0]] & mask[adjacency[:, 1]]
    subset_edges = adjacency[subset_mask]

    patches = []
    if len(subset_edges) > 0:
        G = nx.from_edgelist(subset_edges)
        G.add_nodes_from(selected_indices)
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        patches = [list(c) for c in components]
    else:
        patches = [selected_indices]

    centroids = mesh.cell_centers().points

    # 2. Logic Branch
    if len(patches) >= 2:
        # --- DUAL JAW MODE (Standard) ---
        patch_A = patches[0]
        patch_B = patches[1]

        center_A = np.mean(centroids[patch_A], axis=0)
        center_B = np.mean(centroids[patch_B], axis=0)

        # Calculate Shared Squeeze Axis
        axis_vector = center_A - center_B
        dist = np.linalg.norm(axis_vector)

        # SAFETY CHECK 1: If centers are identical (rare)
        if dist < 1e-3:
            axis_vector = np.array([1.0, 0.0, 0.0])
        else:
            axis_vector = axis_vector / dist

        # Determine shared Z-height
        shared_z = (center_A[2] + center_B[2]) / 2.0
        center_A[2] = shared_z
        center_B[2] = shared_z

        T_matrix_A = get_orientation_matrix(-axis_vector)
        T_matrix_B = get_orientation_matrix(axis_vector)

        pos_A = center_A + (axis_vector * opening_offset)
        pos_B = center_B - (axis_vector * opening_offset)

        configs = [(pos_A, T_matrix_A), (pos_B, T_matrix_B)]

    else:
        # --- SINGLE JAW FALLBACK (Only 1 patch found) ---
        patch_A = patches[0]
        center_A = np.mean(centroids[patch_A], axis=0)
        normals_A = mesh.cell_normals[patch_A]

        avg_normal = np.mean(normals_A, axis=0)
        mag = np.linalg.norm(avg_normal)

        # SAFETY CHECK 2: THE FIX FOR YOUR CRASH
        if mag < 1e-6:
            # Normals cancelled out (Symmetric selection). Default to X.
            avg_normal = np.array([1.0, 0.0, 0.0])
        else:
            avg_normal /= mag

        T_matrix = get_orientation_matrix(-avg_normal)
        pos_A = center_A + (avg_normal * opening_offset)

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

    mesh.celldata["Clamp_Probability"] = probs[:, 0]

    # Create Plotter
    plt = Plotter(title="AUTO CAM - Clamp Visualizer", bg="white", axes=1)

    state = {
        "threshold": 0.90,
        "post_process": True,
        "show_clamp": False,
        "offset": 0.0,  # 0 means "Glued to surface"
        "clamp_actors": []
    }

    txt = Text2D("", pos="bottom-left")
    plt.add(txt)

    def update_view():
        # 1. Selection
        base_mask = probs[:, 0] > state["threshold"]
        if state["post_process"]:
            mask = smart_expand_selection(base_mask, probs[:, 0], adjacency, normals)
            mask = filter_small_islands(mask, adjacency)
        else:
            mask = base_mask

        # 2. Colors
        cols = np.full((mesh.ncells, 4), [220, 220, 220, 50], dtype=np.uint8)
        cols[mask] = [255, 0, 0, 255]
        mesh.cellcolors = cols

        # 3. Clamps
        plt.remove(state["clamp_actors"])
        state["clamp_actors"] = []

        if state["show_clamp"]:
            jaws = generate_clamp_actors(mesh, mask, adjacency, state["offset"])
            state["clamp_actors"] = jaws
            plt.add(jaws)

        txt.text(f"Offset: {state['offset']:.1f} mm")
        plt.render()

    # --- CONTROLS ---
    def slide_thresh(w, e):
        state["threshold"] = w.GetRepresentation().GetValue();
        update_view()

    def slide_offset(w, e):
        state["offset"] = w.GetRepresentation().GetValue();
        update_view()

    def btn_smart(*args):
        state["post_process"] = not state["post_process"];
        update_view()

    def btn_clamp(*args):
        state["show_clamp"] = not state["show_clamp"];
        update_view()

    def btn_load_next(*args):
        # Set the flag to true and close the window
        control_state["load_next"] = True
        plt.close()

    plt.add_slider(slide_thresh, 0.5, 0.99, value=0.90, pos=[(0.1, 0.05), (0.3, 0.05)], title="Confidence")
    plt.add_slider(slide_offset, 0.0, 100.0, value=10.0, pos=[(0.4, 0.05), (0.6, 0.05)], title="Jaw Distance (mm)")

    plt.add_button(btn_smart, states=[" Smart Fill: ON ", " Smart Fill: OFF"], c=["w", "w"], bc=["g", "r"],
                   pos=(0.8, 0.09), size=20)
    plt.add_button(btn_clamp, states=[" Show Clamp ", " Hide Clamp "], c=["w", "w"], bc=["b", "grey"], pos=(0.8, 0.04),
                   size=20)

    # --- NEW BUTTON FOR LOADING NEXT FILE ---
    plt.add_button(btn_load_next, states=[" LOAD NEW FILE "], c=["white"], bc=["orange"],
                   pos=(0.5, 0.95), size=25, font="courier")

    update_view()
    plt.show(mesh, interactive=True)


if __name__ == "__main__":
    # 1. SETUP UI ROOT
    # We create one Tk instance and keep it for the file dialogs
    root = Tk()
    root.withdraw()

    # 2. LOAD MODEL (Once)
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

    # 3. MAIN LOOP
    while True:
        # Ask for STL
        stl_path = askopenfilename(title="Select Test STL", filetypes=[("STL", ".stl")])
        if not stl_path:
            print("No file selected. Exiting.")
            break  # Exit loop if user cancels selection

        print(f"\nProcessing: {stl_path}...")

        # Process
        try:
            print("   Extracting features...")
            feats, tris, points = features.extract_triangle_features(stl_path)

            input_tensor = torch.tensor(feats, dtype=torch.float32).to(device)
            print("   Running Neural Network...")
            with torch.no_grad():
                probs = torch.sigmoid(model(input_tensor)).cpu().numpy()

            # Control State to communicate between Visualize function and Main Loop
            control_state = {"load_next": False}

            # Visualize (This pauses the script until window is closed)
            visualize_inference(points, tris, probs, control_state)

            # Check if we should loop or exit
            if control_state["load_next"]:
                print("♻️ Loading next file...")
                continue  # Loop again
            else:
                print("✅ Done. Exiting.")
                break  # Break loop (User pressed X or Q)

        except Exception as e:
            print(f"❌ Critical Error: {e}")
            # Optional: Allow retry if error
            import traceback

            traceback.print_exc()
            break