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
    def __init__(self, in_dim=15, hidden_dim=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, X): return self.net(X)


# ============================================================
# 2. POST-PROCESSING
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
# 3. ADVANCED CLAMP GENERATION (Valid Pairs Only)
# ============================================================
def get_orientation_matrix(forward_vector):
    x_axis = forward_vector / (np.linalg.norm(forward_vector) + 1e-6)
    world_z = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(world_z, x_axis)
    if np.linalg.norm(y_axis) < 1e-3: y_axis = np.array([0.0, 1.0, 0.0])
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    R = np.array([x_axis, y_axis, z_axis]).T
    T = np.eye(4)
    T[:3, :3] = R
    return T


def get_flat_patches(mesh, mask, adjacency, face_areas):
    selected_indices = np.where(mask)[0]
    if len(selected_indices) == 0: return []

    subset_mask = mask[adjacency[:, 0]] & mask[adjacency[:, 1]]
    subset_edges = adjacency[subset_mask]

    if len(subset_edges) == 0: return [selected_indices]

    normals = mesh.cell_normals
    norm_A = normals[subset_edges[:, 0]]
    norm_B = normals[subset_edges[:, 1]]
    dots = np.einsum('ij,ij->i', norm_A, norm_B)

    # Split sharp corners
    flat_edges = subset_edges[dots > 0.965]

    G = nx.from_edgelist(flat_edges)
    G.add_nodes_from(selected_indices)

    components = list(nx.connected_components(G))
    comp_stats = []
    for comp in components:
        idx_list = list(comp)
        total_area = np.sum(face_areas[idx_list])
        comp_stats.append((total_area, idx_list))

    comp_stats.sort(key=lambda x: x[0], reverse=True)
    patches = [x[1] for x in comp_stats]
    return patches


def generate_clamp_actors(mesh, mask, adjacency, face_areas, opening_offset, base_dims, solution_index=0,
                          min_area=50.0):
    """
    Returns: (actors, active_indices)
    Also generates the VISE BASE at the bottom of the part.
    base_dims = (width, length) from sliders.
    """
    raw_patches = get_flat_patches(mesh, mask, adjacency, face_areas)
    if len(raw_patches) == 0: return [], []

    patches = []
    for p in raw_patches:
        area = np.sum(face_areas[p])
        if area >= min_area:
            patches.append(p)

    if len(patches) == 0:
        print(f"⚠️ All patches were too small (Noise).")
        return [], []

    centroids = mesh.cell_centers().points
    normals = mesh.cell_normals

    patch_normals = []
    for p in patches:
        n = np.mean(normals[p], axis=0)
        n /= (np.linalg.norm(n) + 1e-6)
        patch_normals.append(n)

    valid_pairs = []
    if len(patches) >= 2:
        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                n1 = patch_normals[i]
                n2 = patch_normals[j]
                alignment = np.dot(n1, n2)
                if alignment < -0.5:
                    area_i = np.sum(face_areas[patches[i]])
                    area_j = np.sum(face_areas[patches[j]])
                    score = area_i + area_j
                    valid_pairs.append((score, patches[i], patches[j]))

    valid_pairs.sort(key=lambda x: x[0], reverse=True)

    used_patch_indices = set()
    unique_solutions = []

    # Map patch IDs back to index for tracking
    patch_id_map = {id(p): i for i, p in enumerate(patches)}

    for _, pA, pB in valid_pairs:
        idA = patch_id_map[id(pA)]
        idB = patch_id_map[id(pB)]
        if (idA not in used_patch_indices) and (idB not in used_patch_indices):
            unique_solutions.append(('pair', pA, pB))
            used_patch_indices.add(idA)
            used_patch_indices.add(idB)

    # 6. RETRIEVE REQUESTED SOLUTION
    if solution_index >= len(unique_solutions):
        print(f"⚠️ No more unique solutions found.")
        return [], []

    sol_type, patch_A, patch_B = unique_solutions[solution_index]

    actors = []

    # --- COMMON VISE PARAMETERS ---
    jaw_width = 80.0
    jaw_height = 25.0
    jaw_thickness = 10.0

    # Calculate Part Bottom (Z min)
    # The vise base sits here.
    part_z_min = mesh.bounds()[4]  # [xmin, xmax, ymin, ymax, zmin, zmax]

    # We need the CENTER of the clamp for the base position
    clamp_center_xy = np.array([0.0, 0.0, 0.0])  # Placeholder

    if sol_type == 'pair':
        active_indices = np.concatenate([patch_A, patch_B])

        raw_center_A = np.mean(centroids[patch_A], axis=0)
        raw_center_B = np.mean(centroids[patch_B], axis=0)

        norm_A = np.mean(normals[patch_A], axis=0)
        norm_A[2] = 0.0
        if np.linalg.norm(norm_A) < 1e-6:
            norm_A = np.array([1.0, 0.0, 0.0])
        else:
            norm_A /= np.linalg.norm(norm_A)

        squeeze_axis = -norm_A

        midpoint = (raw_center_A + raw_center_B) / 2.0
        clamp_center_xy = midpoint  # For base placement

        vec_A = raw_center_A - midpoint
        dist_A = np.dot(vec_A, squeeze_axis)
        vec_B = raw_center_B - midpoint
        dist_B = np.dot(vec_B, squeeze_axis)

        aligned_center_A = midpoint + (dist_A * squeeze_axis)
        aligned_center_B = midpoint + (dist_B * squeeze_axis)

        # Jaws are sliding, so their Z is determined by the part surface
        # But we want the JAWS to slide ON TOP of the base.
        # So Jaw Bottom = Part Bottom.
        # Jaw Center Z = Part Bottom + Jaw Height / 2

        jaw_z = part_z_min + (jaw_height / 2.0)
        aligned_center_A[2] = jaw_z
        aligned_center_B[2] = jaw_z

        pos_A = aligned_center_A - (squeeze_axis * opening_offset)
        pos_B = aligned_center_B + (squeeze_axis * opening_offset)

        T_matrix_A = get_orientation_matrix(squeeze_axis)
        T_matrix_B = get_orientation_matrix(-squeeze_axis)

        configs = [(pos_A, T_matrix_A), (pos_B, T_matrix_B)]

    # 7. GENERATE JAWS
    for pos, transform in configs:
        j = Box(pos=(0, 0, 0), length=jaw_thickness, width=jaw_width, height=jaw_height)
        j.apply_transform(transform)
        j.pos(pos)
        j.c("grey").alpha(0.9).linecolor("black")
        actors.append(j)

    # 8. GENERATE VISE BASE
    # Base is a big block underneath.
    # Top of Base = part_z_min.
    # Height of Base = 50mm (arbitrary thick block)
    base_h = 50.0
    base_w = base_dims[0]  # From slider
    base_l = base_dims[1]  # From slider

    # Center of base
    # X,Y = Center of clamp action (midpoint)
    # Z = part_z_min - (base_h / 2)
    base_pos = np.copy(clamp_center_xy)
    base_pos[2] = part_z_min - (base_h / 2.0)

    # Orient base to match Squeeze Axis
    # Squeeze Axis corresponds to Length of the vise usually
    T_base = get_orientation_matrix(squeeze_axis)

    base_box = Box(pos=(0, 0, 0), length=base_l, width=base_w, height=base_h)
    base_box.apply_transform(T_base)
    base_box.pos(base_pos)
    base_box.c("darkgrey").alpha(1.0).linecolor("black")

    actors.append(base_box)

    return actors, active_indices


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
        "base_width": 100.0,
        "base_length": 200.0,
        "clamp_actors": [],
        "solution_index": 0
    }

    txt_info = Text2D("", pos="bottom-left")
    plt.add(txt_info)

    def update_view():
        base_mask = probs[:, 0] > state["threshold"]
        if state["post_process"]:
            mask = smart_expand_selection(base_mask, probs[:, 0], adjacency, normals)
            mask = filter_small_islands(mask, adjacency, face_areas)
        else:
            mask = base_mask

        cols = np.full((mesh.ncells, 4), [220, 220, 220, 50], dtype=np.uint8)
        cols[mask] = [255, 0, 0, 255]

        plt.remove(state["clamp_actors"])
        state["clamp_actors"] = []
        active_indices = []

        status_msg = f"Jaw Offset: {state['offset']:.1f} mm"

        if state["show_clamp"]:
            # Pass tuple (Width, Length) for base
            base_dims = (state["base_width"], state["base_length"])

            jaws, active_indices = generate_clamp_actors(
                mesh, mask, adjacency, face_areas,
                state["offset"], base_dims,
                solution_index=state["solution_index"],
                min_area=50.0
            )

            if len(jaws) == 0 and state["solution_index"] > 0:
                status_msg += " | ⚠️ No more unique solutions!"
            else:
                status_msg += f" | Solution Tier: {state['solution_index'] + 1}"

            state["clamp_actors"] = jaws
            plt.add(jaws)

        if len(active_indices) > 0:
            cols[active_indices] = [0, 255, 0, 255]  # Green

        mesh.cellcolors = cols
        txt_info.text(status_msg)
        plt.render()

    def slide_thresh(w, e):
        state["threshold"] = w.GetRepresentation().GetValue(); update_view()

    def slide_offset(w, e):
        state["offset"] = w.GetRepresentation().GetValue(); update_view()

    # NEW SLIDERS for Vise Base
    def slide_base_w(w, e):
        state["base_width"] = w.GetRepresentation().GetValue(); update_view()

    def slide_base_l(w, e):
        state["base_length"] = w.GetRepresentation().GetValue(); update_view()

    def btn_smart(*args):
        state["post_process"] = not state["post_process"]; update_view()

    def btn_clamp(*args):
        state["show_clamp"] = not state["show_clamp"]; update_view()

    def btn_load_next(*args):
        control_state["load_next"] = True; plt.close()

    def btn_next_sol(*args):
        state["solution_index"] += 1; state["show_clamp"] = True; update_view()

    def btn_reset_sol(*args):
        state["solution_index"] = 0; state["show_clamp"] = True; update_view()

    # Layout
    plt.add_slider(slide_thresh, 0.1, 0.99, value=0.90, pos=[(0.1, 0.05), (0.3, 0.05)], title="Confidence")
    plt.add_slider(slide_offset, 0.0, 100.0, value=10.0, pos=[(0.4, 0.05), (0.6, 0.05)], title="Jaw Open (mm)")

    # Base Dimension Sliders (Right Side)
    plt.add_slider(slide_base_w, 50.0, 300.0, value=100.0, pos=[(0.7, 0.25), (0.9, 0.25)], title="Vise Width")
    plt.add_slider(slide_base_l, 100.0, 500.0, value=200.0, pos=[(0.7, 0.20), (0.9, 0.20)], title="Vise Length")

    plt.add_button(btn_smart, states=[" Smart Fill: ON ", " Smart Fill: OFF"], c=["w", "w"], bc=["g", "r"],
                   pos=(0.8, 0.12), size=20)
    plt.add_button(btn_clamp, states=[" Show Clamp ", " Hide Clamp "], c=["w", "w"], bc=["b", "grey"], pos=(0.8, 0.08),
                   size=20)
    plt.add_button(btn_next_sol, states=[" Next Option -> "], c=["black"], bc=["yellow"], pos=(0.6, 0.12), size=20)
    plt.add_button(btn_reset_sol, states=[" Reset "], c=["white"], bc=["darkblue"], pos=(0.45, 0.12), size=20)
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
            if control_state["load_next"]:
                continue
            else:
                break
        except Exception as e:
            print(f"❌ Critical Error: {e}")
            import traceback

            traceback.print_exc()
            break