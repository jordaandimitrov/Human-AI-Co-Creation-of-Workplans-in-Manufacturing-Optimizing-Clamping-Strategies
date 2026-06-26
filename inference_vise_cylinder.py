"""
inference_vise_cylinder.py

Cylinder vise fixture visualizer.
  - V-block (fixed jaw)  : cradles the two adjacent strips (S1, S2)
  - Flat jaw (moving jaw): presses against the opposite strip (S3)
  - Vise body            : base plate, guide rails, lead screw
"""

import torch
import torch.nn as nn
import numpy as np
import trimesh
import networkx as nx
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from vedo import Mesh, Plotter, Text2D, Box

try:
    import features
except ImportError:
    print("ERROR: features.py not found.")
    exit()


# ── 1. MODEL ──────────────────────────────────────────────────────────────────
class ClampSupportNet(nn.Module):
    def __init__(self, in_dim=15, hidden_dim=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, X): return self.net(X)


# ── 2. POST-PROCESSING ────────────────────────────────────────────────────────
def smart_expand_selection(initial_mask, prob_map, adjacency, normals, low_thresh=0.40):
    mask = initial_mask.copy()
    for _ in range(10):
        prev = np.sum(mask)
        border_A = mask[adjacency[:, 0]] & ~mask[adjacency[:, 1]]
        border_B = mask[adjacency[:, 1]] & ~mask[adjacency[:, 0]]
        sources    = np.concatenate([adjacency[border_A, 0], adjacency[border_B, 1]])
        candidates = np.concatenate([adjacency[border_A, 1], adjacency[border_B, 0]])
        if len(candidates) == 0:
            break
        prob_ok = prob_map[candidates] > low_thresh
        dots    = np.einsum('ij,ij->i', normals[sources], normals[candidates])
        flat_ok = dots > 0.98
        mask[candidates[prob_ok & flat_ok]] = True
        if np.sum(mask) == prev:
            break
    return mask


def filter_small_islands(mask, adjacency, face_areas, min_area=10.0):
    selected = np.where(mask)[0]
    if len(selected) == 0:
        return mask
    sub = mask[adjacency[:, 0]] & mask[adjacency[:, 1]]
    G = nx.from_edgelist(adjacency[sub]) if sub.any() else nx.Graph()
    G.add_nodes_from(selected)
    clean = np.zeros_like(mask, dtype=bool)
    for comp in nx.connected_components(G):
        if np.sum(face_areas[list(comp)]) >= min_area:
            clean[list(comp)] = True
    return clean


# ── 3. HELPERS ────────────────────────────────────────────────────────────────
def find_cylinder_axis(mesh):
    """Longest bounding-box dimension = cylinder axis."""
    b = mesh.bounds()
    dims = [b[1]-b[0], b[3]-b[2], b[5]-b[4]]
    ax = np.zeros(3)
    ax[np.argmax(dims)] = 1.0
    return ax


def get_clusters(mesh, mask, adjacency, face_areas, min_area=5.0):
    selected = np.where(mask)[0]
    if len(selected) == 0:
        return []

    sub = mask[adjacency[:, 0]] & mask[adjacency[:, 1]]
    G = nx.Graph()
    G.add_nodes_from(selected)
    if sub.any():
        G.add_edges_from(adjacency[sub])

    centroids = mesh.cell_centers().points
    normals   = mesh.cell_normals

    out = []
    for comp in nx.connected_components(G):
        idx  = list(comp)
        area = np.sum(face_areas[idx])
        if area < min_area:
            continue
        n = np.mean(normals[idx], axis=0)
        n /= np.linalg.norm(n) + 1e-6
        c = np.mean(centroids[idx], axis=0)
        out.append((area, np.array(idx), n, c))

    out.sort(key=lambda x: x[0], reverse=True)
    return out


_COMBOS = [(0,1,2), (0,2,1), (1,2,0)]   # (pairA, pairB, flat) indices into top-3

def split_vblock_flat(clusters, solution_idx=0):
    """
    Returns (clA, clB, clC) for the chosen solution_idx (0..2), or None if < 3 clusters.
    All three assignments of the top-3 clusters are enumerable.
    """
    if len(clusters) < 3:
        return None
    top = clusters[:3]
    a, b, c = _COMBOS[solution_idx % len(_COMBOS)]
    return top[a], top[b], top[c]


# ── 3. V-BLOCK (fixed jaw) ────────────────────────────────────────────────────
def build_vblock(clA, clB, cyl_ax, vblock_len, pull_offset, notch_w, vblock_height, axial_offset=0.0):
    """
    90° V-groove block. Width equals notch_w exactly — no flat shoulders outside the V.
    notch_w   : opening width of the V-groove (user-controlled)
    vblock_height : depth of the block body below the V opening (user-controlled)
    """
    _, idxA, nA, cA = clA
    _, idxB, nB, cB = clB

    n_avg = (nA + nB).copy()
    n_avg -= np.dot(n_avg, cyl_ax) * cyl_ax
    if np.linalg.norm(n_avg) < 1e-6:
        n_avg = np.cross(cyl_ax, [0, 1, 0])
    n_avg /= np.linalg.norm(n_avg)

    approach_vec = -n_avg
    midpoint = (cA + cB) / 2.0

    vy = approach_vec
    vz = cyl_ax
    vx = np.cross(vy, vz)
    vx /= np.linalg.norm(vx) + 1e-6

    diff = (cA - cB) - np.dot(cA - cB, cyl_ax) * cyl_ax
    strip_dist = np.linalg.norm(diff)

    hw      = notch_w / 2.0
    notch_d = notch_w / 2.0        # 90° V: depth = half-width
    height  = vblock_height

    geometric_shift = notch_d - (strip_dist / 2.0)
    final_pos = midpoint + vy * geometric_shift - vy * pull_offset

    # 5-point profile — width equals notch_w, no shoulder outside the V
    # Front face (z = -L/2): 0..4,  Back face (z = +L/2): 5..9
    L = vblock_len
    pts = [
        [-hw, -height, -L/2],   # 0  bottom-left
        [ hw, -height, -L/2],   # 1  bottom-right
        [ hw,  0,      -L/2],   # 2  top-right  (right V-edge)
        [ 0,  -notch_d,-L/2],   # 3  V-tip
        [-hw,  0,      -L/2],   # 4  top-left   (left V-edge)
        [-hw, -height,  L/2],   # 5
        [ hw, -height,  L/2],   # 6
        [ hw,  0,       L/2],   # 7
        [ 0,  -notch_d, L/2],   # 8
        [-hw,  0,       L/2],   # 9
    ]
    faces = [
        [0,1,2],[0,2,3],[0,3,4],    # front cap
        [5,7,6],[5,8,7],[5,9,8],    # back cap
        [0,1,6,5],                   # bottom
        [1,2,7,6],                   # right side
        [2,3,8,7],                   # right V-face
        [3,4,9,8],                   # left V-face
        [4,0,5,9],                   # left side
    ]

    vblock = Mesh([pts, faces])
    vblock.compute_normals()
    vblock.c("steelblue").alpha(0.90).linecolor("black")

    R = np.array([vx, vy, vz]).T
    T = np.eye(4); T[:3, :3] = R

    vblock.apply_transform(T)
    vblock.pos(final_pos + cyl_ax * axial_offset)

    return vblock, approach_vec, midpoint, idxA, idxB


# ── 4. FLAT JAW (moving jaw) ──────────────────────────────────────────────────
def build_flat_jaw(clC, cyl_ax, jaw_len, jaw_h, jaw_t, pull_offset, axial_offset=0.0):
    """
    Flat jaw plate + jaw body, oriented against strip C.
    """
    _, idxC, nC, cC = clC

    # Outward approach in the radial plane
    app = nC - np.dot(nC, cyl_ax) * cyl_ax
    if np.linalg.norm(app) < 1e-6:
        app = np.cross(cyl_ax, [0, 0, 1])
    app /= np.linalg.norm(app)

    vx = np.cross(app, cyl_ax)
    vx /= np.linalg.norm(vx) + 1e-6

    # Face of jaw sits at cC + app*pull_offset; center is jaw_t/2 further out
    face_pos  = cC + app * pull_offset
    jaw_center = face_pos + app * jaw_t / 2.0

    plate = Box(pos=(0, 0, 0), length=jaw_h, width=jaw_t, height=jaw_len)
    plate.c("steelblue").alpha(0.90).linecolor("black")

    R = np.array([vx, app, cyl_ax]).T
    T = np.eye(4); T[:3, :3] = R

    plate.apply_transform(T)
    plate.pos(jaw_center + cyl_ax * axial_offset)

    return plate, app, cC, idxC


# ── 5. MAIN FIXTURE BUILDER ───────────────────────────────────────────────────
def generate_fixture(mesh, mask, adjacency, face_areas, cyl_ax,
                      vblock_len, vblock_pull, flat_pull,
                      notch_w, vblock_height, jaw_h, jaw_t,
                      vblock_axial=0.0, flat_axial=0.0,
                      solution_idx=0, min_area=5.0):
    clusters = get_clusters(mesh, mask, adjacency, face_areas, min_area)
    result   = split_vblock_flat(clusters, solution_idx)
    if result is None:
        print(f"  Only {len(clusters)} cluster(s) detected — need at least 3.")
        return [], [], 0

    clA, clB, clC = result

    vblock_mesh, _, _, idxA, idxB = build_vblock(
        clA, clB, cyl_ax, vblock_len, vblock_pull, notch_w, vblock_height, vblock_axial)

    flat_mesh, _, _, idxC = build_flat_jaw(
        clC, cyl_ax, vblock_len, jaw_h, jaw_t, flat_pull, flat_axial)

    n_solutions = min(len(clusters), 3)
    actors = [vblock_mesh, flat_mesh]
    active = np.concatenate([idxA, idxB, idxC]).astype(int)
    return actors, active, n_solutions


# ── 7. VISUALIZATION ──────────────────────────────────────────────────────────
def visualize_inference(points, tris, probs, control_state):
    mesh = Mesh([points, tris])
    tm   = trimesh.Trimesh(vertices=points, faces=tris, process=False)
    adjacency  = tm.face_adjacency
    face_areas = tm.area_faces
    normals    = mesh.cell_normals
    cyl_ax     = find_cylinder_axis(mesh)

    plt = Plotter(title="Cylinder Vise + V-Block Fixture", bg="white", axes=0)

    state = {
        "threshold":    0.50,
        "show_fixture": True,
        "vblock_pull":  0.0,
        "flat_pull":    0.0,
        "vblock_len":   80.0,
        "vb_width":     60.0,
        "vb_height":    50.0,
        "jaw_h":        40.0,
        "jaw_t":        18.0,
        "solution_idx":  0,
        "n_solutions":   3,
        "post_process":  True,
        "vblock_axial":  0.0,
        "flat_axial":    0.0,
        "fill_thresh":  0.40,
        "actors":       [],
    }

    txt = Text2D("", pos="bottom-left", c="black", s=0.75)
    plt.add(txt)

    def update_view():
        base_mask = probs[:, 0] > state["threshold"]
        if state["post_process"]:
            mask = smart_expand_selection(base_mask, probs[:, 0], adjacency, normals, low_thresh=state["fill_thresh"])
            mask = filter_small_islands(mask, adjacency, face_areas)
        else:
            mask = base_mask

        cols = np.full((mesh.ncells, 4), [200, 200, 200, 255], dtype=np.uint8)
        cols[mask] = [255, 0, 0, 255]

        plt.remove(state["actors"])
        state["actors"] = []
        active = []

        if state["show_fixture"]:
            actors, active, n_sol = generate_fixture(
                mesh, mask, adjacency, face_areas, cyl_ax,
                state["vblock_len"], state["vblock_pull"], state["flat_pull"],
                state["vb_width"], state["vb_height"], state["jaw_h"], state["jaw_t"],
                vblock_axial=state["vblock_axial"], flat_axial=state["flat_axial"],
                solution_idx=state["solution_idx"],
            )
            state["n_solutions"] = n_sol
            if actors:
                state["actors"] = actors
                plt.add(actors)
                txt.text(
                    f"Solution {state['solution_idx'] % max(n_sol,1) + 1}/{max(n_sol,1)}  |  "
                    f"Thresh: {state['threshold']:.2f}  |  "
                    f"V-Block pull: {state['vblock_pull']:.0f} mm  |  "
                    f"Flat pull: {state['flat_pull']:.0f} mm"
                )
            else:
                txt.text("Need >= 3 strips — lower threshold or check model.")
        else:
            txt.text(f"Threshold: {state['threshold']:.2f}")

        if len(active) > 0:
            cols[active] = [0, 220, 0, 255]

        mesh.cellcolors = cols
        plt.render()

    # Slider callbacks
    def s_thresh(w, e):     state["threshold"]   = w.GetRepresentation().GetValue(); update_view()
    def s_vb_pull(w, e):    state["vblock_pull"] = w.GetRepresentation().GetValue(); update_view()
    def s_flat_pull(w, e):  state["flat_pull"]   = w.GetRepresentation().GetValue(); update_view()
    def s_len(w, e):        state["vblock_len"]  = w.GetRepresentation().GetValue(); update_view()
    def s_vb_w(w, e):       state["vb_width"]    = w.GetRepresentation().GetValue(); update_view()
    def s_vb_h(w, e):       state["vb_height"]   = w.GetRepresentation().GetValue(); update_view()
    def s_jaw_h(w, e):      state["jaw_h"]       = w.GetRepresentation().GetValue(); update_view()
    def s_jaw_t(w, e):      state["jaw_t"]       = w.GetRepresentation().GetValue(); update_view()
    def s_fill(w, e):         state["fill_thresh"]   = w.GetRepresentation().GetValue(); update_view()
    def s_vb_axial(w, e):     state["vblock_axial"]  = w.GetRepresentation().GetValue(); update_view()
    def s_flat_axial(w, e):   state["flat_axial"]    = w.GetRepresentation().GetValue(); update_view()

    # Button callbacks
    def btn_smart(*args):
        state["post_process"] = not state["post_process"]; update_view()
    def btn_fixture(*args):
        state["show_fixture"] = not state["show_fixture"]; update_view()
    def btn_cycle(*args):
        state["solution_idx"] = (state["solution_idx"] + 1) % max(state["n_solutions"], 1)
        update_view()
    def btn_next(*args):
        control_state["load_next"] = True; plt.close()

    # Row 1 (y=0.14): Confidence, jaw pull offsets, block length
    plt.add_slider(s_thresh,     0.10, 0.99, value=0.50, pos=[(0.05,0.14),(0.21,0.14)], title="Confidence")
    plt.add_slider(s_vb_pull,    0.0,  80.0, value=0.0,  pos=[(0.27,0.14),(0.43,0.14)], title="V-Block Pull (mm)")
    plt.add_slider(s_flat_pull,  0.0,  80.0, value=0.0,  pos=[(0.49,0.14),(0.65,0.14)], title="Flat Jaw Pull (mm)")
    plt.add_slider(s_len,       30.0, 200.0, value=80.0, pos=[(0.71,0.14),(0.87,0.14)], title="Block Length (mm)")

    # Row 2 (y=0.27): V-block and flat jaw shape
    plt.add_slider(s_vb_w,      20.0, 150.0, value=60.0,  pos=[(0.05,0.27),(0.21,0.27)], title="V-Block Width (mm)")
    plt.add_slider(s_vb_h,      10.0, 120.0, value=50.0,  pos=[(0.27,0.27),(0.43,0.27)], title="V-Block Height (mm)")
    plt.add_slider(s_jaw_h,     10.0, 100.0, value=40.0,  pos=[(0.49,0.27),(0.65,0.27)], title="Flat Jaw Height (mm)")
    plt.add_slider(s_jaw_t,      5.0,  60.0, value=18.0,  pos=[(0.71,0.27),(0.87,0.27)], title="Flat Jaw Depth (mm)")

    # Row 3 (y=0.40): Axial positioning + smart fill threshold
    plt.add_slider(s_fill,       0.05, 0.90, value=0.40,   pos=[(0.05,0.40),(0.21,0.40)], title="Fill Threshold")
    plt.add_slider(s_vb_axial, -150.0,150.0, value=0.0,    pos=[(0.27,0.40),(0.43,0.40)], title="V-Block Axial (mm)")
    plt.add_slider(s_flat_axial,-150.0,150.0, value=0.0,   pos=[(0.49,0.40),(0.65,0.40)], title="Flat Jaw Axial (mm)")

    # Row 4 (y=0.53): Buttons
    plt.add_button(btn_smart, states=[" Smart Fill: ON ", " Smart Fill: OFF"],
                   c=["w","w"], bc=["g","r"], pos=(0.18, 0.53), size=18)
    plt.add_button(btn_fixture, states=[" Hide Fixture ", " Show Fixture "],
                   c=["w","w"], bc=["grey","b"], pos=(0.38, 0.53), size=18)
    plt.add_button(btn_cycle, states=[" Next Solution "],
                   c=["white"], bc=["darkgreen"], pos=(0.58, 0.53), size=18)
    plt.add_button(btn_next, states=[" LOAD NEW FILE "],
                   c=["white"], bc=["orange"], pos=(0.5, 0.95), size=24, font="courier")

    update_view()
    plt.show(mesh, interactive=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    model_path = askopenfilename(title="Select Cylinder Model (.pth)",
                                  filetypes=[("Model", "*.pth")])
    if not model_path:
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ClampSupportNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded on {device}.")

    while True:
        stl_path = askopenfilename(title="Select Cylinder STL",
                                    filetypes=[("STL", "*.stl")])
        if not stl_path:
            break
        print(f"\nProcessing: {stl_path}...")
        try:
            feats, tris, points = features.extract_triangle_features(stl_path)
            with torch.no_grad():
                probs = torch.sigmoid(
                    model(torch.tensor(feats, dtype=torch.float32).to(device))
                ).cpu().numpy()

            control_state = {"load_next": False}
            visualize_inference(points, tris, probs, control_state)

            if not control_state["load_next"]:
                break
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()
            break
