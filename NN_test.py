import os
import torch
import torch.nn as nn
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from vedo import Mesh, Plotter, color_map, Text2D


# ============================================================
# 1. MODEL ARCHITECTURE
# (Updated to in_dim=12 to match the Cluster Model)
# ============================================================
class ClampSupportNet(nn.Module):
    def __init__(self, in_dim=12, hidden_dim=128, out_dim=2):  # <--- CHANGED TO 12
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, X):
        return self.net(X)


# ============================================================
# 2. FEATURE EXTRACTION (UPDATED WITH OUTERNESS)
# ============================================================
def extract_triangle_features_stl(stl_file_path):
    print(f"Processing: {os.path.basename(stl_file_path)}...")

    mesh = Mesh(stl_file_path)
    mesh.triangulate()

    points = mesh.points
    tris = np.array(mesh.cells).astype(np.int64)
    normals = mesh.cell_normals
    centroids = mesh.cell_centers().points
    areas = mesh.area()
    part_center = np.array(mesh.center_of_mass(), dtype=np.float32)
    dist_to_center = np.linalg.norm(centroids - part_center, axis=1)

    # ----------------------------
    # BASIC FEATURES
    # ----------------------------
    rel_pos = centroids - part_center
    radial_dist = np.linalg.norm(rel_pos[:, 0:2], axis=1)

    z_values = centroids[:, 2]
    min_z, max_z = np.min(z_values), np.max(z_values)
    height_range = max_z - min_z if (max_z - min_z) > 1e-6 else 1.0
    rel_height = (z_values - min_z) / height_range

    v0 = points[tris[:, 0]]
    v1 = points[tris[:, 1]]
    v2 = points[tris[:, 2]]
    edge_a = np.linalg.norm(v1 - v0, axis=1)
    edge_b = np.linalg.norm(v2 - v1, axis=1)
    edge_c = np.linalg.norm(v0 - v2, axis=1)

    # ----------------------------
    # OUTERNESS (Wall Proximity)
    # ----------------------------
    b = mesh.bounds()
    dist_x = np.minimum(abs(centroids[:, 0] - b[0]), abs(centroids[:, 0] - b[1]))
    dist_y = np.minimum(abs(centroids[:, 1] - b[2]), abs(centroids[:, 1] - b[3]))
    dist_to_wall = np.minimum(dist_x, dist_y)
    max_dim = max(b[1] - b[0], b[3] - b[2])
    outerness = 1.0 - (dist_to_wall / (max_dim * 0.5 + 1e-6))
    outerness = np.clip(outerness, 0, 1)

    # ----------------------------
    # SMART RAY (Opposite Face Check)
    # ----------------------------
    ray_directions = -normals
    ray_ends = centroids + ray_directions * 500.0
    opposite_quality = np.zeros(len(tris))

    my_verticality = 1.0 - np.abs(normals[:, 2])
    check_indices = np.where(my_verticality > 0.8)[0]

    for idx in check_indices:
        origin = centroids[idx]
        target = ray_ends[idx]

        hits = mesh.intersect_with_line(origin, target)

        valid_hit = False
        if len(hits) > 0:
            for hit in hits:
                if np.linalg.norm(hit - origin) < 1.0:
                    continue
                try:
                    _, hit_idx = mesh.closest_point(hit, return_cell_id=True)
                except:
                    continue
                if hit_idx >= 0 and hit_idx < len(normals):
                    opp_normal = normals[hit_idx]
                    alignment = np.dot(normals[idx], opp_normal)
                    opp_verticality = 1.0 - abs(opp_normal[2])
                    if alignment < -0.85 and opp_verticality > 0.85:
                        valid_hit = True
                        break

        opposite_quality[idx] = 1.0 if valid_hit else 0.0

    # ----------------------------
    # NORMALIZATION (Same as training)
    # ----------------------------
    max_area = np.max(areas) if np.max(areas) > 1e-6 else 1.0
    max_dist = np.max(dist_to_center) if np.max(dist_to_center) > 1e-6 else 1.0
    max_rad = np.max(radial_dist) if np.max(radial_dist) > 1e-6 else 1.0
    max_edge = max(np.max(edge_a), np.max(edge_b), np.max(edge_c))
    max_edge = max_edge if max_edge > 1e-6 else 1.0

    # ----------------------------
    # FINAL 13-DIM FEATURE MATRIX
    # ----------------------------
    feats = np.zeros((len(tris), 13), dtype=np.float32)
    feats[:, 0] = areas / max_area
    feats[:, 1] = 1.0
    feats[:, 2:5] = normals
    feats[:, 5] = dist_to_center / max_dist
    feats[:, 6] = radial_dist / max_rad
    feats[:, 7] = rel_height
    feats[:, 8] = edge_a / max_edge
    feats[:, 9] = edge_b / max_edge
    feats[:, 10] = edge_c / max_edge
    feats[:, 11] = outerness
    feats[:, 12] = opposite_quality

    return feats, tris, points



# ============================================================
# 3. VISUALIZATION (With Threshold Filtering)
# ============================================================
def visualize_triangles(points, tris, tri_probs, threshold=0.90):
    """
    Visualizes results using a single Mesh object and Scalar Mapping.
    """
    print("Generating visualization mesh...")

    # 1. Create one single mesh
    mesh = Mesh([points, tris])

    # 2. Prepare Scalar Data (The Probability of being a Clamp)
    # tri_probs shape is (N, 2), we want class 0 (Clamp) or class 1?
    # Based on your training script:
    # y[clamp_indices, 0] = 1.0 -> Class 0 is Clamp.
    clamp_probability = tri_probs[:, 0]

    # 3. Add this data to the mesh
    mesh.celldata["Probability"] = clamp_probability

    # 4. Create the Plotter
    plt = Plotter(title="Inference Results", bg="white", axes=1)

    # 5. Define a custom coloring function
    # We color based on threshold manually for crisp cut-off
    def apply_threshold_color(thresh):
        # Default Grey
        colors = np.full((mesh.ncells, 4), [200, 200, 200, 50], dtype=np.uint8)  # RGBA

        # High confidence Clamp -> Red
        mask = clamp_probability > thresh
        colors[mask] = [255, 0, 0, 255]  # Red, Opaque

        mesh.cellcolors = colors

    apply_threshold_color(threshold)

    # 6. Interaction
    instructions = Text2D(f"Threshold: {threshold}", pos="bottom-left")
    plt.add(instructions)

    def on_slider(widget, event):
        val = widget.GetRepresentation().GetValue()
        apply_threshold_color(val)
        instructions.text(f"Threshold: {val:.2f}")

    plt.add_slider(
        on_slider,
        xmin=0.0, xmax=1.0, value=threshold,
        pos=[(0.1, 0.1), (0.4, 0.1)], title="Confidence"
    )

    plt.show(mesh, interactive=True)

# ============================================================
# 4. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    Tk().withdraw()

    # 1. Load Model Weights
    print("Please select the 'separation_model_12dim.pth' file...")
    model_path = askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
    if not model_path:
        print("No model selected. Exiting.")
        exit()

    # 2. Select New STL Part
    print("Please select the STL file to test...")
    stl_path = askopenfilename(filetypes=[("STL Files", "*.stl")])
    if not stl_path:
        print("No STL selected. Exiting.")
        exit()

    # 3. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # Note: Input Dimension is now 12
    model = ClampSupportNet(in_dim=13, hidden_dim=128, out_dim=2)

    # 4. Smart Load State Dict
    checkpoint = torch.load(model_path, map_location=device)
    if list(checkpoint.keys())[0].startswith('module.'):
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # 5. Extract Features & Predict
    try:
        features, tris, points = extract_triangle_features_stl(stl_path)
        input_tensor = torch.tensor(features, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

        print("Prediction complete.")

        # 6. Visualize with Threshold 0.90
        visualize_triangles(points, tris, probs, threshold=0.90)

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()