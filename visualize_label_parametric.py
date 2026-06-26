from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import numpy as np
from vedo import Mesh, Plotter, Text2D
import os
import json
import random
import math

# IMPORT SHARED FEATURES
try:
    import features
except ImportError:
    print("❌ ERROR: 'features.py' not found. Please ensure it is in the same folder.")
    exit()

# --------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------
Z_CLAMP_HEIGHT = 25.0
RAY_SCORE_THRESHOLD = 0.5


# --------------------------------------------------------

def get_z_bounds(mesh):
    vertices = mesh.points
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    return min_z, max_z


def generate_labels_dict(mesh, final_clamp_indices):
    num_faces = mesh.ncells
    labels_dict = {}
    for i in range(num_faces):
        labels_dict[str(i)] = "unselected"
    for index in final_clamp_indices:
        labels_dict[str(index)] = "clamp"
    return labels_dict


def auto_label_physically(mesh, feature_matrix):
    """
    Uses the robust features to decide labels automatically.
    """
    global Z_CLAMP_HEIGHT, RAY_SCORE_THRESHOLD

    centroids = mesh.cell_centers().points
    min_z, max_z = get_z_bounds(mesh)
    Z_THRESHOLD_BOTTOM = min_z + Z_CLAMP_HEIGHT

    ray_scores = feature_matrix[:, 12]

    # 1. Height Check
    is_bottom = (centroids[:, 2] <= Z_THRESHOLD_BOTTOM)

    # 2. Physics Check
    has_partner = (ray_scores > RAY_SCORE_THRESHOLD)

    # 3. Combine
    valid_mask = is_bottom & has_partner
    final_clamp_indices = np.where(valid_mask)[0].astype(int)

    # Visuals (Apply colors for saving)
    num_faces = mesh.ncells
    face_colors = np.full((num_faces, 3), 200, dtype=np.uint8)
    face_colors[final_clamp_indices] = [255, 0, 0]
    mesh.cellcolors = face_colors

    print(f"   [Auto-Label] Found {len(final_clamp_indices)} valid clamp faces.")
    return mesh, final_clamp_indices


if __name__ == "__main__":
    Tk().withdraw()
    # 1. Select Source Files
    stl_files = askopenfilenames(title="Select SOURCE STL Files", filetypes=[("STL", ".stl")])
    if not stl_files: exit()

    # 2. Setup Output Directory
    output_dir = "training_set"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_json_path = os.path.join(output_dir, 'all_part_labels.json')

    num_files = len(stl_files)
    print(f"Selected {num_files} files. Destination: {os.path.abspath(output_dir)}")

    # 3. Setup Grid
    n_cols = int(math.ceil(math.sqrt(num_files)))
    n_rows = int(math.ceil(num_files / n_cols))

    VP = Plotter(shape=(n_rows, n_cols), bg='white', size=(1800, 1000), sharecam=False, title="Dataset Generator")

    all_labels_data = {}

    for i, stl_file in enumerate(stl_files):
        filename = os.path.basename(stl_file)

        # Define the NEW path inside training_set
        save_path = os.path.join(output_dir, filename)

        print(f"\n[{i + 1}/{num_files}] Processing {filename}...")

        # Load Original
        mesh = Mesh(stl_file)

        # Center & Rotate
        cm = mesh.center_of_mass()
        mesh.shift(-cm)

        rotation_angle = random.uniform(0, 360)
        mesh.rotate_z(rotation_angle)
        mesh.compute_normals()

        # --- STEP A: SAVE GEOMETRY ---
        # We save the rotated geometry to the training folder immediately.
        # The feature extractor needs to read this specific file.
        print(f"   Saving geometry to: {save_path}")
        mesh.write(save_path)

        # --- STEP B: EXTRACT FEATURES ---
        # Read from the NEW file so features match the rotated coordinates
        try:
            feats, _, _ = features.extract_triangle_features(save_path)
        except Exception as e:
            print(f"   ❌ Feature extraction failed: {e}")
            continue

        # --- STEP C: LABEL ---
        labeled_mesh, final_indices = auto_label_physically(mesh, feats)

        # --- STEP D: SAVE VISUALS ---
        # Overwrite the file in training_set with the colored version.
        # (Standard STLs don't strictly support color, but Vedo/Binary STL allows it).
        # Even if color is lost in some viewers, the Geometry is correct.
        labeled_mesh.write(save_path)

        # --- STEP E: SAVE METADATA ---
        # Crucial: The key in JSON must match the path the training script looks for.
        # We store the relative path "training_set/part.stl"
        json_key = os.path.join(output_dir, filename).replace("\\", "/")

        labels_dict = generate_labels_dict(labeled_mesh, final_indices)
        labels_dict["_meta_rotation_z"] = rotation_angle
        all_labels_data[json_key] = labels_dict

        # Add to Viewer
        VP.at(i).show(labeled_mesh, title=f"#{i}: {filename}", axes=0, interactive=False)
        rot_txt = Text2D(f"Rot: {rotation_angle:.0f}°", pos="bottom-left", s=0.8, c="black")
        VP.at(i).add(rot_txt)

    print("\n✅ Processing complete.")
    print("   Review the window. Close it to finalize the JSON file.")

    VP.interactive()

    # Write the master index file
    with open(output_json_path, 'w') as f:
        json.dump(all_labels_data, f, indent=2)

    print(f"✅ JSON Labels saved to: {output_json_path}")
    print(f"✅ All {num_files} STLs saved to: {output_dir}/")