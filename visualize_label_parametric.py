from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import numpy as np
from vedo import Mesh, Plotter
import os
import json
import random

# --------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------
Z_CLAMP_HEIGHT = 25.0
Y_TOLERANCE = 4.0

# Normal Alignment Threshold
# 0.95 = Allows ~5% deviation (approx 18 degrees tilt).
NORMAL_ALIGNMENT_THRESHOLD = 0.95


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


def color_faces_geometrically(mesh, rotation_angle_deg):
    """
    Calculates constraints using Vector Projection and specific Dot Product thresholds.
    """
    global Z_CLAMP_HEIGHT, Y_TOLERANCE, NORMAL_ALIGNMENT_THRESHOLD

    # 1. FORCE UPDATE NORMALS
    mesh.compute_normals()

    # 2. Calculate Rotation Vectors
    theta = np.radians(rotation_angle_deg)

    # Rotated Y-Axis Vector (The ideal clamping direction)
    rot_y_vec = np.array([-np.sin(theta), np.cos(theta), 0])

    # 3. Get Mesh Data
    min_z, max_z = get_z_bounds(mesh)
    Z_THRESHOLD_BOTTOM = min_z + Z_CLAMP_HEIGHT

    centroids = mesh.cell_centers().points
    normals = mesh.cell_normals

    # --- Filter A: Normal Alignment (The 5% Check) ---
    # Dot Product ranges from 0 (perpendicular) to 1 (perfectly aligned).
    alignment_score = np.abs(np.dot(normals, rot_y_vec))

    y_aligned_mask = (alignment_score >= NORMAL_ALIGNMENT_THRESHOLD)
    y_normal_indices = np.where(y_aligned_mask)[0].astype(int)

    # --- Filter B: Height Check ---
    clamp_z_mask = (centroids[:, 2] <= Z_THRESHOLD_BOTTOM)
    clamp_z_indices = np.where(clamp_z_mask)[0].astype(int)

    # Intersection of A & B
    yz_intersection = np.intersect1d(clamp_z_indices, y_normal_indices).astype(int)

    # --- Filter C: Boundary Width Check (Projected) ---
    if len(yz_intersection) == 0:
        return mesh, []

    # Project centroids onto the Rotated Y-Axis to find "Width"
    projected_positions = np.dot(centroids, rot_y_vec)

    # Extract only the relevant faces to find the bounds
    relevant_projections = projected_positions[yz_intersection]

    # Robust boundary finding (Percentile based to ignore corner spikes)
    max_val = np.percentile(relevant_projections, 99)
    min_val = np.percentile(relevant_projections, 1)

    # Check boundaries
    cond1 = (projected_positions >= max_val - Y_TOLERANCE)
    cond2 = (projected_positions <= min_val + Y_TOLERANCE)

    boundary_mask = cond1 | cond2

    # Get indices
    boundary_indices = np.where(boundary_mask)[0].astype(int)
    final_clamp_indices = np.intersect1d(yz_intersection, boundary_indices).astype(int)

    # --- Coloring ---
    num_faces = mesh.ncells
    face_colors = np.full((num_faces, 3), 200, dtype=np.uint8)  # Grey
    face_colors[final_clamp_indices] = [255, 0, 0]  # Red
    mesh.cellcolors = face_colors

    return mesh, final_clamp_indices


# --------------------------------------------------------

if __name__ == "__main__":

    Tk().withdraw()
    stl_files = askopenfilenames(title="Select STL Part Files", filetypes=[("STL", ".stl")])
    if not stl_files:
        exit()

    all_labels_data = {}
    output_dir = "training_set"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_json_path = os.path.join(output_dir, 'all_part_labels.json')

    num_files = len(stl_files)
    n_rows = 2
    n_cols = int(np.ceil(num_files / n_rows))

    VP = Plotter(shape=(n_rows, n_cols), bg='white', size=(1600, 800))

    for i, stl_file in enumerate(stl_files):
        VP.at(i).camera.Elevation(10)

        mesh = Mesh(stl_file)

        # 1. Center
        cm = mesh.center_of_mass()
        mesh.shift(-cm)

        # 2. Random Rotation
        rotation_angle = random.uniform(0, 360)
        mesh.rotate_z(rotation_angle)

        # 3. Apply Logic with new Normal Threshold
        labeled_mesh, final_indices = color_faces_geometrically(mesh, rotation_angle)

        # 4. OVERWRITE THE FILE
        # This saves the mesh in its current state (Centered + Rotated)
        print(f"Overwriting file: {stl_file}...")
        labeled_mesh.write(stl_file)

        # 5. Save Labels Data
        json_key = 'training_set/' + os.path.basename(stl_file)
        labels_dict = generate_labels_dict(labeled_mesh, final_indices)
        labels_dict["_meta_rotation_z"] = rotation_angle
        all_labels_data[json_key] = labels_dict

        VP.show(labeled_mesh, title=f"Part {i} ({rotation_angle:.0f}°)", axes=0, interactive=False)

    VP.interactive()

    with open(output_json_path, 'w') as f:
        json.dump(all_labels_data, f, indent=2)

    print(f"✅ Saved labels to {output_json_path}")
    print(f"✅ Overwrote all original STL files with rotated versions.")