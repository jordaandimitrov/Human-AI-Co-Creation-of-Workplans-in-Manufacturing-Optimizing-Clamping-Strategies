import numpy as np
from vedo import Mesh, Plotter, Text2D
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import os
import json
import math
import random

# ----------------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------------
ANGLE_TOLERANCE_DEG = 10         # tolerance for angle windows around ±45°
SIDE_NORMAL_THRESHOLD = 0.2      # |nz| < threshold for side faces
OUTPUT_DIR = "training_set_cylinders"
JSON_OUT = os.path.join(OUTPUT_DIR, "cylinder_labels.json")


# ----------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------

def detect_cylinder_patches_with_angle(mesh, separation_deg=90, angle_tol_deg=10):
    """
    Detect two long contact strips on the same half of a cylinder.
    Separation (in deg) defines the angle between the patches.
    """

    normals = mesh.cell_normals
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]

    # Angle of each face normal projected into XY plane
    angles = np.arctan2(ny, nx)
    angles = (angles + 2*np.pi) % (2*np.pi)   # → [0, 2π]

    # Choose base contact angle (center of one strip)
    base_angle = np.pi / 4           # 45°, same orientation as before

    # Separation between the two strips
    sep = np.deg2rad(separation_deg)

    # Define target angles for two patches
    target1 = base_angle
    target2 = base_angle + sep

    # Normalize so target2 is inside [0, 2π]
    target2 = target2 % (2*np.pi)

    tol = np.deg2rad(angle_tol_deg)

    # Condition 1: nearly horizontal normals
    is_side = np.abs(nz) < 0.2

    # Condition 2: inside angular windows
    patch1 = np.abs(angles - target1) < tol
    patch2 = np.abs(angles - target2) < tol

    mask = is_side & (patch1 | patch2)
    indices = np.where(mask)[0].astype(int)

    print(f"   [Cylinder] Found {len(indices)} faces at separation = {separation_deg}°")

    return indices


def generate_labels_dict(mesh, selected_faces):
    labels = {}
    for i in range(mesh.ncells):
        labels[str(i)] = "unselected"
    for f in selected_faces:
        labels[str(f)] = "clamp"
    return labels


# ----------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------

if __name__ == "__main__":

    # File selection
    Tk().withdraw()
    stl_files = askopenfilenames(title="Select CYLINDRICAL STL FILES", filetypes=[("STL files", "*.stl")])
    if not stl_files:
        exit()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"\nSelected {len(stl_files)} cylinder files.")
    print(f"Output directory = {OUTPUT_DIR}")

    # Grid layout for viewer
    n_files = len(stl_files)
    n_cols = int(math.ceil(math.sqrt(n_files)))
    n_rows = int(math.ceil(n_files / n_cols))

    vp = Plotter(shape=(n_rows, n_cols), bg="white", size=(1800, 1000), sharecam=False)

    json_data = {}

    # ----------------------------------------------------
    # PROCESS EACH CYLINDER
    # ----------------------------------------------------
    for i, stl in enumerate(stl_files):
        fname = os.path.basename(stl)
        save_path = os.path.join(OUTPUT_DIR, fname)

        print(f"\n[{i+1}/{n_files}] Processing: {fname}")

        # Load mesh
        mesh = Mesh(stl)

        # Center mesh
        cm = mesh.center_of_mass()
        mesh.shift(-cm)

        # Random rotation around Z
        rotation = random.uniform(0, 0)
        mesh.rotate_z(rotation)
        mesh.compute_normals()

        # Save rotated geometry before feature extraction
        #mesh.write(save_path)

        # Cylinder-specific auto labeling
        indices = detect_cylinder_patches_with_angle(mesh)

        # Color faces
        face_colors = np.full((mesh.ncells, 3), 200, dtype=np.uint8)
        face_colors[indices] = [255, 0, 0]  # red for clamp faces
        mesh.cellcolors = face_colors

        # Save colored mesh
        #mesh.write(save_path)

        # Build JSON entry
        json_key = save_path.replace("\\", "/")
        labels_dict = generate_labels_dict(mesh, indices)
        labels_dict["_meta_rotation_z"] = rotation

        json_data[json_key] = labels_dict

        # Show in viewer
        vp.at(i).show(mesh, title=f"#{i} {fname}", axes=0)
        vp.at(i).add(Text2D(f"Rot: {rotation:.0f}°", pos="bottom-left", c="black", s=0.8))

    print("\nAll parts processed. Close the viewer to save JSON.")
    vp.interactive()

    # Save JSON
    with open(JSON_OUT, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n✅ JSON saved to: {JSON_OUT}")
    print(f"✅ Labeled STLs saved to: {OUTPUT_DIR}/")
