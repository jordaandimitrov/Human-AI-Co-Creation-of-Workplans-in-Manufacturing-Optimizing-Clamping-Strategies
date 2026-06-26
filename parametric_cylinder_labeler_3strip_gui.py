import numpy as np
from vedo import Mesh, Plotter, Text2D
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import os
import json
import math
import random
from features import compute_outerness

# ----------------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------------
ANGLE_TOLERANCE_DEG = 10
SIDE_NORMAL_THRESHOLD = 0.2
OUTPUT_DIR = "training_set_cylinders"
JSON_OUT = os.path.join(OUTPUT_DIR, "all_part_labels.json")

# ----------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------

def detect_cylinder_patches_with_angle(mesh, rotation_deg=0, separation_deg=45, angle_tol_deg=10, outerness_thresh=0.85):
    normals = mesh.cell_normals
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]

    angles = np.arctan2(ny, nx)
    angles = (angles + 2*np.pi) % (2*np.pi)
    rotation_rad = np.deg2rad(rotation_deg)

    sep = np.deg2rad(separation_deg)
    base_angle = (np.pi - sep/2 + rotation_rad) % (2*np.pi)   # midpoint of S1/S2 pinned to 180°

    target1 = base_angle
    target2 = (base_angle + sep) % (2*np.pi)
    target3 = (base_angle + sep/2 + np.pi) % (2*np.pi)

    tol = np.deg2rad(angle_tol_deg)

    def circ_diff(a, b):
        d = np.abs(a - b) % (2*np.pi)
        return np.minimum(d, 2*np.pi - d)

    is_side = np.abs(nz) < 0.2

    patch1 = circ_diff(angles, target1) < tol
    patch2 = circ_diff(angles, target2) < tol
    patch3 = circ_diff(angles, target3) < tol

    centroids = mesh.cell_centers().points
    outerness = compute_outerness(mesh, centroids)
    is_outer = outerness >= outerness_thresh

    mask = is_side & is_outer & (patch1 | patch2 | patch3)
    indices = np.where(mask)[0].astype(int)

    a1 = math.degrees(target1) % 360
    a2 = math.degrees(target2) % 360
    a3 = math.degrees(target3) % 360

    print(f"   [Cylinder] Found {len(indices)} faces (3 strips, separation = {separation_deg}°)")
    print(f"   Strip angles: {a1:.1f}°  {a2:.1f}°  {a3:.1f}°")

    return indices, a1, a2, a3


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

    Tk().withdraw()
    stl_files = askopenfilenames(title="Select CYLINDRICAL STL FILES", filetypes=[("STL files", "*.stl")])
    if not stl_files:
        exit()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nSelected {len(stl_files)} cylinder files.")
    print(f"Output directory = {OUTPUT_DIR}")

    n_files = len(stl_files)
    n_cols = int(math.ceil(math.sqrt(n_files)))
    n_rows = int(math.ceil(n_files / n_cols))

    vp = Plotter(shape=(n_rows, n_cols), bg="white", size=(1800, 1000), sharecam=False)

    json_data = {}

    for i, stl in enumerate(stl_files):
        fname = os.path.basename(stl)
        save_path = os.path.join(OUTPUT_DIR, fname)

        print(f"\n[{i+1}/{n_files}] Processing: {fname}")

        mesh = Mesh(stl)
        cm = mesh.center_of_mass()
        mesh.shift(-cm)

        rotation = random.uniform(0, 360)
        mesh.rotate_z(rotation)
        mesh.compute_normals()

        indices, a1, a2, a3 = detect_cylinder_patches_with_angle(mesh, rotation_deg=rotation)

        face_colors = np.full((mesh.ncells, 3), 200, dtype=np.uint8)
        face_colors[indices] = [255, 0, 0]
        mesh.cellcolors = face_colors
        mesh.write(save_path)

        json_key = save_path.replace("\\", "/")
        labels_dict = generate_labels_dict(mesh, indices)
        labels_dict["_meta_rotation_z"] = rotation
        json_data[json_key] = labels_dict

        vp.at(i).show(mesh, title=f"#{i} {fname}", axes=0)
        vp.at(i).add(Text2D(
            f"Rot: {rotation:.0f}°\nS1: {a1:.1f}°  S2: {a2:.1f}°  S3: {a3:.1f}°",
            pos="bottom-left", c="black", s=0.8
        ))

    print("\nAll parts processed. Close the viewer to save JSON.")
    vp.interactive()

    with open(JSON_OUT, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\nDone. JSON saved to: {JSON_OUT}")
    print(f"Labeled STLs saved to: {OUTPUT_DIR}/")
