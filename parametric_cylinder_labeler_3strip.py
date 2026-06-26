import numpy as np
from vedo import Mesh
import os
import json
import math
import random
import argparse
from features import compute_outerness

# ----------------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------------
ANGLE_TOLERANCE_DEG = 10         # tolerance for angle windows around ±45°
SIDE_NORMAL_THRESHOLD = 0.2      # |nz| < threshold for side faces
OUTPUT_DIR = "training_set_cylinders"
JSON_OUT = os.path.join(OUTPUT_DIR, "all_part_labels.json")

# ----------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------

def detect_cylinder_patches_with_angle(mesh, rotation_deg=0, separation_deg=45, angle_tol_deg=10, outerness_thresh=0.85):
    """
    Detect three contact strips on a cylinder:
      - strip 1 at base_angle (45° + rotation)
      - strip 2 at base_angle + separation_deg
      - strip 3 at base_angle + 180° (opposite side)
    """

    normals = mesh.cell_normals
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]

    # Angle of each face normal projected into XY plane
    angles = np.arctan2(ny, nx)
    angles = (angles + 2*np.pi) % (2*np.pi)   # → [0, 2π]
    rotation_rad = np.deg2rad(rotation_deg)

    # Separation between the first two strips
    sep = np.deg2rad(separation_deg)

    # Base contact angle (center of first strip)
    base_angle = (np.pi - sep/2 + rotation_rad) % (2*np.pi)   # midpoint of S1/S2 pinned to 180°

    # Define target angles for three patches
    target1 = base_angle
    target2 = (base_angle + sep) % (2*np.pi)
    target3 = (base_angle + sep/2 + np.pi) % (2*np.pi)   # opposite of midpoint between strip1 and strip2

    tol = np.deg2rad(angle_tol_deg)

    def circ_diff(a, b):
        d = np.abs(a - b) % (2*np.pi)
        return np.minimum(d, 2*np.pi - d)

    # Condition 1: nearly horizontal normals (side faces)
    is_side = np.abs(nz) < 0.2

    # Condition 2: inside angular windows (circular distance handles 0/2π wrap)
    patch1 = circ_diff(angles, target1) < tol
    patch2 = circ_diff(angles, target2) < tol
    patch3 = circ_diff(angles, target3) < tol

    # Condition 3: outerness filter
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

    parser = argparse.ArgumentParser(description="Headless cylinder strip labeler")
    parser.add_argument("input_dir", help="Directory containing STL files to process")
    parser.add_argument("--out", default=OUTPUT_DIR, help="Output directory (default: training_set_cylinders)")
    args = parser.parse_args()

    stl_files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".stl")
    ])
    if not stl_files:
        print(f"No STL files found in {args.input_dir}")
        exit(1)

    os.makedirs(args.out, exist_ok=True)
    json_out = os.path.join(args.out, "all_part_labels.json")

    print(f"\nFound {len(stl_files)} STL files in {args.input_dir}")
    print(f"Output directory = {args.out}")

    json_data = {}

    for i, stl in enumerate(stl_files):
        fname = os.path.basename(stl)
        save_path = os.path.join(args.out, fname)

        print(f"\n[{i+1}/{len(stl_files)}] Processing: {fname}")

        mesh = Mesh(stl)
        cm = mesh.center_of_mass()
        mesh.shift(-cm)

        rotation = random.uniform(0, 360)
        mesh.rotate_z(rotation)
        mesh.compute_normals()

        indices, a1, a2, a3 = detect_cylinder_patches_with_angle(mesh, rotation_deg=rotation)
        print(f"   Strip angles: S1={a1:.1f}°  S2={a2:.1f}°  S3={a3:.1f}°")

        face_colors = np.full((mesh.ncells, 3), 200, dtype=np.uint8)
        face_colors[indices] = [255, 0, 0]
        mesh.cellcolors = face_colors
        mesh.write(save_path)

        json_key = save_path.replace("\\", "/")
        labels_dict = generate_labels_dict(mesh, indices)
        labels_dict["_meta_rotation_z"] = rotation
        json_data[json_key] = labels_dict

    with open(json_out, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\nDone. JSON saved to: {json_out}")
    print(f"Labeled STLs saved to: {args.out}/")
