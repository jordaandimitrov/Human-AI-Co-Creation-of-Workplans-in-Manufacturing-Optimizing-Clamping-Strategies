"""
Auto-label cylindrical parts for v-block fixture (90° v-block, 45° faces).

Clamp faces are those whose outward normal points downward-outward at roughly
45° — the zone that contacts the two angled faces of the v-block.

For a 90° v-block the nominal contact normal_z = -sin(45°) ≈ -0.707.
We use a broad zone around that to capture the full grip area.

Usage:
    python label_vblock_headless.py
    python label_vblock_headless.py --input_dir raw_vblock_parts --output_dir training_set_vblock --augment 3
"""

import os
import glob
import json
import random
import argparse
import numpy as np
from vedo import Mesh

try:
    import features
except ImportError:
    print("ERROR: features.py not found. Put it in the same folder.")
    exit(1)

# ── V-block labeling thresholds ───────────────────────────────────────────────
# For a 90° v-block (45° faces), nominal contact: normal_z ≈ -0.707
OUTERNESS_THRESH = 0.55   # face must be on the outer surface
NORMAL_Z_MAX     = -0.20  # must point at least somewhat downward
NORMAL_Z_MIN     = -0.97  # exclude faces that point nearly straight down (base)
LATERAL_MIN      = 0.20   # must have a sideways component (excludes end-caps)


def auto_label_vblock(mesh, feature_matrix, debug_name=""):
    """
    Returns indices of triangles that would contact a 90° v-block.

    A face qualifies if:
      - It is on the outer surface (high outerness)
      - Its outward normal points downward at an angle (not straight up/down/sideways)
      - It has a significant lateral component (not an axial end-cap)
    """
    normals   = np.array(mesh.cell_normals)
    outerness = feature_matrix[:, 11]
    lateral   = np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2)

    mask = (
        (normals[:, 2] <  NORMAL_Z_MAX)   &
        (normals[:, 2] >  NORMAL_Z_MIN)   &
        (lateral       >  LATERAL_MIN)    &
        (outerness     >  OUTERNESS_THRESH)
    )

    indices = np.where(mask)[0].astype(int)

    if len(indices) == 0:
        print(f"  WARNING: no clamp faces for {debug_name}")
        print(f"    normal_z  : {normals[:,2].min():.3f} – {normals[:,2].max():.3f}")
        print(f"    outerness : {outerness.min():.3f} – {outerness.max():.3f}")
        print(f"    lateral   : {lateral.min():.3f} – {lateral.max():.3f}")

    return indices


def generate_labels_dict(mesh, clamp_indices):
    labels = {str(i): "unselected" for i in range(mesh.ncells)}
    for idx in clamp_indices:
        labels[str(idx)] = "clamp"
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="raw_vblock_parts",
                        help="Folder containing raw STL files")
    parser.add_argument("--output_dir", default="training_set_vblock",
                        help="Folder to save labeled dataset")
    parser.add_argument("--augment",    type=int, default=3,
                        help="Rotated copies per base part")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stl_files = glob.glob(os.path.join(args.input_dir, "*.stl"))
    if not stl_files:
        print(f"No STL files found in {args.input_dir}")
        exit(1)

    total_expected = len(stl_files) * args.augment
    print(f"Found {len(stl_files)} base parts × {args.augment} augmentations = {total_expected} total.")

    all_labels_data = {}
    total_processed = 0

    for stl_file in stl_files:
        base_name   = os.path.basename(stl_file)
        name_no_ext = os.path.splitext(base_name)[0]

        try:
            original_mesh = Mesh(stl_file)
        except Exception as e:
            print(f"Load failed: {base_name} — {e}")
            continue

        for aug_idx in range(args.augment):
            new_filename = f"{name_no_ext}_aug{aug_idx}.stl"
            save_path    = os.path.join(args.output_dir, new_filename)

            mesh = original_mesh.clone()
            mesh.shift(-np.array(mesh.center_of_mass()))

            rotation_angle = random.uniform(0, 360)
            mesh.rotate_z(rotation_angle)
            mesh.compute_normals()
            mesh.write(save_path)

            try:
                feats, _, _ = features.extract_triangle_features(save_path)
            except Exception as e:
                print(f"  Feature extraction failed: {new_filename} — {e}")
                continue

            clamp_indices = auto_label_vblock(mesh, feats, debug_name=new_filename)

            json_key    = os.path.join(args.output_dir, new_filename).replace("\\", "/")
            labels_dict = generate_labels_dict(mesh, clamp_indices)
            labels_dict["_meta_rotation_z"] = rotation_angle
            all_labels_data[json_key] = labels_dict

            total_processed += 1
            if total_processed % 20 == 0:
                print(f"  {total_processed} / {total_expected} done...")

    out_json = os.path.join(args.output_dir, "all_part_labels.json")
    with open(out_json, "w") as f:
        json.dump(all_labels_data, f, indent=2)

    print("=" * 60)
    print(f"Done. {total_processed} parts labeled.")
    print(f"Labels saved to {out_json}")
    print("=" * 60)
    print("Next step: upload training_set_vblock/ to the cluster and run train_vblock.py")
