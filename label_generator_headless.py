import os
import glob
import json
import random
import argparse
import numpy as np
from vedo import Mesh

# --- IMPORT SHARED FEATURES ---
try:
    import features
except ImportError:
    print("❌ ERROR: 'features.py' not found. Please ensure it is in the same folder.")
    exit(1)

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


def auto_label_physically(mesh, feature_matrix, debug_name=""):
    """
    Uses robust features (Ray Score) with detailed debug prints.
    """
    global Z_CLAMP_HEIGHT, RAY_SCORE_THRESHOLD

    centroids = mesh.cell_centers().points
    min_z, max_z = get_z_bounds(mesh)
    Z_THRESHOLD_BOTTOM = min_z + Z_CLAMP_HEIGHT

    # Col 12 = Ray Score, Col 11 = Outerness
    ray_scores = feature_matrix[:, 12]
    outerness = feature_matrix[:, 11]

    # 1. Height Check
    is_bottom = (centroids[:, 2] <= Z_THRESHOLD_BOTTOM)
    count_bottom = np.sum(is_bottom)

    # 2. Physics Check
    has_partner = (ray_scores > RAY_SCORE_THRESHOLD)
    count_partner = np.sum(has_partner)

    # 3. Combine
    valid_mask = is_bottom & has_partner
    final_clamp_indices = np.where(valid_mask)[0].astype(int)

    # --- DIAGNOSTIC LOG (Only print if result is weird or for first few items) ---
    if len(final_clamp_indices) == 0:
        print(f"\n⚠️  [DEBUG: {debug_name}] NO CLAMPS SELECTED!")
        print(f"    - Z-Range of Part: {min_z:.2f} to {max_z:.2f}")
        print(f"    - Z-Threshold: <= {Z_THRESHOLD_BOTTOM:.2f}")
        print(f"    - Faces in Bottom Zone: {count_bottom}")
        print(f"    - Faces with High Ray Score (> {RAY_SCORE_THRESHOLD}): {count_partner}")
        print(f"    - Max Ray Score Found: {np.max(ray_scores):.4f}")
        print(f"    - Mean Outerness: {np.mean(outerness):.4f}")

        if count_partner == 0:
            print("    ❌ ROOT CAUSE: Ray Caster found 0 valid pairs.")
            print("       Check: 1. Features.py 'robust unpack' logic.")
            print("       Check: 2. Normals orientation (Inside/Outside).")
        elif count_bottom == 0:
            print("    ❌ ROOT CAUSE: Height Check failed. Part might be floating.")
        else:
            print("    ❌ ROOT CAUSE: Intersection is empty. High Score faces are not at the bottom.")

    return final_clamp_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Dataset Generator")
    parser.add_argument("--input_dir", type=str, default="raw_parts", help="Folder with source STLs")
    parser.add_argument("--output_dir", type=str, default="training_set", help="Folder to save dataset")
    parser.add_argument("--augment", type=int, default=1, help="Rotated copies per file")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_json_path = os.path.join(args.output_dir, 'all_part_labels.json')
    search_path = os.path.join(args.input_dir, "*.stl")
    stl_files = glob.glob(search_path)

    if not stl_files:
        print(f"❌ No STL files found in {args.input_dir}")
        exit(1)

    print(f"Found {len(stl_files)} files. Processing...")

    all_labels_data = {}
    total_processed = 0

    for file_idx, stl_file in enumerate(stl_files):
        base_name = os.path.basename(stl_file)
        name_no_ext = os.path.splitext(base_name)[0]

        try:
            original_mesh = Mesh(stl_file)
        except Exception as e:
            print(f"⚠️ Load failed: {base_name} - {e}")
            continue

        for aug_idx in range(args.augment):
            new_filename = f"{name_no_ext}_aug{aug_idx}.stl" if args.augment > 1 else base_name
            save_path = os.path.join(args.output_dir, new_filename)

            mesh = original_mesh.clone()

            # Center & Rotate
            cm = mesh.center_of_mass()
            mesh.shift(-cm)

            rotation_angle = random.uniform(0, 360)
            mesh.rotate_z(rotation_angle)
            mesh.compute_normals()

            # Save Geometry
            mesh.write(save_path)

            # Extract
            try:
                feats, _, _ = features.extract_triangle_features(save_path)
            except Exception as e:
                print(f"   ❌ Extract failed: {new_filename} - {e}")
                continue

            # Label (Pass filename for debugging context)
            final_indices = auto_label_physically(mesh, feats, debug_name=new_filename)

            # Save Colors
            num_faces = mesh.ncells
            colors = np.full((num_faces, 3), 200, dtype=np.uint8)
            if len(final_indices) > 0:
                colors[final_indices] = [255, 0, 0]
            mesh.cellcolors = colors
            mesh.write(save_path)

            # Metadata
            json_key = os.path.join(args.output_dir, new_filename).replace("\\", "/")
            labels_dict = generate_labels_dict(mesh, final_indices)
            labels_dict["_meta_rotation_z"] = rotation_angle
            all_labels_data[json_key] = labels_dict

            total_processed += 1
            if total_processed % 10 == 0:
                print(f"   Processed {total_processed} items...")

    with open(output_json_path, 'w') as f:
        json.dump(all_labels_data, f, indent=2)

    print("=" * 60)
    print(f"✅ COMPLETE. Total: {total_processed}")
    print("=" * 60)