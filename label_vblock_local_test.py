"""
Local test for v-block labeling on a handful of cylindrical parts.

Loads STL files from training_set_cylinders/ (or --input_dir), applies the
90° v-block labeling criteria, prints a per-part summary, and optionally
opens a vedo visualizer showing clamp faces in red and the rest in gray.

Usage:
    python label_vblock_local_test.py
    python label_vblock_local_test.py --input_dir my_parts --no_viz
    python label_vblock_local_test.py --max_parts 3
"""

import os
import glob
import argparse
import numpy as np
from vedo import Mesh, Plotter, Text2D

try:
    import features
except ImportError:
    print("ERROR: features.py not found. Run from the project root.")
    exit(1)

from label_vblock_headless import auto_label_vblock


def _label_with_stats(mesh, feats, path):
    clamp_idx = auto_label_vblock(mesh, feats, debug_name=os.path.basename(path))
    total = mesh.ncells
    pct   = 100 * len(clamp_idx) / total if total else 0
    print(f"  {os.path.basename(path):<45} "
          f"triangles={total:5d}  clamp={len(clamp_idx):4d}  ({pct:5.1f}%)")
    return clamp_idx


def visualize(mesh, clamp_idx, title=""):
    n = mesh.ncells
    colors = np.full((n, 3), [160, 160, 160], dtype=np.uint8)
    if len(clamp_idx):
        colors[clamp_idx] = [220, 50, 50]

    m = mesh.clone()
    m.cellcolors = colors

    plt = Plotter(title=title, axes=1)
    plt.add(m)
    plt.add(Text2D(f"{title}\nclamp={len(clamp_idx)} / {n} triangles",
                   pos="top-left", s=0.8, c="white"))
    plt.show(interactive=True)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="training_set_cylinders",
                        help="Folder with STL files to test")
    parser.add_argument("--max_parts", type=int, default=0,
                        help="Limit to first N parts (0 = all)")
    parser.add_argument("--no_viz",   action="store_true",
                        help="Skip the visual window, just print stats")
    args = parser.parse_args()

    stl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.stl")))
    if not stl_files:
        print(f"No STL files found in {args.input_dir}")
        exit(1)

    if args.max_parts > 0:
        stl_files = stl_files[:args.max_parts]

    print(f"Testing v-block labeling on {len(stl_files)} parts from '{args.input_dir}'\n")
    print(f"  {'file':<45} {'triangles':>10}  {'clamp':>6}  {'%':>6}")
    print("  " + "-" * 72)

    for path in stl_files:
        try:
            mesh = Mesh(path)
        except Exception as e:
            print(f"  Load failed: {path} — {e}")
            continue

        mesh.shift(-np.array(mesh.center_of_mass()))
        mesh.compute_normals()

        try:
            feats, _, _ = features.extract_triangle_features(path)
        except Exception as e:
            print(f"  Feature extraction failed: {path} — {e}")
            continue

        clamp_idx = _label_with_stats(mesh, feats, path)

        if not args.no_viz:
            visualize(mesh, clamp_idx, title=os.path.basename(path))

    print("\nDone. Red faces = v-block contact (clamp). Gray = not selected.")
    print("If the labeling looks correct, run label_vblock_headless.py for the full dataset.")
