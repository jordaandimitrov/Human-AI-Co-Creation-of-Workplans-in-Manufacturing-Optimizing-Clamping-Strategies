"""
Generate parametric cylindrical STL parts for v-block fixture training.

Outputs raw STL files to raw_vblock_parts/.
Run label_vblock_headless.py afterwards to label and augment them.

Usage:
    python generate_vblock_parts.py
"""

import os
import numpy as np
import trimesh

OUTPUT_DIR  = "raw_vblock_parts"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

parts = []   # list of (name_prefix, trimesh.Trimesh)


# ── 1. Smooth cylinders ──────────────────────────────────────────────────────
# Wide range of radii and heights; sections=32/48 gives smooth surfaces.
for radius in [8, 12, 18, 25, 35, 50]:
    for height in [15, 30, 50, 80, 120]:
        for sections in [32, 48]:
            m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
            parts.append((f"cyl_r{radius}_h{height}_s{sections}", m))


# ── 2. Faceted cylinders (low section count = polygonal prism) ───────────────
# Hexagonal (6), octagonal (8), dodecagonal (12) bars — common stock shapes.
for sides in [6, 8, 12]:
    for radius in [10, 20, 35]:
        for height in [30, 60, 100]:
            m = trimesh.creation.cylinder(radius=radius, height=height, sections=sides)
            parts.append((f"prism_s{sides}_r{radius}_h{height}", m))


# ── 3. Hollow cylinders / pipes ───────────────────────────────────────────────
for r_outer, r_inner in [(20, 12), (30, 20), (40, 28), (50, 36), (25, 15), (35, 22), (60, 45)]:
    for height in [30, 60, 100]:
        m = trimesh.creation.annulus(r_min=r_inner, r_max=r_outer, height=height, sections=32)
        parts.append((f"pipe_ro{r_outer}_ri{r_inner}_h{height}", m))


# ── 4. Short discs / flanges ─────────────────────────────────────────────────
for radius in [25, 35, 50, 70]:
    for height in [8, 12, 20]:
        m = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
        parts.append((f"disc_r{radius}_h{height}", m))


# ── 5. Long thin rods ────────────────────────────────────────────────────────
for radius in [5, 8, 12, 16]:
    for height in [80, 120, 180]:
        m = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
        parts.append((f"rod_r{radius}_h{height}", m))


# ── 6. Stepped cylinders (two stacked cylinders) ─────────────────────────────
# Constructed by concatenating two cylinder meshes — not watertight at the
# junction, but sufficient for feature extraction and labeling.
step_configs = [
    (25, 15, 40, 40),   # (r_bottom, r_top, h_bottom, h_top)
    (35, 20, 30, 50),
    (30, 18, 50, 30),
    (40, 25, 40, 60),
    (20, 12, 30, 40),
    (50, 30, 20, 60),
]
for r_bot, r_top, h_bot, h_top in step_configs:
    cyl_bot = trimesh.creation.cylinder(radius=r_bot, height=h_bot, sections=32)
    cyl_top = trimesh.creation.cylinder(radius=r_top, height=h_top, sections=32)
    cyl_bot.apply_translation([0, 0, -h_top / 2])
    cyl_top.apply_translation([0, 0,  h_bot / 2])
    m = trimesh.util.concatenate([cyl_bot, cyl_top])
    parts.append((f"stepped_rb{r_bot}_rt{r_top}_hb{h_bot}_ht{h_top}", m))


# ── Save all parts ────────────────────────────────────────────────────────────
print(f"Saving {len(parts)} base parts to {OUTPUT_DIR}/")

for name, mesh in parts:
    # Center at origin
    mesh.apply_translation(-mesh.centroid)
    path = os.path.join(OUTPUT_DIR, f"{name}.stl")
    mesh.export(path)

print(f"Done. {len(parts)} STL files saved.")
print(f"Next step: run label_vblock_headless.py to label and augment them.")
