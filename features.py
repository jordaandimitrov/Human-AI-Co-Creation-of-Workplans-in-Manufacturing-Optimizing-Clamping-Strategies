import numpy as np
import trimesh
from vedo import Mesh


# --------------------------------------------------------------
#   OUTERNESS – (already in your system)
# --------------------------------------------------------------
def compute_outerness(mesh, centroids):
    try:
        tm_mesh = trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells, process=False)
        hull = tm_mesh.convex_hull
        _, distances, _ = trimesh.proximity.closest_point(hull, centroids)

        b = mesh.bounds()
        max_dim = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        decay_factor = 20.0 / (max_dim + 1e-6)

        outerness = np.exp(-distances * decay_factor)
    except Exception as e:
        print(f"   ⚠️ Outerness Fallback: {e}")
        cm = mesh.center_of_mass()
        dists = np.linalg.norm(centroids - cm, axis=1)
        outerness = dists / (np.max(dists) + 1e-6)

    return outerness.astype(np.float32)



# --------------------------------------------------------------
#   FRAGILITY FEATURE 1 — EDGE RATIO
# --------------------------------------------------------------
def compute_edge_ratio(mesh):
    tris = np.array(mesh.cells)
    pts = mesh.points

    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]

    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)

    max_e = np.maximum(e0, np.maximum(e1, e2))
    min_e = np.minimum(e0, np.minimum(e1, e2))

    ratio = min_e / (max_e + 1e-6)
    edge_fragility = (1 - ratio)  # small/min edge difference → fragile

    return edge_fragility.astype(np.float32)



# --------------------------------------------------------------
#   FRAGILITY FEATURE 2 — CURVATURE (normal variation)
# --------------------------------------------------------------
def compute_curvature(mesh):
    tris = np.array(mesh.cells)
    normals = mesh.cell_normals

    tm = trimesh.Trimesh(vertices=mesh.points, faces=tris, process=False)
    adjacency = tm.face_adjacency

    curvature = np.zeros(len(tris), dtype=np.float32)

    for f0, f1 in adjacency:
        curvature[f0] += np.linalg.norm(normals[f0] - normals[f1])
        curvature[f1] += np.linalg.norm(normals[f0] - normals[f1])

    curvature /= (np.max(curvature) + 1e-6)
    return curvature.astype(np.float32)



# --------------------------------------------------------------
#   FRAGILITY FEATURE 3 — SLENDERNESS
# --------------------------------------------------------------
def compute_slenderness(mesh):
    areas = mesh.area()
    centroids = mesh.cell_centers().points
    cm = mesh.center_of_mass()

    dist = np.linalg.norm(centroids - cm, axis=1)

    norm_dist = dist / (np.max(dist) + 1e-6)
    norm_area = areas / (np.max(areas) + 1e-6)

    slend = norm_dist / (norm_area + 1e-6)
    slend /= (np.max(slend) + 1e-6)

    return slend.astype(np.float32)



# --------------------------------------------------------------
#   FRAGILITY FEATURE 4 — THICKNESS RATIO (thin walls)
# --------------------------------------------------------------
def compute_thickness_ratio(mesh):
    tris = np.array(mesh.cells)
    normals = mesh.cell_normals
    pts = mesh.points

    tm = trimesh.Trimesh(vertices=pts, faces=tris, process=False)
    adjacency = tm.face_adjacency

    thickness = np.zeros(len(tris), dtype=np.float32)

    for f0, f1 in adjacency:
        dot = np.abs(np.dot(normals[f0], normals[f1]))
        thickness[f0] += (1 - dot)
        thickness[f1] += (1 - dot)

    thickness /= (np.max(thickness) + 1e-6)

    return thickness.astype(np.float32)



# --------------------------------------------------------------
#   FULL FEATURE EXTRACTOR
# --------------------------------------------------------------
def extract_triangle_features(stl_path):

    # ---------------------------------------
    # 1. Load and triangulate
    # ---------------------------------------
    mesh = Mesh(stl_path)
    mesh.triangulate()

    points = mesh.points
    tris = np.array(mesh.cells).astype(np.int64)
    normals = mesh.cell_normals
    centroids = mesh.cell_centers().points
    areas = mesh.area()
    part_center = np.array(mesh.center_of_mass(), dtype=np.float32)
    dist_to_center = np.linalg.norm(centroids - part_center, axis=1)

    # ---------------------------------------
    # 2. Normalization factors
    # ---------------------------------------
    max_area = np.max(areas) if np.max(areas) > 1e-6 else 1.0
    max_dist = np.max(dist_to_center) if np.max(dist_to_center) > 1e-6 else 1.0

    rel_pos = centroids - part_center
    radial_dist = np.linalg.norm(rel_pos[:, :2], axis=1)
    max_rad = np.max(radial_dist) if np.max(radial_dist) > 1e-6 else 1.0

    z = centroids[:, 2]
    min_z, max_z = np.min(z), np.max(z)
    height_range = max_z - min_z if (max_z - min_z) > 1e-6 else 1.0
    rel_height = (z - min_z) / height_range

    v0 = points[tris[:, 0]]
    v1 = points[tris[:, 1]]
    v2 = points[tris[:, 2]]

    edge_a = np.linalg.norm(v1 - v0, axis=1)
    edge_b = np.linalg.norm(v2 - v1, axis=1)
    edge_c = np.linalg.norm(v0 - v2, axis=1)

    max_edge = max(np.max(edge_a), np.max(edge_b), np.max(edge_c))
    if max_edge < 1e-6:
        max_edge = 1.0

    # ---------------------------------------
    # 3. Existing features
    # ---------------------------------------
    outerness = compute_outerness(mesh, centroids)

    # ---------------------------------------
    # 4. Ray Score (your existing implementation)
    # ---------------------------------------
    tm = trimesh.Trimesh(vertices=points, faces=tris, process=False)
    opposite_quality = np.zeros(len(tris), dtype=np.float32)

    verticality = 1.0 - np.abs(normals[:, 2])
    is_elite = (verticality > 0.8) & (outerness > 0.85)

    candidate_indices = np.where(is_elite)[0]

    if len(candidate_indices) > 0:
        c_norms = normals[candidate_indices]
        c_cents = centroids[candidate_indices]

        world_z = np.array([0, 0, 1])
        tangent = np.cross(c_norms, world_z)
        tangent /= (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-6)
        bitangent = np.cross(c_norms, tangent)

        offset = 3.0
        origins_list = [
            c_cents,
            c_cents + tangent * offset,
            c_cents - tangent * offset,
            c_cents + bitangent * offset,
            c_cents - bitangent * offset,
        ]

        accumulator = np.zeros(len(candidate_indices), dtype=np.float32)

        for origins in origins_list:
            start = origins - c_norms * 0.01

            try:
                res = tm.ray.intersects_id(
                    ray_origins=start,
                    ray_directions=-c_norms,
                    multiple_hits=True,
                    return_locations=True
                )
                if len(res) == 3:
                    idx_tri, idx_ray, locs = res
                elif len(res) == 2:
                    idx_tri, idx_ray = res
                    locs = None
                else:
                    idx_tri, idx_ray, locs = [], [], None
            except:
                idx_tri, idx_ray, locs = [], [], None

            if len(idx_ray) == 0:
                continue

            if locs is not None:
                d = np.linalg.norm(locs - start[idx_ray], axis=1)
                mask = d < 500.0
                idx_tri = idx_tri[mask]
                idx_ray = idx_ray[mask]

            if len(idx_ray) == 0:
                continue

            target_norm = normals[idx_tri]
            target_out = outerness[idx_tri]
            shooter_norms = c_norms[idx_ray]

            align = np.einsum("ij,ij->i", shooter_norms, target_norm)
            t_vert = 1.0 - np.abs(target_norm[:, 2])

            valid = (align < -0.85) & (t_vert > 0.85) & (target_out > 0.85)

            successful = np.unique(idx_ray[valid])
            accumulator[successful] += 0.2

        opposite_quality[candidate_indices] = np.clip(accumulator, 0, 1.0)

    # --------------------------------------------------
    # 5. NEW FRAGILITY FEATURES
    # --------------------------------------------------
    edge_frag = compute_edge_ratio(mesh)
    curvature = compute_curvature(mesh)
    slender = compute_slenderness(mesh)
    thickness = compute_thickness_ratio(mesh)

    # --------------------------------------------------
    # 6. Assemble Full Feature Vector (17 features)
    # --------------------------------------------------
    feats = np.zeros((len(tris), 17), dtype=np.float32)

    # OLD (13)
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
    feats[:, 13] = cylindricity        # <-- NEW FEATURE

    # NEW (4)
    feats[:, 13] = edge_frag
    feats[:, 14] = curvature
    feats[:, 15] = slender
    feats[:, 16] = thickness

    return feats, tris, points
