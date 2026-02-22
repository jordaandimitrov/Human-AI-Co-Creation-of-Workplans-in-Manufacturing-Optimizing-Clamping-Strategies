import numpy as np
import trimesh
from vedo import Mesh



def compute_cylindricity(centroids, normals, outerness_map, radius=None, outerness_thresh=0.85, max_cv=0.15):
    """
    Compute a strict Z-axis cylinder score per triangle.
    Triangles must:
      1. Be outer faces (outerness >= threshold)
      2. Have normals roughly pointing radially outward
      3. Be mostly vertical (normals not along Z)
      4. Be close to expected radius (optional)
      5. Belong to a shape with a mostly constant radius (rejects boxes)

    Parameters:
        centroids       : (N,3) array of triangle centroids
        normals         : (N,3) array of triangle normals
        outerness_map   : (N,) array of outerness per triangle
        radius          : optional expected cylinder radius
        outerness_thresh: minimum outerness to consider a triangle
        max_cv          : maximum allowed variation in radius (Standard Dev / Mean) to not be penalized

    Returns:
        cylindricity : (N,) array in [0,1], 1 = perfect cylinder side
    """
    N = len(centroids)
    cylindricity = np.zeros(N, dtype=np.float32)

    # Step 1: Outerness filter
    eligible = outerness_map >= outerness_thresh
    if not np.any(eligible):
        return cylindricity  # nothing to score

    # Step 2: Radial vectors from Z-axis
    xy = centroids[:, :2]
    center_xy = xy.mean(axis=0)
    radial_vec = xy - center_xy
    radial_len = np.linalg.norm(radial_vec, axis=1) + 1e-6
    radial_unit = radial_vec / radial_len[:, None]

    # Step 3: Normals XY projection and verticality
    normals_xy = normals[:, :2]
    norm_len = np.linalg.norm(normals_xy, axis=1) + 1e-6
    normals_xy_unit = normals_xy / norm_len[:, None]

    verticality = 1.0 - np.abs(normals[:, 2])  # 1 = mostly vertical, 0 = along Z

    # Step 4: Radial alignment (cosine between normal XY and radial vector)
    radial_alignment = np.einsum("ij,ij->i", normals_xy_unit, radial_unit)
    radial_alignment = np.clip(radial_alignment, 0, 1)

    # FIX 1: Exponentiate to harshly penalize the flat, off-center faces of a box
    radial_alignment = radial_alignment ** 4

    # FIX 2: Box-penalty based on radius variance
    # A cylinder has a constant radius. A box's radius fluctuates.
    candidate_mask = eligible & (verticality > 0.8)
    box_penalty = 1.0

    if np.any(candidate_mask):
        valid_radii = radial_len[candidate_mask]
        radius_mean = np.mean(valid_radii)
        radius_std = np.std(valid_radii)
        cv = radius_std / (radius_mean + 1e-6)  # Coefficient of Variation

        # If the CV is high (radius fluctuates a lot), scale down the score.
        # A perfect cylinder has CV ~ 0. A square box has CV ~ 0.12 - 0.15.
        if cv > (max_cv * 0.33):
            # Smoothly drop the penalty to 0 as it approaches max_cv
            box_penalty = np.clip(1.0 - (cv - (max_cv * 0.33)) / (max_cv * 0.67), 0.0, 1.0)

    # Step 5: Optional radius check
    if radius is not None:
        radial_error = np.abs(radial_len - radius) / (radius + 1e-6)
        radius_score = np.exp(-5 * radial_error)
    else:
        radius_score = np.ones(N, dtype=np.float32)

    # Step 6: Combine scores for eligible triangles
    score = radial_alignment * verticality * radius_score * box_penalty
    score[~eligible] = 0.0  # zero-out ineligible triangles

    # Step 7: Clip final score
    cylindricity = np.clip(score, 0, 1).astype(np.float32)

    return cylindricity

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


def extract_triangle_features(stl_path):
    # 1. Load
    mesh = Mesh(stl_path)
    mesh.triangulate()
    points = mesh.points
    tris = np.array(mesh.cells).astype(np.int64)
    normals = mesh.cell_normals
    centroids = mesh.cell_centers().points
    areas = mesh.area()
    part_center = np.array(mesh.center_of_mass(), dtype=np.float32)
    dist_to_center = np.linalg.norm(centroids - part_center, axis=1)

    # 2. Normalization Factors
    max_area = np.max(areas) if np.max(areas) > 1e-6 else 1.0
    max_dist = np.max(dist_to_center) if np.max(dist_to_center) > 1e-6 else 1.0
    rel_pos = centroids - part_center
    radial_dist = np.linalg.norm(rel_pos[:, 0:2], axis=1)
    max_rad = np.max(radial_dist) if np.max(radial_dist) > 1e-6 else 1.0
    z_values = centroids[:, 2]
    min_z, max_z = np.min(z_values), np.max(z_values)
    height_range = max_z - min_z if (max_z - min_z) > 1e-6 else 1.0
    rel_height = (z_values - min_z) / height_range

    v0 = points[tris[:, 0]]
    v1 = points[tris[:, 1]]
    v2 = points[tris[:, 2]]
    edge_a = np.linalg.norm(v1 - v0, axis=1)
    edge_b = np.linalg.norm(v2 - v1, axis=1)
    edge_c = np.linalg.norm(v0 - v2, axis=1)
    max_edge = max(np.max(edge_a), np.max(edge_b), np.max(edge_c))
    if max_edge < 1e-6: max_edge = 1.0

    # 3. Features
    outerness = compute_outerness(mesh, centroids)
    opposite_quality = np.zeros(len(tris), dtype=np.float32)
    tm = trimesh.Trimesh(vertices=points, faces=tris, process=False)
    verticality = 1.0 - np.abs(normals[:, 2])
    is_elite_shooter = (verticality > 0.8) & (outerness > 0.85)
    candidate_indices = np.where(is_elite_shooter)[0]

    if len(candidate_indices) > 0:
        c_norms = normals[candidate_indices]
        c_cents = centroids[candidate_indices]

        # --- RESTORED 5 RAYS ---
        world_z = np.array([0, 0, 1])
        tangent = np.cross(c_norms, world_z)
        norms = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-6
        tangent = tangent / norms
        bitangent = np.cross(c_norms, tangent)
        offset_dist = 3.0

        origins_list = [
            c_cents,
            c_cents + (tangent * offset_dist),
            c_cents - (tangent * offset_dist),
            c_cents + (bitangent * offset_dist),
            c_cents - (bitangent * offset_dist)
        ]

        hit_accumulator = np.zeros(len(candidate_indices), dtype=np.float32)

        for origins in origins_list:
            start_pts = origins - (c_norms * 0.01)
            vectors = -c_norms

            try:
                # Robust Call
                res = tm.ray.intersects_id(
                    ray_origins=start_pts,
                    ray_directions=vectors,
                    multiple_hits=True,
                    return_locations=True
                )
                if len(res) == 3:
                    index_tri, index_ray, locations = res
                elif len(res) == 2:
                    index_tri, index_ray = res
                    locations = None
                else:
                    index_tri, index_ray, locations = [], [], None
            except Exception:
                index_tri, index_ray, locations = [], [], None

            if len(index_ray) > 0:
                # Manual Max Distance Filter (Replaces max_d)
                if locations is not None:
                    origins_for_hits = start_pts[index_ray]
                    dists = np.linalg.norm(locations - origins_for_hits, axis=1)
                    dist_mask = dists < 500.0
                    index_tri = index_tri[dist_mask]
                    index_ray = index_ray[dist_mask]

                if len(index_ray) > 0:
                    target_normals = normals[index_tri]
                    target_out = outerness[index_tri]
                    shooter_normals = c_norms[index_ray]

                    align = np.einsum('ij,ij->i', shooter_normals, target_normals)
                    t_vert = 1.0 - np.abs(target_normals[:, 2])
                    is_valid_hit = (align < -0.85) & (t_vert > 0.85) & (target_out > 0.85)

                    successful_rays = index_ray[is_valid_hit]
                    unique_successful_rays = np.unique(successful_rays)

                    # RESTORED 0.2 WEIGHT
                    hit_accumulator[unique_successful_rays] += 0.2

        opposite_quality[candidate_indices] = np.clip(hit_accumulator, 0, 1.0)

    # ---------------------------------------------------------
    # INSERTED: CYLINDRICITY (NOTHING ELSE MODIFIED)
    # ---------------------------------------------------------
    radial_vec = np.column_stack([
        centroids[:, 0] - part_center[0],
        centroids[:, 1] - part_center[1],
        np.zeros(len(centroids))
    ])
    radial_norm = np.linalg.norm(radial_vec, axis=1) + 1e-6
    radial_unit = radial_vec / radial_norm[:, None]



    cylindricity = compute_cylindricity(centroids, normals, outerness)
    #cylindricity = np.abs(np.einsum("ij,ij->i", radial_unit, normals)).astype(np.float32)
    # ---------------------------------------------------------

    # 4. Assembly  (now 14 features)
    feats = np.zeros((len(tris), 14), dtype=np.float32)
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

    return feats, tris, points
