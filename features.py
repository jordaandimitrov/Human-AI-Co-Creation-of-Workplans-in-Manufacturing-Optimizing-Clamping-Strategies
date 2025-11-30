# features.py
import numpy as np
import trimesh
from vedo import Mesh


def compute_outerness(mesh, centroids):
    """
    Calculates 'Outerness' using the Convex Hull (Shrink Wrap) method.
    """
    try:
        tm_mesh = trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells, process=False)
        hull = tm_mesh.convex_hull
        _, distances, _ = trimesh.proximity.closest_point(hull, centroids)

        b = mesh.bounds()
        max_dim = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        decay_factor = 20.0 / (max_dim + 1e-6)

        outerness = np.exp(-distances * decay_factor)

    except Exception as e:
        cm = mesh.center_of_mass()
        dists = np.linalg.norm(centroids - cm, axis=1)
        outerness = dists / (np.max(dists) + 1e-6)

    return outerness.astype(np.float32)


def extract_triangle_features(stl_path):
    # 1. Load Mesh
    mesh = Mesh(stl_path)
    mesh.triangulate()

    points = mesh.points
    tris = np.array(mesh.cells).astype(np.int64)
    normals = mesh.cell_normals
    centroids = mesh.cell_centers().points
    areas = mesh.area()
    part_center = np.array(mesh.center_of_mass(), dtype=np.float32)
    dist_to_center = np.linalg.norm(centroids - part_center, axis=1)

    # --- A. STANDARD FEATURES ---
    rel_pos = centroids - part_center
    radial_dist = np.linalg.norm(rel_pos[:, 0:2], axis=1)

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

    # --- B. IMPORTED OUTERNESS ---
    outerness = compute_outerness(mesh, centroids)

    # --- C. X-RAY CASTING (Penetrating Holes) ---
    opposite_quality = np.zeros(len(tris), dtype=np.float32)
    tm = trimesh.Trimesh(vertices=points, faces=tris, process=False)

    verticality = 1.0 - np.abs(normals[:, 2])

    # ELITE SHOOTERS: Must be Vertical AND Outer
    is_elite_shooter = (verticality > 0.8) & (outerness > 0.85)

    candidate_indices = np.where(is_elite_shooter)[0]

    if len(candidate_indices) > 0:
        c_norms = normals[candidate_indices]
        c_cents = centroids[candidate_indices]

        # Jitter vectors
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
                # CRITICAL CHANGE: multiple_hits=True
                # This returns ALL triangles the ray pierces, not just the first one.
                # Returns: index_tri (face hit), index_ray (which ray), locations
                index_tri, index_ray, _ = tm.ray.intersects_id(
                    ray_origins=start_pts,
                    ray_directions=vectors,
                    multiple_hits=True,
                    max_d=500.0
                )
            except:
                index_tri, index_ray = [], []

            if len(index_ray) > 0:
                # We now have a list of ALL hits.
                # We need to filter them to find if *any* hit was valid.

                # 1. Get properties of the faces we hit
                target_normals = normals[index_tri]
                target_out = outerness[index_tri]

                # 2. Get properties of the shooters
                # index_ray maps back to our subset 0..N
                shooter_normals = c_norms[index_ray]

                # 3. Check Validity for EVERY hit
                # Alignment
                align = np.einsum('ij,ij->i', shooter_normals, target_normals)
                # Verticality
                t_vert = 1.0 - np.abs(target_normals[:, 2])
                # Outerness (The Filter)
                t_is_outer = target_out > 0.85

                # A hit is valid if: Opposing Normal + Vertical + Outer
                is_valid_hit = (align < -0.85) & (t_vert > 0.85) & t_is_outer

                # 4. Filter: Keep only the ray indices that had a VALID hit
                successful_rays = index_ray[is_valid_hit]

                # 5. Deduplicate
                # If a ray hits 2 valid walls (rare), it still counts as 1 success.
                unique_successful_rays = np.unique(successful_rays)

                # 6. Accumulate
                hit_accumulator[unique_successful_rays] += 0.2

        opposite_quality[candidate_indices] = hit_accumulator

    # --- D. ASSEMBLY ---
    max_area = np.max(areas) if np.max(areas) > 1e-6 else 1.0
    max_dist = np.max(dist_to_center) if np.max(dist_to_center) > 1e-6 else 1.0
    max_rad = np.max(radial_dist) if np.max(radial_dist) > 1e-6 else 1.0

    feats = np.zeros((len(tris), 13), dtype=np.float32)
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

    return feats, tris, points