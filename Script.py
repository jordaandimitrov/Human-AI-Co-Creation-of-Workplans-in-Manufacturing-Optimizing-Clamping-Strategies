import torch
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# 1. Load STL mesh
# -----------------------------
mesh = o3d.io.read_triangle_mesh(r"C:\Users\jorda\Desktop\Unif\MA3\Thesis\Data\Random box.stl")
mesh.compute_vertex_normals()
print("Mesh has", np.asarray(mesh.vertices).shape[0], "vertices")

# -----------------------------
# 2. Sample a point cloud from the mesh surface
# -----------------------------
pcd = mesh.sample_points_uniformly(number_of_points=10000)
points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)
num_points = len(points)
print("Point cloud shape:", points.shape)

# -----------------------------
# 3. Start with all points gray
# -----------------------------
colors = np.tile([0.7, 0.7, 0.7], (num_points, 1))

# -----------------------------
# 4. Define faces to highlight
# -----------------------------
faces = [
    {"normal": np.array([1, 0, 0]), "color": [1.0, 0.0, 0.0]},   # right face, red
    {"normal": np.array([-1, 0, 0]), "color": [1.0, 0.0, 0.0]},   # top face, green
    {"normal": np.array([0, 1, 0]), "color": [0.0, 1.0, 0.0]},
{"normal": np.array([0, -1, 0]), "color": [0.0, 1.0, 0.0]}
]

threshold = 0.95  # normal alignment threshold

# -----------------------------
# 5. Highlight each face
# -----------------------------
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(points)

for face in faces:
    desired_normal = face["normal"]
    face_color = face["color"]

    # Select points whose normals align with the face
    dot_products = normals @ desired_normal
    aligned_mask = dot_products > threshold
    aligned_points = points[aligned_mask]

    # Compute geometric center
    face_center = aligned_points.mean(axis=0)

    # Find 10 nearest neighbors to the center
    _, idxs = nbrs.kneighbors([face_center])

    # Color these points
    colors[idxs[0]] = face_color

# Assign colors back to point cloud
pcd.colors = o3d.utility.Vector3dVector(colors)

# -----------------------------
# 6. Convert to PyTorch tensor
# -----------------------------
points_tensor = torch.tensor(points, dtype=torch.float32)
normals_tensor = torch.tensor(normals, dtype=torch.float32)
print("Torch tensor:", points_tensor.shape, points_tensor.dtype)

# -----------------------------
# 7. Example dummy labels
# -----------------------------
labels = torch.zeros(points_tensor.shape[0], dtype=torch.long)
labels[:200] = 1  # just demo
print("Labels shape:", labels.shape, "Unique:", labels.unique())

# -----------------------------
# 8. Visualize
# -----------------------------
o3d.visualization.draw_geometries([pcd])
