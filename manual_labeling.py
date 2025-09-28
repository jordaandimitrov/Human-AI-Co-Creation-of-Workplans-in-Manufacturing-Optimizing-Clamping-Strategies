import torch
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import re

# -----------------------------
# 0. Open file explorer to select STL file
# -----------------------------
Tk().withdraw()  # hide the root Tk window
file_path = askopenfilename(title="Select an STL file", filetypes=[("STL files", "*.stl")])

if not file_path:
    raise ValueError("No STL file selected!")

print("Selected STL file:", file_path)

# Extract base filename without extension
base_name = os.path.splitext(os.path.basename(file_path))[0]

# -----------------------------
# 1. Load mesh and sample points
# -----------------------------
mesh = o3d.io.read_triangle_mesh(file_path)
mesh.compute_vertex_normals()
pcd = mesh.sample_points_uniformly(number_of_points=50000)
points = np.asarray(pcd.points)

# -----------------------------
# 2. Set all points to gray for picking
# -----------------------------
gray_color = [0.7, 0.7, 0.7]
colors = np.tile(gray_color, (len(points), 1))
pcd.colors = o3d.utility.Vector3dVector(colors)

# -----------------------------
# 3. Manual point picking
# -----------------------------
print("Pick seed points in the window (multiple allowed), then press 'q' to close.")
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
vis.add_geometry(pcd)
vis.run()  # wait for user input
picked_indices = vis.get_picked_points()  # returns a list of indices
vis.destroy_window()

if not picked_indices:
    raise ValueError("No points were picked! Pick at least one seed point.")

print(f"Picked seed point indices: {picked_indices}")

# -----------------------------
# 4. Find k nearest neighbors for each seed
# -----------------------------
k = 20
nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)

all_selected_indices = set()  # use a set to avoid duplicates

for seed_idx in picked_indices:
    distances, indices = nbrs.kneighbors([points[seed_idx]])
    all_selected_indices.update(indices[0])  # add neighbors to the set

selected_indices = np.array(list(all_selected_indices))

# -----------------------------
# 5. Assign labels
# -----------------------------
labels = np.zeros(len(points), dtype=np.int64)
labels[selected_indices] = 1  # mark all neighbors of all seeds

# -----------------------------
# 6. Visualize labeled points
# -----------------------------
colors = np.tile(gray_color, (len(points), 1))  # start all gray
colors[selected_indices] = [1, 0, 0]           # highlight selected points in red
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])

# -----------------------------
# 7. Save dataset with original filename + incremental number
# -----------------------------
os.makedirs("dataset", exist_ok=True)

existing_files = os.listdir("dataset")
pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.pt")
numbers = [int(pattern.match(f).group(1)) for f in existing_files if pattern.match(f)]
next_number = max(numbers, default=0) + 1

save_path = f"dataset/{base_name}_{next_number:03d}.pt"
torch.save({
    "points": torch.tensor(points, dtype=torch.float32),
    "labels": torch.tensor(labels, dtype=torch.long)
}, save_path)

print(f"✅ Saved {save_path} with gray base and selected points highlighted")
