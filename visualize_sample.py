import torch
import open3d as o3d
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# -----------------------------
# 0. Open file explorer to select a .pt file
# -----------------------------
Tk().withdraw()  # hide root window
file_path = askopenfilename(title="Select a .pt sample", filetypes=[("PyTorch files", "*.pt")])

if not file_path:
    raise ValueError("No file selected!")

print("Selected file:", file_path)

# -----------------------------
# 1. Load the .pt file
# -----------------------------
data = torch.load(file_path)

points = data["points"].numpy()   # (N,3)
labels = data["labels"].numpy()   # (N,)

num_points = len(points)
print("Loaded sample with", num_points, "points")

# -----------------------------
# 2. Assign colors based on labels
# -----------------------------
colors = np.tile([0.7, 0.7, 0.7], (num_points, 1))  # default gray
colors[labels == 1] = [1.0, 0.0, 0.0]               # red for highlighted

# -----------------------------
# 3. Create Open3D point cloud
# -----------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# -----------------------------
# 4. Visualize
# -----------------------------
o3d.visualization.draw_geometries([pcd])
