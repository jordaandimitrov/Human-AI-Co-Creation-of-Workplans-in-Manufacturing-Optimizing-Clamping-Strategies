from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import numpy as np
from vedo import Mesh, Plotter
import os

# --------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------
# NEW VALUE: Now set to 15.0 units (15cm) for the bottom clamping region.
Z_CLAMP_HEIGHT = 15.0
# Tolerance for the Y-Boundary check. Adjust based on expected wall thickness.
Y_TOLERANCE = 2.0


# --------------------------------------------------------

def get_z_bounds(mesh):
    """Finds the lowest and highest Z coordinate of any vertex in the mesh."""
    # mesh.points gives all vertex coordinates (Nx3 numpy array)
    vertices = mesh.points
    # Z-coordinates are in the third column (index 2)
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    return min_z, max_z


def filter_outside_faces(mesh):
    """
    Filters faces by ensuring the mesh is manifold and extracts only the external surface.
    (Currently TEMPORARILY DISABLED to confirm Z and Y filters work).
    """

    print("Starting surface extraction for visibility...")

    # For now, TEMPORARILY disable the ray-casting filter by returning ALL indices
    num_faces = mesh.ncells
    all_indices = np.arange(num_faces).astype(int)

    print(f"Filter 1 (Visibility): {num_faces} faces remain (TEMPORARILY SKIPPED).")
    return all_indices


def color_faces_geometrically(mesh, outside_indices):
    """
    Colors the faces based on three combined geometric rules:
    1. Z-Position (Clamping Region: ONLY the lowest 15cm)
    2. Y-Orientation (Normal Dominance)
    3. Y-Boundary Check (Excluding internal slot walls)
    4. Visibility (Ray Casting result from outside_indices)
    """

    # --- Parameters ---
    # Fetch global Z_CLAMP_HEIGHT (now 15.0) and Y_TOLERANCE (2.0)
    global Z_CLAMP_HEIGHT, Y_TOLERANCE

    # 1. Setup Coordinates and Thresholds
    min_z, max_z = get_z_bounds(mesh)

    # Define the single threshold boundary:
    # We only care about the BOTTOM region: from min_z up to min_z + Z_CLAMP_HEIGHT (15.0)
    Z_THRESHOLD_BOTTOM = min_z + Z_CLAMP_HEIGHT

    num_faces = mesh.ncells
    face_colors = np.full((num_faces, 3), 200, dtype=np.uint8)

    centroids = mesh.cell_centers().points
    centroid_zs = centroids[:, 2]
    normals = mesh.cell_normals

    # --- Filter A (Y-Normal Dominance) ---
    y_dominant_mask = (np.abs(normals[:, 1]) > np.abs(normals[:, 0])) & \
                      (np.abs(normals[:, 1]) > np.abs(normals[:, 2]))
    y_normal_indices = np.where(y_dominant_mask)[0].astype(int)
    print(f"Filter A (Y-Normal Dominance): {len(y_normal_indices)} faces remain.")

    # --- Filter B (Z-Position: Only Bottom 15cm Region) ---

    # Condition: Centroid is in the bottom region (Z <= Z_THRESHOLD_BOTTOM)
    clamp_z_mask = (centroid_zs <= Z_THRESHOLD_BOTTOM)

    clamp_z_indices = np.where(clamp_z_mask)[0].astype(int)

    # --- PRINTING CHANGE ---
    print(f"Filter B (Z-Clamp Position: ONLY Bottom 15cm): {len(clamp_z_indices)} faces remain.")

    # Intersection 1: Z-Clamp and Y-Normal
    yz_intersection = np.intersect1d(clamp_z_indices, y_normal_indices).astype(int)
    print(f"Intersection (Z & Y): {len(yz_intersection)} faces remain.")

    # --- Filter C (Y-Boundary Check to exclude slots) ---

    # 1. Get the global bounding box Y extremes
    bounds = mesh.bounds()
    min_y = bounds[2]
    max_y = bounds[3]

    # 2. Get Centroids ONLY for the faces that passed the YZ filter
    filtered_centroids = centroids[yz_intersection]

    # 3. Create a mask to identify faces near the ABSOLUTE Front or Back boundary:
    cond1 = (filtered_centroids[:, 1] >= max_y - Y_TOLERANCE)
    cond2 = (filtered_centroids[:, 1] <= min_y + Y_TOLERANCE)

    boundary_mask = cond1 | cond2

    # Get the indices of the faces that are on the main Y-boundary
    outward_indices = yz_intersection[np.where(boundary_mask)[0]].astype(int)
    print(f"Filter C (Y-Boundary Check): {len(outward_indices)} faces remain.")

    # --- Final Intersection with Filter 1 (Visibility) ---
    final_clamp_indices = np.intersect1d(outward_indices, outside_indices).astype(int)

    # 5. Apply the corresponding color
    face_colors[final_clamp_indices] = [255, 0, 0]  # Red for "clamp"

    # 6. Apply the colors to the mesh for visualization
    mesh.cellcolors = face_colors

    print(f"Mesh colored. Z-Bounds: [{min_z:.2f}, {max_z:.2f}], FINAL Labeled faces: {len(final_clamp_indices)}")

    return mesh


# --------------------------------------------------------


if __name__ == "__main__":
    # Note: We must ensure Z_CLAMP_HEIGHT is set to the desired value here
    Z_CLAMP_HEIGHT = 15.0

    Tk().withdraw()

    # Allow selection of multiple files (e.g., 10 files)
    stl_files = askopenfilenames(title="Select STL Part Files", filetypes=[("STL", ".stl")])

    if not stl_files:
        print("File selection cancelled. Exiting.")
        exit()

    # 1. Initialize the Plotter for multiple sub-windows
    num_files = len(stl_files)
    n_rows = 2
    n_cols = int(np.ceil(num_files / n_rows))

    VP = Plotter(shape=(n_rows, n_cols), bg='white', size=(1600, 800))

    # 2. Loop through the files and render each one
    for i, stl_file in enumerate(stl_files):
        VP.at(i).camera.Elevation(5)

        mesh = Mesh(stl_file)

        # --- Apply Filters and Coloring ---
        outside_indices = filter_outside_faces(mesh)

        # Note: color_faces_geometrically now uses the updated global Z_CLAMP_HEIGHT (15.0)
        labeled_mesh = color_faces_geometrically(mesh, outside_indices)

        # Add a title to the subplot
        part_filename = os.path.basename(stl_file)

        # 3. Render the mesh in the current cell
        VP.show(labeled_mesh,
                title=f"Part {i}: {part_filename}",
                axes=0,
                interactive=False)

    # 4. Display the entire Plotter window
    VP.interactive()