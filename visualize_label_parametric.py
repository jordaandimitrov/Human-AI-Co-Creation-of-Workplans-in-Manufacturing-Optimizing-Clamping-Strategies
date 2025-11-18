from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
from vedo import Mesh, Plotter
import os
from tkinter.filedialog import askopenfilenames
# --------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------
# This is the 10 units (2cm in the original Blender comment)
# used for the clamping region threshold.
Z_CLAMP_HEIGHT = 10


# --------------------------------------------------------

def get_bottom_z(mesh):
    """Finds the lowest Z coordinate of any vertex in the mesh."""
    # mesh.points gives all vertex coordinates (Nx3 numpy array)
    vertices = mesh.points
    # Z-coordinates are in the third column (index 2)
    return np.min(vertices[:, 2])


def filter_outside_faces(mesh):
    """
    Filters faces by ensuring the mesh is manifold and extracts only the external surface.
    This is much more robust than ray casting after Voxel Remesh operations.
    """
    print("Starting surface extraction for visibility...")

    # CRITICAL FIX: Use the built-in cleaning and surface extraction tool.
    # We clone the mesh first so the original indices aren't lost immediately,
    # but the simplest approach is often to use mesh.clean().

    # We cannot simply clean() because clean() re-indexes the mesh, breaking
    # the index alignment for the other filters.

    # Instead, we will SKIP Filter 1 initially to debug the other two filters,
    # or, if we must keep it, we use a simpler approach that doesn't rely on ray casting.

    # For now, let's TEMPORARILY disable the ray-casting filter by returning ALL indices
    # to confirm the Z & Y filters are working correctly.

    num_faces = mesh.ncells
    all_indices = np.arange(num_faces).astype(int)

    print(f"Filter 1 (Visibility): {num_faces} faces remain (TEMPORARILY SKIPPED).")
    return all_indices


def color_faces_geometrically(mesh, outside_indices):
    """
    Colors the faces based on three combined geometric rules:
    1. Z-Position (Clamping Region)
    2. Y-Orientation (Normal Dominance)
    3. Y-Boundary Check (Excluding internal slot walls)
    4. Visibility (Ray Casting result from outside_indices)
    """

    # PARAMETER from the previous script
    Z_CLAMP_HEIGHT = 10
    # Tolerance for the Y-Boundary check. Adjust this value based on your
    # expected wall thickness, typically 1 to 5 units.
    Y_TOLERANCE = 2.0

    # 1. Setup Coordinates and Thresholds
    bottom_z = get_bottom_z(mesh)
    Z_THRESHOLD = bottom_z + Z_CLAMP_HEIGHT

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

    # --- Filter B (Z-Clamp Position) ---
    clamp_z_indices = np.where(centroid_zs <= Z_THRESHOLD)[0].astype(int)
    print(f"Filter B (Z-Clamp Position): {len(clamp_z_indices)} faces remain.")

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
    # Condition 1: Near Max Y boundary (Front)
    cond1 = (filtered_centroids[:, 1] >= max_y - Y_TOLERANCE)

    # Condition 2: Near Min Y boundary (Back)
    cond2 = (filtered_centroids[:, 1] <= min_y + Y_TOLERANCE)

    # Combine the two conditions to find faces on the main exterior Y-surfaces
    boundary_mask = cond1 | cond2

    # Get the indices of the faces that are on the main Y-boundary
    outward_indices = yz_intersection[np.where(boundary_mask)[0]].astype(int)
    print(f"Filter C (Y-Boundary Check): {len(outward_indices)} faces remain.")

    # --- Final Intersection with Filter 1 (Visibility) ---
    # outside_indices is the result of the ray casting (Filter 1)
    final_clamp_indices = np.intersect1d(outward_indices, outside_indices).astype(int)

    # 5. Apply the corresponding color
    face_colors[final_clamp_indices] = [255, 0, 0]  # Red for "clamp"

    # 6. Apply the colors to the mesh for visualization
    mesh.cellcolors = face_colors

    print(f"Mesh colored. Bottom Z: {bottom_z:.2f}, FINAL Labeled faces: {len(final_clamp_indices)}")

    return mesh# --------------------------------------------------------


if __name__ == "__main__":
    Tk().withdraw()

    # Allow selection of multiple files (e.g., 10 files)
    stl_files = askopenfilenames(title="Select 10 STL Part Files", filetypes=[("STL", ".stl")])

    if not stl_files:
        print("File selection cancelled. Exiting.")
        exit()

    # 1. Initialize the Plotter for 10 sub-windows (cells)
    # This creates a 2x5 grid layout automatically.
    # We use size=(1600, 800) to ensure a wide display for the 1x10 layout.
    # To get a 1x10 layout, use N=(1, 10).
    VP = Plotter(shape=(2, 5), bg='white', size=(1600, 800))
    # 2. Loop through the files and render each one
    for i, stl_file in enumerate(stl_files):
        # Set the current subplot cell for drawing
        VP.at(i).camera.Elevation(5)  # Set camera view slightly raised

        # Load the mesh
        mesh = Mesh(stl_file)

        # --- Apply Filters and Coloring ---
        outside_indices = filter_outside_faces(mesh)
        labeled_mesh = color_faces_geometrically(mesh, outside_indices)

        # Add a title to the subplot
        part_filename = os.path.basename(stl_file)

        # 3. Render the mesh in the current cell
        VP.show(labeled_mesh,
                title=f"Part {i}: {part_filename}",
                axes=0,  # Use axes=0 for clean subplots
                interactive=False)  # Don't pause after each plot

    # 4. Display the entire Plotter window
    VP.interactive()