from tkinter import Tk
from tkinter.filedialog import askopenfilenames, asksaveasfilename
import numpy as np
from vedo import Mesh, Plotter
import os
import json

# --------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------
# This is the 15.0 units (15cm) used for the clamping region threshold.
Z_CLAMP_HEIGHT = 25.0
# Tolerance for the Y-Boundary check. Adjust based on expected wall thickness.
Y_TOLERANCE = 2.0


# --------------------------------------------------------

def get_z_bounds(mesh):
    """Finds the lowest and highest Z coordinate of any vertex in the mesh."""
    vertices = mesh.points
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


def generate_labels_dict(mesh, final_clamp_indices):
    """
    Creates a dictionary of {face_index (string): label (string)} for a single mesh.
    """
    num_faces = mesh.ncells
    labels_dict = {}

    # Initialize all labels to "unselected"
    for i in range(num_faces):
        labels_dict[str(i)] = "unselected"

    # Overwrite labeled indices with "clamp"
    for index in final_clamp_indices:
        labels_dict[str(index)] = "clamp"

    return labels_dict


def color_faces_geometrically(mesh, outside_indices):
    """
    Calculates and returns the colored mesh and the final labeled indices.
    """
    # --- Parameters ---
    # The parameters are defined globally but can be used locally here:
    global Z_CLAMP_HEIGHT, Y_TOLERANCE # Keep global declarations for safety if parameters change

    # 1. Setup Coordinates and Thresholds
    min_z, max_z = get_z_bounds(mesh)
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
    clamp_z_mask = (centroid_zs <= Z_THRESHOLD_BOTTOM)
    clamp_z_indices = np.where(clamp_z_mask)[0].astype(int)
    print(f"Filter B (Z-Clamp Position: ONLY Bottom 15cm): {len(clamp_z_indices)} faces remain.")

    # Intersection 1: Z-Clamp and Y-Normal
    yz_intersection = np.intersect1d(clamp_z_indices, y_normal_indices).astype(int)
    print(f"Intersection (Z & Y): {len(yz_intersection)} faces remain.")

    # --- Filter C (Y-Boundary Check to exclude slots) ---
    bounds = mesh.bounds()
    min_y = bounds[2]
    max_y = bounds[3]

    # 2. Get Centroids ONLY for the faces that passed the YZ filter
    filtered_centroids = centroids[yz_intersection]

    # 3. Create a mask to identify faces near the ABSOLUTE Front or Back boundary:
    # **FIX APPLIED HERE:** Y_TOLERANCE is now accessible
    cond1 = (filtered_centroids[:, 1] >= max_y - Y_TOLERANCE)
    cond2 = (filtered_centroids[:, 1] <= min_y + Y_TOLERANCE)

    boundary_mask = cond1 | cond2

    # Get the indices of the faces that are on the main Y-boundary
    outward_indices = yz_intersection[np.where(boundary_mask)[0]].astype(int)
    print(f"Filter C (Y-Boundary Check): {len(outward_indices)} faces remain.")

    # --- Final Intersection with Filter 1 (Visibility) ---
    final_clamp_indices = np.intersect1d(outward_indices, outside_indices).astype(int)

    # 5. Apply the corresponding color
    face_colors[final_clamp_indices] = [255, 0, 0]

    # 6. Apply the colors to the mesh for visualization
    mesh.cellcolors = face_colors

    print(f"Mesh colored. FINAL Labeled faces: {len(final_clamp_indices)}")

    return mesh, final_clamp_indices


# --------------------------------------------------------

if __name__ == "__main__":


    Tk().withdraw()

    stl_files = askopenfilenames(title="Select STL Part Files", filetypes=[("STL", ".stl")])
    if not stl_files:
        print("File selection cancelled. Exiting.")
        exit()

    # --- 1. Master Dictionary Initialization ---
    all_labels_data = {}
    #output_dir = os.path.dirname(stl_files[0])
    output_dir = "training_set"
    # Prompt user for the final JSON filename
   # output_json_path = asksaveasfilename(
    #    defaultextension=".json",
    #    initialdir=output_dir,
    #    initialfile="all_part_labels.json",
    #    title="Select location to save master JSON file"
    #)
    output_json_path = r'training_set/all_part_labels.json'

    if not output_json_path:
        print("Saving cancelled. Exiting.")
        exit()
    # ------------------------------------------

    num_files = len(stl_files)
    n_rows = 2
    n_cols = int(np.ceil(num_files / n_rows))

    VP = Plotter(shape=(n_rows, n_cols), bg='white', size=(1600, 800))

    # 2. Loop through all files
    for i, stl_file in enumerate(stl_files):
        VP.at(i).camera.Elevation(5)

        mesh = Mesh(stl_file)

        # --- Apply Filters and Coloring ---
        outside_indices = filter_outside_faces(mesh)
        labeled_mesh, final_indices = color_faces_geometrically(mesh, outside_indices)

        # FIX: Use the full, normalized file path as the dictionary key
        json_key = 'training_set/' + os.path.basename(stl_file)

        # 3. Generate Labels Dictionary and Add to Master Dictionary
        labels_dict = generate_labels_dict(labeled_mesh, final_indices)
        all_labels_data[json_key] = labels_dict
        print(f"Accumulated labels for {json_key}.")

        # 4. Render the mesh
        VP.show(labeled_mesh,
                title=f"Part {i}: {os.path.basename(stl_file)}",
                axes=0,
                interactive=False)

    # 5. Display the entire Plotter window
    VP.interactive()

    # --- 6. Save the Master Dictionary to a Single JSON File ---
    with open(output_json_path, 'w') as f:
        json.dump(all_labels_data, f, indent=2)

    print(f"✅ All {num_files} parts' labels successfully saved to: {output_json_path}")