from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
from vedo import Mesh, Plotter
import os
# --------------------------------------------------------
# PARAMETERS
# --------------------------------------------------------
# This is the 10 units (2cm in the original Blender comment)
# used for the clamping region threshold.
Z_CLAMP_HEIGHT = 10

# --------------------------------------------------------

def get_bottom_z(mesh):
    """Finds the lowest Z coordinate of any vertex in the mesh."""
    # mesh.points() gives all vertex coordinates (Nx3 numpy array)
    vertices = mesh.points
    # Z-coordinates are in the third column (index 2)
    return np.min(vertices[:, 2])


def color_faces_geometrically(mesh):
    """
    Calculates the 'clamp' label geometrically and colors the mesh faces.
    The rule: faces whose centroid Z is within Z_CLAMP_HEIGHT of the bottom_z.
    """

    # PARAMETER from the previous script
    Z_CLAMP_HEIGHT = 10

    # 1. Determine the reference Z-coordinates
    bottom_z = get_bottom_z(mesh)
    Z_THRESHOLD = bottom_z + Z_CLAMP_HEIGHT

    num_faces = mesh.ncells
    face_colors = np.full((num_faces, 3), 200, dtype=np.uint8)

    # 2. Get the centroid of every cell (face/triangle)
    centroids = mesh.cell_centers()

    # --- THE FIX IS HERE ---
    # Convert the vedo.Points object to a NumPy array before slicing
    centroids_array = centroids.points

    # Get the Z coordinates only (index 2)
    centroid_zs = centroids_array[:, 2]
    # -----------------------

    # 3. Find the indices that satisfy the geometric rule
    clamp_indices = np.where(centroid_zs <= Z_THRESHOLD)[0]

    # 4. Apply the corresponding color
    face_colors[clamp_indices] = [255, 0, 0]  # Red for "clamp"

    # 5. Apply the colors to the mesh for visualization
    mesh.cellcolors = face_colors

    print(f"Mesh colored. Bottom Z: {bottom_z:.2f}, Threshold Z: {Z_THRESHOLD:.2f}")

    return mesh


# --------------------------------------------------------
if __name__ == "__main__":
    Tk().withdraw()

    # The viewer now only needs the STL file
    stl_file = askopenfilename(title="Select STL Part File", filetypes=[("STL", ".stl")])

    if not stl_file:
        print("File selection cancelled. Exiting.")
        exit()

        # Load the mesh
    mesh = Mesh(stl_file)
    print("Retriangulating mesh...")
    #mesh.triangulate()
    # Color the faces based on the geometric rule defined above
    mesh = color_faces_geometrically(mesh)

    part_filename = os.path.basename(stl_file)

    # Display the mesh with color labels and axes
    Plotter().show(mesh, axes=1, title=f"Labeled: {part_filename}")

    # ... (rest of the script)

