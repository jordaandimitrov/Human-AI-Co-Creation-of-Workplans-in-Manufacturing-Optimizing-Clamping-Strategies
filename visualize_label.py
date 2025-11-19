from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json
import numpy as np
from vedo import Mesh, Plotter
import os


# ----------------------------------------------------------------------
# VIEWER FUNCTION
# ----------------------------------------------------------------------

def visualize_labels(stl_file_path, all_labels_data):
    """
    Loads an STL mesh, retrieves its labels using the full file path
    as the key from the JSON data, and colors the mesh based on the labels.
    """

    # 1. Determine the key: Use the entire absolute file path, standardized.
    # This ensures consistency with the JSON creation script.
    json_key = os.path.normpath(stl_file_path).replace('\\', '/')

    labels_for_part = all_labels_data.get(json_key)

    if labels_for_part is None:
        print(
            f"ERROR: No labels found for key '{json_key}'. Please ensure the STL path exactly matches a key in your JSON file.")
        # Fallback to display the gray mesh
        labels_for_part = {}

    # 2. Read STL and load mesh
    try:
        mesh = Mesh(stl_file_path)
    except Exception as e:
        raise RuntimeError(f"Error loading STL file: {e}")

    num_faces = mesh.ncells

    # Initialize all faces to default gray (RGB 200, 200, 200)
    face_colors = np.full((num_faces, 3), 200, dtype=np.uint8)
    labeled_count = 0

    # 3. Apply colors based on JSON data (Red for 'clamp')
    for t_idx_str, label in labels_for_part.items():
        try:
            t_idx = int(t_idx_str)
        except ValueError:
            continue

        if t_idx < num_faces:
            if label == "clamp":
                face_colors[t_idx] = [255, 0, 0]  # Red for "clamp"
                labeled_count += 1
            # "unselected" faces remain the initialized gray color (200)

    # 4. Apply colors using the robust UINT8 format
    mesh.cellcolors = face_colors

    print(f"Mesh loaded with {num_faces} faces.")
    print(f"✅ {labeled_count} faces successfully colored red ('clamp').")

    # 5. Show mesh
    pl = Plotter(title=f"Label Viewer: {os.path.basename(stl_file_path)} (Labeled)", axes=1)
    pl.add(mesh)

    pl.show(interactive=True, viewup="z", resetcam=True)


# ---------------- Main ----------------
if __name__ == "__main__":
    Tk().withdraw()

    # 1. Select the STL file to view (get full path)
    stl_file_path = askopenfilename(
        title="Select STL file to view",
        filetypes=[("STL files", "*.stl")]
    )
    if not stl_file_path:
        raise ValueError("No STL file selected!")

    # 2. Select the single master JSON file
    labels_file = askopenfilename(
        title="Select Master JSON Labels file",
        filetypes=[("JSON files", "*.json")]
    )
    if not labels_file:
        raise ValueError("No JSON labels file selected!")

    # 3. Load ALL labels from the master JSON file
    with open(labels_file) as f:
        all_labels_data = json.load(f)

    # 4. Visualize
    # Pass the full file path and the entire loaded JSON data
    visualize_labels(stl_file_path, all_labels_data)