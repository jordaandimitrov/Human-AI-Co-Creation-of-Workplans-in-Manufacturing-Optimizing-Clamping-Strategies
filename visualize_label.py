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
    Loads an STL mesh, calculates its relative path starting from 'training_set',
    retrieves labels using that relative key, and colors the mesh.
    """

    # 1. Standardize the path separators to forward slashes (JSON standard)
    full_path_str = os.path.normpath(stl_file_path).replace('\\', '/')

    # 2. Extract relative path starting from 'training_set'
    target_folder = "training_set"

    if target_folder in full_path_str:
        # Find the index where "training_set" begins
        start_index = full_path_str.find(target_folder)
        # Slice the string to keep only "training_set/..."
        json_key = full_path_str[start_index:]
        print(f"DEBUG: Converted absolute path to relative key: '{json_key}'")
    else:
        # Fallback if the file isn't inside a 'training_set' folder
        print(f"WARNING: '{target_folder}' folder not found in path. Using absolute path as fallback.")
        json_key = full_path_str

    labels_for_part = all_labels_data.get(json_key)

    if labels_for_part is None:
        print(f"ERROR: No labels found for key '{json_key}'.")
        print(">> Check: Does your JSON file use keys like 'training_set/subdir/file.stl'?")
        labels_for_part = {}

    # 3. Read STL and load mesh
    try:
        mesh = Mesh(stl_file_path)
    except Exception as e:
        raise RuntimeError(f"Error loading STL file: {e}")

    num_faces = mesh.ncells

    # Initialize all faces to default gray (RGB 200, 200, 200)
    face_colors = np.full((num_faces, 3), 200, dtype=np.uint8)
    labeled_count = 0

    # 4. Apply colors based on JSON data (Red for 'clamp')
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

    # 5. Apply colors using the robust UINT8 format
    mesh.cellcolors = face_colors

    print(f"Mesh loaded with {num_faces} faces.")
    print(f"✅ {labeled_count} faces successfully colored red ('clamp').")

    # 6. Show mesh
    # Display the relative key in the window title for verification
    pl = Plotter(title=f"Label Viewer: {json_key}", axes=1)
    pl.add(mesh)

    pl.show(interactive=True, viewup="z", resetcam=True)


# ---------------- Main ----------------
if __name__ == "__main__":
    Tk().withdraw()

    print("--- STL Label Viewer (Relative Path Mode) ---")

    # 1. Select the STL file to view (returns absolute path)
    stl_file_path = askopenfilename(
        title="Select STL file (Must be inside 'training_set' folder)",
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
    print(f"Loading labels from {os.path.basename(labels_file)}...")
    with open(labels_file) as f:
        all_labels_data = json.load(f)

    # 4. Visualize
    visualize_labels(stl_file_path, all_labels_data)