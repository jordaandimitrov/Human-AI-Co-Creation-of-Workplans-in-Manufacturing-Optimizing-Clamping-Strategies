from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json
import numpy as np
from vedo import Mesh, Plotter
import os


# --- OCC/STEP/Subdivision libraries and functions are REMOVED ---

# ---------------- Visualization ----------------
def visualize_labels(stl_file, all_labels_data):
    """
    Loads an STL mesh, retrieves its corresponding labels from the JSON data,
    and colors the mesh based on the 'clamp'/'unselected' labels.
    """

    # 1. Determine the part name key
    part_filename = os.path.basename(stl_file)
    # Key is 'part_000' if the file is 'part_000.stl'
    part_name_key = os.path.splitext(part_filename)[0]

    labels_for_part = all_labels_data.get(part_name_key)

    if labels_for_part is None:
        print(f"No labels found for key '{part_name_key}' in the JSON data. Showing mesh gray.")
        # If no specific labels, we treat the object as having no labeled faces
        labels_for_part = {}

    # 2. Read STL and load mesh
    try:
        mesh = Mesh(stl_file)
    except Exception as e:
        raise RuntimeError(f"Error loading STL file: {e}")

    tris = np.array(mesh.cells).reshape(-1, 3)
    num_faces = len(tris)

    # Initialize all faces to default gray
    face_colors = np.full((num_faces, 3), 200, dtype=np.uint8)

    # 3. Apply colors based on JSON data
    labeled_count = 0
    for t_idx_str, label in labels_for_part.items():
        try:
            t_idx = int(t_idx_str)
        except ValueError:
            continue  # Skip non-integer keys

        if t_idx < num_faces:
            if label == "clamp":
                face_colors[t_idx] = [255, 0, 0]  # Red for "clamp"
                labeled_count += 1
            # Note: We skip the "unselected" label as faces are already gray

    mesh.cellcolors = face_colors

    print(f"Mesh loaded with {num_faces} faces. {labeled_count} faces colored red ('clamp').")

    # 4. Show mesh
    pl = Plotter(title=f"Label Viewer: {part_name_key}")
    # Add the single colored mesh to the plotter
    pl.add(mesh)

    pl.show(interactive=True, axes=1, viewup="z", resetcam=True)


# ---------------- Main ----------------
if __name__ == "__main__":
    Tk().withdraw()

    # 1. Select the STL file to view
    stl_file = askopenfilename(
        title="Select STL file to view",
        filetypes=[("STL files", "*.stl")]
    )
    if not stl_file:
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
    # Pass the selected STL file path and the entire loaded JSON data
    visualize_labels(stl_file, all_labels_data)