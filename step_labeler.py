import json
from vedo import Plotter, Mesh
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
import numpy as np

def read_step_file(filename):
    reader = STEPControl_Reader()
    status = reader.ReadFile(filename)
    if status != IFSelect_RetDone:
        raise ValueError("Error reading STEP file")
    reader.TransferRoots()
    return reader.OneShape()

def mesh_faces(shape):
    """Mesh faces and return list of Mesh objects and their indices"""
    BRepMesh_IncrementalMesh(shape, 0.05, True, True)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    meshes = []
    face_indices = []
    idx = 0
    while exp.More():
        face = topods.Face(exp.Current())
        loc = face.Location()
        triang = BRep_Tool.Triangulation(face, loc)
        if triang:
            nodes = np.array([[triang.Node(i).X(),
                               triang.Node(i).Y(),
                               triang.Node(i).Z()]
                              for i in range(1, triang.NbNodes() + 1)])
            tris = np.array([[triang.Triangle(i).Get()[0]-1,
                              triang.Triangle(i).Get()[1]-1,
                              triang.Triangle(i).Get()[2]-1]
                             for i in range(1, triang.NbTriangles() + 1)])
            mesh = Mesh([nodes, tris], c="lightgray", alpha=1.0)
            meshes.append(mesh)
            face_indices.append(idx)
        exp.Next()
        idx += 1
    return meshes, face_indices

def interactive_labeling(save_json="labels.json"):
    # Open file explorer
    Tk().withdraw()
    file_path = askopenfilename(title="Select an STP file", filetypes=[("STP files", "*.stp")])
    if not file_path:
        print("No file selected. Exiting.")
        return

    shape = read_step_file(file_path)
    meshes, face_indices = mesh_faces(shape)
    selected = set()  # indices of selected faces

    pl = Plotter(title="Click faces to select clamping faces (red)")

    def on_click(evt):
        mesh = evt.actor
        if mesh is None:
            return
        # Toggle selection
        idx = mesh.user_data
        if idx in selected:
            selected.remove(idx)
            mesh.c("lightgray")
        else:
            selected.add(idx)
            mesh.c("red")
        pl.render()

    # Attach face index to each mesh
    for m, idx in zip(meshes, face_indices):
        m.user_data = idx
        pl.add(m)

    def on_key(evt):
        # evt.keypress is a string of the pressed key
        if evt.keypress == "s":
            labels = [1 if i in selected else 0 for i in face_indices]
            try:
                with open(save_json) as f:
                    all_labels = json.load(f)
            except:
                all_labels = {}
            all_labels[file_path] = labels
            with open(save_json, "w") as f:
                json.dump(all_labels, f, indent=4)
            print(f"Saved labels for {file_path}")

    pl.add_callback("key press", on_key)
    pl.add_callback("mouse click", on_click)
    pl.show(interactive=True)

# Run the interactive labeling
if __name__ == "__main__":
    interactive_labeling("labels.json")
