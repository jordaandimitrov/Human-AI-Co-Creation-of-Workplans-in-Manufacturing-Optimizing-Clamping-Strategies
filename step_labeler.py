from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json
import numpy as np
from vedo import Mesh, Plotter
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopoDS import topods


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


from vedo import Plotter, Mesh, Text2D


def interactive_labeling(save_json="labels.json"):
    # --- File picker ---
    Tk().withdraw()
    file_path = askopenfilename(
        title="Select an STP file",
        filetypes=[("STP files", "*.stp"), ("STEP files", "*.step")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    shape = read_step_file(file_path)
    meshes, face_indices = mesh_faces(shape)

    # --- State ---
    selected_clamp = set()
    selected_support = set()
    mode = ["clamp"]

    pl = Plotter(title="Face Labeler (Press C/U to toggle mode, S to save)")

    # Text2D overlay for mode info
    mode_text = Text2D("Mode: CLAMP  [C]=Clamp  [U]=Support  [S]=Save",
                       pos="top-left", c="yellow", bg="black", font="courier")
    pl.add(mode_text)

    def update_text():
        mode_text.text(f"Mode: {mode[0].upper()}  [C]=Clamp  [U]=Support  [S]=Save")
        mode_text.c("yellow" if mode[0] == "clamp" else "cyan")

    def update_colors():
        for m, idx in zip(meshes, face_indices):
            if idx in selected_clamp:
                m.c("red")
            elif idx in selected_support:
                m.c("blue")
            else:
                m.c("lightgray")
        pl.render()

    def on_click(evt):
        mesh = evt.actor
        if mesh is None:
            return
        idx = mesh.user_data

        if mode[0] == "clamp":
            if idx in selected_clamp:
                selected_clamp.remove(idx)
            else:
                selected_clamp.add(idx)
        else:  # support mode
            if idx in selected_support:
                selected_support.remove(idx)
            else:
                selected_support.add(idx)

        update_colors()

    def on_key(evt):
        key = evt.keypress.lower()

        if key == "c":
            mode[0] = "clamp"
            print("🔴 Switched to CLAMP mode.")
            update_text()
        elif key == "u":
            mode[0] = "support"
            print("🔵 Switched to SUPPORT mode.")
            update_text()
        elif key == "s":
            clamp_labels = [1 if i in selected_clamp else 0 for i in face_indices]
            support_labels = [1 if i in selected_support else 0 for i in face_indices]

            try:
                with open(save_json) as f:
                    all_labels = json.load(f)
            except:
                all_labels = {}

            all_labels[file_path] = {
                "clamp_labels": clamp_labels,
                "support_labels": support_labels
            }

            with open(save_json, "w") as f:
                json.dump(all_labels, f, indent=4)

            print(f"💾 Saved labels for {file_path}")

        update_colors()

    # Attach face indices to meshes
    for m, idx in zip(meshes, face_indices):
        m.user_data = idx
        pl.add(m)

    pl.add_callback("mouse click", on_click)
    pl.add_callback("key press", on_key)
    update_colors()

    pl.show(interactive=True)



if __name__ == "__main__":
    interactive_labeling("labels.json")
