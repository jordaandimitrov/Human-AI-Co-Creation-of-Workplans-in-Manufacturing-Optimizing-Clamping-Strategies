# ============================================================
# TRIANGLE-LEVEL CLAMP & SUPPORT CLASSIFIER (Enhanced Features)
# ============================================================

import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ---------- OpenCascade ----------
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.TopoDS import topods
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

# ---------- Vedo ----------
from vedo import Mesh, Plotter, color_map, Text2D

# ============================================================
# STEP 1: STEP file & triangle feature extraction (enhanced)
# ============================================================

def read_step_file(filename):
    reader = STEPControl_Reader()
    status = reader.ReadFile(filename)
    if status != IFSelect_RetDone:
        raise ValueError(f"Error reading STEP file: {filename}")
    reader.TransferRoots()
    return reader.OneShape()

def extract_triangle_features(shape, mesh_deflection=0.05, subdivide_levels=2):
    """
    Extract triangle features with 20-dimensional enhanced descriptor:
    - area, perimeter, aspect ratio
    - normal, centroid, distance to part center
    - face index, number of neighbors, average neighbor area, avg normal similarity
    """
    from vedo import Mesh

    # ---------- Mesh the STEP shape ----------
    BRepMesh_IncrementalMesh(shape, mesh_deflection, True, True)

    all_points, all_tris, point_map, current_index = [], [], {}, 0
    face_to_points, face_to_tris = {}, {}

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0
    while exp.More():
        face = topods.Face(exp.Current())
        triang = BRep_Tool.Triangulation(face, face.Location())
        if triang:
            nodes = np.array([[triang.Node(i).X(),
                               triang.Node(i).Y(),
                               triang.Node(i).Z()]
                              for i in range(1, triang.NbNodes() + 1)])
            tris_local = np.array([[triang.Triangle(i).Value(1) - 1,
                                    triang.Triangle(i).Value(2) - 1,
                                    triang.Triangle(i).Value(3) - 1]
                                   for i in range(1, triang.NbTriangles() + 1)])
            face_point_set = set()
            for li, p in enumerate(nodes):
                key = tuple(np.round(p, 6))
                if key not in point_map:
                    point_map[key] = current_index
                    all_points.append(p)
                    current_index += 1
                face_point_set.add(point_map[key])
            for tri in tris_local:
                gtri = [point_map[tuple(np.round(nodes[int(idx)],6))] for idx in tri]
                all_tris.append(gtri)
            face_to_points[face_idx] = face_point_set
        exp.Next()
        face_idx += 1

    points = np.array(all_points)
    tris = np.array(all_tris, dtype=int)

    # ---------- Subdivide ----------
    def subdivide_mesh(mesh, levels=1):
        pts, t = mesh.points, np.array(mesh.cells).reshape(-1,3)
        for _ in range(levels):
            new_pts = pts.tolist()
            new_tris = []
            midpoint_map = {}
            def midpoint(i,j):
                key = tuple(sorted([i,j]))
                if key in midpoint_map: return midpoint_map[key]
                m = (pts[i]+pts[j])/2
                new_pts.append(m)
                idx = len(new_pts)-1
                midpoint_map[key] = idx
                return idx
            for tri in t:
                v0,v1,v2 = tri
                m01,m12,m20 = midpoint(v0,v1), midpoint(v1,v2), midpoint(v2,v0)
                new_tris += [[v0,m01,m20],[v1,m12,m01],[v2,m20,m12],[m01,m12,m20]]
            pts = np.array(new_pts)
            t = np.array(new_tris)
        return Mesh([pts,t])

    if subdivide_levels>0:
        mesh = Mesh([points,tris])
        mesh = subdivide_mesh(mesh, levels=subdivide_levels)
        points = mesh.points
        tris = np.array(mesh.cells,dtype=int).reshape(-1,3)

    # ---------- Map triangles to faces ----------
    tri_to_face=[]
    face_centroids=[]
    for fidx, pts_idx in face_to_points.items():
        pts_face = np.array([points[i] for i in pts_idx])
        face_centroids.append(np.mean(pts_face, axis=0))
    face_centroids = np.array(face_centroids)
    for tri in tris:
        centroid = np.mean(points[tri], axis=0)
        tri_to_face.append(int(np.argmin(np.linalg.norm(face_centroids - centroid, axis=1))))
    tri_to_face = np.array(tri_to_face)

    # ---------- Build neighbor info ----------
    tri_neighbors = [[] for _ in range(len(tris))]
    vert_to_tris = {i:[] for i in range(len(points))}
    for i,t in enumerate(tris):
        for v in t: vert_to_tris[v].append(i)
    for i,t in enumerate(tris):
        neighbor_set = set()
        for v in t:
            neighbor_set.update(vert_to_tris[v])
        neighbor_set.discard(i)
        tri_neighbors[i] = list(neighbor_set)

    # ---------- Compute features ----------
    feats=[]
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    part_center = np.array([props.CentreOfMass().X(),
                            props.CentreOfMass().Y(),
                            props.CentreOfMass().Z()])
    for i, tri in enumerate(tris):
        a,b,c = points[tri]
        centroid = (a+b+c)/3.0
        normal = np.cross(b-a, c-a)
        normal /= (np.linalg.norm(normal)+1e-9)
        area = 0.5*np.linalg.norm(np.cross(b-a, c-a))
        e0,e1,e2 = np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)
        perimeter = e0+e1+e2
        aspect = max(e0,e1,e2)/(min(e0,e1,e2)+1e-8)
        dist_center = np.linalg.norm(centroid - part_center)

        # neighbor info
        neighbors = tri_neighbors[i]
        n_neighbors = len(neighbors)
        avg_area = np.mean([0.5*np.linalg.norm(np.cross(points[tris[j][1]]-points[tris[j][0]],
                                                        points[tris[j][2]]-points[tris[j][0]])) for j in neighbors]) if neighbors else 0.0
        # Compute average normal similarity with neighbors safely
        if neighbors:
            sims = []
            for j in neighbors:
                v0, v1, v2 = points[tris[j][0]], points[tris[j][1]], points[tris[j][2]]
                n = np.cross(v1 - v0, v2 - v0)
                n /= (np.linalg.norm(n) + 1e-9)
                sims.append(np.dot(normal, n))
            avg_normal_sim = float(np.mean(sims))
        else:
            avg_normal_sim = 0.0

        feats.append([area, perimeter, aspect,
                      normal[0], normal[1], normal[2],
                      centroid[0], centroid[1], centroid[2],
                      dist_center,
                      tri_to_face[i], n_neighbors, avg_area, avg_normal_sim,
                      a[0], a[1], a[2],
                      b[0], b[1], b[2],
                      c[0], c[1], c[2]])

    feats = np.array(feats,dtype=np.float32)

    # Build face_to_tri mapping
    face_to_tri={}
    for idx, fidx in enumerate(tri_to_face):
        face_to_tri.setdefault(fidx, []).append(idx)

    return feats, tris, tri_to_face, face_to_tri, points

# ============================================================
# STEP 2: Dataset
# ============================================================

class TriangleDataset(Dataset):
    def __init__(self, parts, augment_rot=True):
        self.parts = parts
        self.augment_rot = augment_rot

    def __len__(self):
        return sum(p["tri_features"].shape[0] for p in self.parts)

    def __getitem__(self, idx):
        for p in self.parts:
            n_tri = p["tri_features"].shape[0]
            if idx < n_tri:
                X = p["tri_features"].copy()
                tri_verts = p["tri_verts"]
                tri_to_face = np.array(p["tri_to_face"], dtype=int)
                labels = p["labels"]
                break
            idx -= n_tri
        else:
            raise IndexError("Triangle index out of range")

        n_tri = X.shape[0]
        tri_clamp = np.zeros(n_tri,dtype=np.float32)
        tri_support = np.zeros(n_tri,dtype=np.float32)

        clamp_tris = np.array(labels.get("clamp_tris", []), dtype=int)
        if len(clamp_tris)>0:
            tri_clamp[clamp_tris] = 1.0

        support_faces = np.array(labels.get("support_faces", []), dtype=int)
        if len(support_faces)>0:
            tri_support = np.isin(tri_to_face, support_faces).astype(np.float32)

        if self.augment_rot and X.shape[1]>=9:
            angle = np.random.uniform(0,2*np.pi)
            R = np.array([[np.cos(angle),-np.sin(angle),0],
                          [np.sin(angle),np.cos(angle),0],
                          [0,0,1]])
            X[:,3:6] = X[:,3:6] @ R.T
            X[:,6:9] = X[:,6:9] @ R.T

        return torch.tensor(X,dtype=torch.float32), torch.tensor(
            np.stack([tri_clamp, tri_support],axis=1),dtype=torch.float32
        )

# ============================================================
# STEP 3: Model
# ============================================================

class ClampSupportNet(nn.Module):
    def __init__(self,in_dim=20,hidden_dim=64,out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim)
        )

    def forward(self,X):
        return self.net(X)

# ============================================================
# STEP 4: Training
# ============================================================

def train_model_triangles(model,dataloader,epochs=50,lr=1e-3,device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        total_loss=0.0
        batches=0
        for X,y in dataloader:
            if X.dim()==3 and X.shape[0]==1:
                X = X.squeeze(0)
                y = y.squeeze(0)
            X,y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches +=1
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/max(1,batches):.4f}")
    return model

# ============================================================
# STEP 5: Prediction
# ============================================================

def predict_best_triangles(model, features, top_k=5, zero_support=False):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(features,dtype=torch.float32)
        logits = model(X)
        probs = torch.sigmoid(logits).numpy()
    if zero_support: probs[:,1]=0.0
    clamp_probs = probs[:,0]
    support_probs = probs[:,1]
    top_clamp = np.argsort(-clamp_probs)[:top_k]
    top_support = np.argsort(-support_probs)[:top_k]
    return {"clamp_top":top_clamp,"support_top":top_support}, probs

# ============================================================
# STEP 6: Visualization
# ============================================================

def visualize_triangles(points,tris,tri_probs):
    meshes=[]
    for i,t in enumerate(tris):
        mesh=Mesh([points[t], [[0,1,2]]])
        mesh.tri_idx=i
        meshes.append(mesh)
    plt=Plotter(title="Clamp/Support Triangles",bg="gray",axes=1)
    view_mode={"current":0}
    overlay = Text2D("Mode: Clamp",s=1.5)
    face_info = Text2D("",pos="top-right",c="white",s=1.2)
    plt.add(overlay); plt.add(face_info)
    def update_colors():
        for i,m in enumerate(meshes):
            prob=tri_probs[i,view_mode["current"]]
            m.c(color_map(prob,name="jet",vmin=0,vmax=1))
        plt.render()
    def on_pick(evt):
        if evt.actor and hasattr(evt.actor,"tri_idx"):
            idx=evt.actor.tri_idx
            prob=tri_probs[idx,view_mode["current"]]
            label="Clamp" if view_mode["current"]==0 else "Support"
            face_info.text(f"Triangle {idx}\n{label}: {prob:.3f}")
            plt.render()
    def on_key(evt):
        key=evt.keypress.lower()
        if key=="c": view_mode["current"]=0; overlay.text("Mode: Clamp"); update_colors()
        elif key=="u": view_mode["current"]=1; overlay.text("Mode: Support"); update_colors()
    for m in meshes: plt.add(m)
    plt.add_callback("mouse click",on_pick)
    plt.add_callback("key press",on_key)
    update_colors()
    plt.show(interactive=True)

# ============================================================
# STEP 7: Load labeled dataset
# ============================================================

def load_labeled_dataset(labels_file):
    with open(labels_file) as f: all_labels=json.load(f)
    all_parts=[]
    for stp_file, labels in all_labels.items():
        if not os.path.exists(stp_file):
            print(f"⚠️ {stp_file} not found, skipping")
            continue
        try:
            subdivide_levels = labels.get("subdivide_levels",0)
            shape = read_step_file(stp_file)
            tri_feats, tri_verts, tri_to_face, face_to_tri, points = extract_triangle_features(
                shape, subdivide_levels=subdivide_levels
            )
            print(f"✅ Processed {stp_file} with subdivision level {subdivide_levels}")
            all_parts.append({
                "tri_features": tri_feats,
                "tri_verts": tri_verts,
                "tri_to_face": tri_to_face,
                "face_to_tri": face_to_tri,
                "points": points,
                "labels": labels,
                "file_path": stp_file
            })
        except Exception as e:
            print(f"⚠️ Error processing {stp_file}: {e}")
    return all_parts

# ============================================================
# STEP 8: Main
# ============================================================

if __name__=="__main__":
    Tk().withdraw()
    model_path="clamp_support_model.pth"
    labels_path=r"C:\Users\jorda\Desktop\Unif\MA3\Thesis\Git\labels.json"

    model=ClampSupportNet(in_dim=23, hidden_dim=64, out_dim=2)

    choice=input("Train new model (t) or load existing (l)? ").strip().lower()

    if choice=="t" or not os.path.exists(model_path):
        print("🧠 Loading labeled dataset...")
        all_parts=load_labeled_dataset(labels_path)
        if not all_parts: raise ValueError("No labeled parts found!")
        dataset=TriangleDataset(all_parts, augment_rot=True)
        dataloader=DataLoader(dataset,batch_size=1,shuffle=True)
        if os.path.exists(model_path):
            try: model.load_state_dict(torch.load(model_path)); print("Loaded existing weights.")
            except: pass
        print("🚀 Training model...")
        trained_model=train_model_triangles(model,dataloader,epochs=10,lr=1e-3)
        torch.save(trained_model.state_dict(),model_path)
        torch.save(trained_model.state_dict(),f"clamp_support_model_{datetime.now():%Y%m%d_%H%M}.pth")
        print(f"✅ Model saved to {model_path}")
    else:
        print("⚡ Loading pretrained model...")
        model.load_state_dict(torch.load(model_path))
        trained_model=model

    while True:
        test_file=askopenfilename(title="Select STEP file",filetypes=[("STP files","*.stp")])
        if not test_file: break
        shape=read_step_file(test_file)
        tri_feats, tri_verts, tri_to_face, face_to_tri, points = extract_triangle_features(shape)
        tops, probs = predict_best_triangles(trained_model, tri_feats, top_k=5, zero_support=True)
        print("\n=== Triangle Probabilities (Clamp | Support) ===")
        for i,p in enumerate(probs):
            print(f"Tri {i}: clamp={p[0]:.4f} | support={p[1]:.4f}")
        visualize_triangles(points, tri_verts, probs)
        cont=input("\nPress [t] to test another file, or any other key to quit: ").strip().lower()
        if cont!="t": break
