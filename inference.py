import torch
import torch.nn as nn
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from vedo import Mesh, Plotter, Text2D
import features  # Shared features module


# MODEL DEFINITION (Must match training)
class ClampSupportNet(nn.Module):
    def __init__(self, in_dim=13, hidden_dim=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, X): return self.net(X)


def visualize_inference(points, tris, probs, initial_thresh=0.90):
    """
    Efficient Scalar Visualization (No Loops)
    """
    print("Building visualization mesh...")

    # 1. Create SINGLE mesh
    mesh = Mesh([points, tris])

    # 2. Attach Probability Data (Class 0 = Clamp)
    clamp_probs = probs[:, 0]
    mesh.celldata["Clamp_Probability"] = clamp_probs

    # 3. Setup Plotter
    plt = Plotter(title="Inference Results", bg="white", axes=1)

    # 4. Dynamic Coloring Function
    def apply_threshold(thresh):
        # Create RGBA colors array
        # Default: Light Grey, Semi-Transparent
        colors = np.full((mesh.ncells, 4), [200, 200, 200, 50], dtype=np.uint8)

        # High Confidence: Red, Opaque
        mask = clamp_probs > thresh
        colors[mask] = [255, 0, 0, 255]

        mesh.cellcolors = colors

    apply_threshold(initial_thresh)

    # 5. UI Elements
    txt = Text2D(f"Threshold: {initial_thresh:.2f}", pos="bottom-left", s=1.2)
    plt.add(txt)

    def on_slider(widget, event):
        val = widget.GetRepresentation().GetValue()
        apply_threshold(val)
        txt.text(f"Threshold: {val:.2f}")

    plt.add_slider(on_slider, xmin=0.0, xmax=1.0, value=initial_thresh,
                   pos=[(0.1, 0.05), (0.4, 0.05)], title="Confidence Filter")

    print("✅ Displaying. Use slider to adjust filter.")
    plt.show(mesh, interactive=True)


if __name__ == "__main__":
    Tk().withdraw()

    # SELECT FILES
    model_path = askopenfilename(title="Select Model .pth", filetypes=[("Model", "*.pth")])
    if not model_path: exit()

    stl_path = askopenfilename(title="Select Test STL", filetypes=[("STL", ".stl")])
    if not stl_path: exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD MODEL
    model = ClampSupportNet(in_dim=13).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # EXTRACT & PREDICT
    print("Extracting features (this may take a moment)...")
    feats, tris, points = features.extract_triangle_features(stl_path)

    input_tensor = torch.tensor(feats, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    visualize_inference(points, tris, probs)