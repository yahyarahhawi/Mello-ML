#!/usr/bin/env python3
"""
Generate UMAP transformations (2D & 3D) from main.json and save as:
  - main_UMAP2D.json
  - main_UMAP3D.json
Also writes interactive Plotly HTMLs:
  - umap_2d_visualization.html
  - umap_3d_visualization.html
"""

import json
import numpy as np
import argparse
from pathlib import Path

import plotly.graph_objects as go
from sklearn.decomposition import PCA

try:
    import umap
except ImportError:
    raise SystemExit("❌ UMAP not found. Install with: pip install umap-learn")

# ---------------------------
# IO
# ---------------------------

def load_data(path="main.json"):
    """Load the main.json data (expects a list of objects, each with 'name' and 'books_vector')."""
    p = Path(path)
    if not p.exists():
        print(f"❌ Error: {path} not found")
        return None
    with p.open("r") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} users from {path}")
    return data

def extract_features(data, vector_key="books_vector", name_key="name"):
    """Extract embedding vectors + labels from the data."""
    features, labels, bad = [], [], 0
    for entry in data:
        name = entry.get(name_key, "Unknown")
        vec = entry.get(vector_key)
        if isinstance(vec, list) and len(vec) > 0:
            features.append(vec)
            labels.append(name)
        else:
            bad += 1
    if bad:
        print(f"⚠️ Skipped {bad} entries with missing/invalid '{vector_key}'")
    if not features:
        print("❌ No valid vectors found.")
        return None, None
    X = np.asarray(features, dtype=np.float32)
    return X, labels

# ---------------------------
# Math helpers
# ---------------------------

def unit_normalize_rows(X, eps=1e-9):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)

def maybe_pca_50(X, use_pca=True, random_state=42):
    if not use_pca or X.shape[1] <= 50:
        return X
    print("🔧 PCA → 50D before UMAP (often improves speed/stability)")
    pca = PCA(n_components=50, random_state=random_state)
    return pca.fit_transform(X)

# ---------------------------
# UMAP core
# ---------------------------

def run_umap(X, n_components=2, n_neighbors=20, min_dist=0.1, metric="cosine", random_state=42):
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(X)
    print(f"✅ UMAP({n_components}D) complete. Shape: {coords.shape}")
    return coords, reducer

# ---------------------------
# Save outputs
# ---------------------------

def save_umap_json(coords, labels, out_path):
    """Save coordinates to JSON (compatible with your PCA/TSNE format)."""
    out = {"users": []}
    if coords.shape[1] == 2:
        for i, name in enumerate(labels):
            out["users"].append({
                "name": name,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "isCurrentUser": False,
                "special_user": False
            })
    else:
        for i, name in enumerate(labels):
            out["users"].append({
                "name": name,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
                "isCurrentUser": False,
                "special_user": False
            })
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"💾 Saved: {out_path}")

def plot_2d(coords, labels, out_html="umap_2d_visualization.html", title="UMAP 2D Visualization"):
    fig = go.Figure(
        data=go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode='markers',
            marker=dict(size=7, opacity=0.85),
            text=labels,
            hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        width=1000,
        height=800
        # (No explicit colors to keep Plotly defaults)
    )
    fig.write_html(out_html)
    print(f"📊 2D visualization saved to {out_html}")
    # fig.show()  # enable if you want to open a window

def plot_3d(coords, labels, out_html="umap_3d_visualization.html", title="UMAP 3D Visualization"):
    fig = go.Figure(
        data=go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(size=5, opacity=0.85),
            text=labels,
            hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='UMAP-1',
            yaxis_title='UMAP-2',
            zaxis_title='UMAP-3'
        ),
        width=1000,
        height=800
    )
    fig.write_html(out_html)
    print(f"📊 3D visualization saved to {out_html}")
    # fig.show()

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="UMAP projection from main.json")
    parser.add_argument("--input", default="main.json", help="Path to input JSON (default: main.json)")
    parser.add_argument("--vector-key", default="books_vector", help="Key for vectors in each entry")
    parser.add_argument("--name-key", default="name", help="Key for display name in each entry")
    parser.add_argument("--neighbors", type=int, default=20, help="UMAP n_neighbors (try 10, 20, 40)")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (smaller = tighter clusters)")
    parser.add_argument("--pca50", action="store_true", help="Apply PCA→50D before UMAP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("🧭 Loading data…")
    data = load_data(args.input)
    if data is None:
        return

    print("📦 Extracting features…")
    X, labels = extract_features(data, vector_key=args.vector_key, name_key=args.name_key)
    if X is None:
        return

    print("🧼 Unit-normalizing (cosine-friendly)…")
    X = unit_normalize_rows(X)

    if args.pca50:
        X_proc = maybe_pca_50(X, use_pca=True, random_state=args.seed)
    else:
        X_proc = X

    print("🎯 Running UMAP 2D…")
    coords2d, reducer2d = run_umap(
        X_proc,
        n_components=2,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed
    )

    print("🎯 Running UMAP 3D…")
    coords3d, reducer3d = run_umap(
        X_proc,
        n_components=3,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed
    )

    print("💾 Saving JSON outputs…")
    save_umap_json(coords2d, labels, "main_UMAP2D.json")
    save_umap_json(coords3d, labels, "main_UMAP3D.json")

    print("🖼️ Creating Plotly visualizations…")
    plot_2d(coords2d, labels, out_html="umap_2d_visualization.html")
    plot_3d(coords3d, labels, out_html="umap_3d_visualization.html")

    print("\n✅ Done.\nFiles created:")
    print("  - main_UMAP2D.json")
    print("  - main_UMAP3D.json")
    print("  - umap_2d_visualization.html")
    print("  - umap_3d_visualization.html")

if __name__ == "__main__":
    main()
