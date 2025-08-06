#!/usr/bin/env python3
"""
Generate final 3D PCA projection with Yahya as current user and update React app.
"""

import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    print("📊 Generating final 3D PCA with Yahya...")
    
    # Load combined embeddings with Yahya
    try:
        with open('book_gemini_embeddings_with_yahya.json', 'r') as f:
            embeddings_data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: book_gemini_embeddings_with_yahya.json not found.")
        return
    
    print(f"📈 Loaded {len(embeddings_data)} user embeddings (including Yahya)")
    
    # Extract embeddings and names
    embeddings = []
    names = []
    
    for user in embeddings_data:
        embeddings.append(user['books_vector'])
        names.append(user['name'])
    
    embeddings = np.array(embeddings)
    print(f"📊 Embeddings shape: {embeddings.shape}")
    
    # Perform PCA to 3D
    print("🔄 Performing PCA reduction to 3D...")
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    print(f"📈 PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"📈 Total variance captured: {np.sum(pca.explained_variance_ratio_):.1%}")
    
    # Create visualization data
    users_3d = []
    for i, name in enumerate(names):
        user_data = {
            "name": name,
            "x": float(embeddings_3d[i, 0]),
            "y": float(embeddings_3d[i, 1]), 
            "z": float(embeddings_3d[i, 2]),
            "isCurrentUser": (name == "Yahya Rahhawi")
        }
        users_3d.append(user_data)
    
    # Create output data structure
    pca_data = {
        "users": users_3d,
        "explained_variance": {
            "pc1": float(pca.explained_variance_ratio_[0]),
            "pc2": float(pca.explained_variance_ratio_[1]),
            "pc3": float(pca.explained_variance_ratio_[2]),
            "total": float(np.sum(pca.explained_variance_ratio_))
        },
        "statistics": {
            "total_users": len(users_3d),
            "dimensions_original": embeddings.shape[1],
            "dimensions_reduced": 3,
            "embedding_model": "Google Gemini text-embedding-004"
        }
    }
    
    # Save for React app
    react_output = "../Mello-prototype/src/data/pca_3d_data.json"
    with open(react_output, 'w') as f:
        json.dump(pca_data, f, indent=2)
    
    # Save embeddings for similarity calculations
    react_embeddings = "../Mello-prototype/src/data/combined_profiles_e5.json"
    
    # Convert to format expected by React app
    react_embeddings_data = []
    for user in embeddings_data:
        react_user = {
            "name": user['name'],
            "books_vector": user['books_vector'],
            "movies_vector": [0.0] * len(user['books_vector']),  # Empty for books-only
            "music_vector": [0.0] * len(user['books_vector']),   # Empty for books-only
            "book_taste": user.get('book_taste', '')
        }
        react_embeddings_data.append(react_user)
    
    with open(react_embeddings, 'w') as f:
        json.dump(react_embeddings_data, f, indent=2)
    
    # Create visualization plot
    print("🎨 Creating 3D visualization...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with Yahya highlighted
    for user in users_3d:
        if user['isCurrentUser']:
            ax.scatter(user['x'], user['y'], user['z'], c='red', s=100, alpha=1.0, label='Yahya Rahhawi')
        else:
            ax.scatter(user['x'], user['y'], user['z'], c='blue', s=30, alpha=0.7)
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)")
    ax.set_title(f"3D PCA: Gemini Book Embeddings with Yahya\\n({pca_data['explained_variance']['total']:.1%} total variance explained)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('final_gemini_book_pca_3d_with_yahya.png', dpi=300, bbox_inches='tight')
    
    # Check spread and statistics
    coords_array = np.array([[u['x'], u['y'], u['z']] for u in users_3d])
    
    print(f"\\n🎉 Successfully generated final 3D PCA with Yahya as current user!")
    print(f"📁 React data saved to: {react_output}")
    print(f"📁 Embeddings saved to: {react_embeddings}")  
    print(f"📁 Visualization saved to: final_gemini_book_pca_3d_with_yahya.png")
    print(f"📊 Explained variance: {pca_data['explained_variance']['total']:.1%}")
    print(f"📊 PC1: {pca_data['explained_variance']['pc1']:.1%}")
    print(f"📊 PC2: {pca_data['explained_variance']['pc2']:.1%}")
    print(f"📊 PC3: {pca_data['explained_variance']['pc3']:.1%}")
    print(f"📊 3D spread - X: [{coords_array[:, 0].min():.3f}, {coords_array[:, 0].max():.3f}]")
    print(f"📊 3D spread - Y: [{coords_array[:, 1].min():.3f}, {coords_array[:, 1].max():.3f}]")
    print(f"📊 3D spread - Z: [{coords_array[:, 2].min():.3f}, {coords_array[:, 2].max():.3f}]")
    print(f"👤 Yahya Rahhawi marked as current user")

if __name__ == "__main__":
    main()