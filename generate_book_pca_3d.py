#!/usr/bin/env python3
"""
Generate 3D PCA projection for the 200 book users.
Creates visualization data for the React app.
"""

import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    print("📊 Generating 3D PCA projection for book users...")
    
    # Load embeddings
    try:
        with open('book_embeddings_e5_200.json', 'r') as f:
            embeddings_data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: book_embeddings_e5_200.json not found. Run generate_book_embeddings.py first.")
        return
    
    # Extract embeddings and names
    embeddings = []
    names = []
    
    for user in embeddings_data:
        embeddings.append(user['books_vector'])
        names.append(user['name'])
    
    embeddings = np.array(embeddings)
    print(f"📈 Loaded {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
    
    # Perform PCA to 3D
    print("🔄 Performing PCA reduction to 3D...")
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    # Create visualization data
    users_3d = []
    for i, name in enumerate(names):
        # Mark Yahya as current user if present
        is_current_user = name.lower() == "yahya rahhawi"
        
        user_data = {
            "name": name,
            "x": float(embeddings_3d[i, 0]),
            "y": float(embeddings_3d[i, 1]), 
            "z": float(embeddings_3d[i, 2]),
            "isCurrentUser": is_current_user
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
            "dimensions_reduced": 3
        }
    }
    
    # Save for React app
    react_output = "../Mello-prototype/src/data/pca_3d_data.json"
    with open(react_output, 'w') as f:
        json.dump(pca_data, f, indent=2)
    
    # Also save embeddings for similarity calculations
    react_embeddings = "../Mello-prototype/src/data/combined_profiles_e5.json"
    
    # Convert to format expected by React app (with combined_profiles structure)
    react_embeddings_data = []
    for user in embeddings_data:
        react_user = {
            "name": user['name'],
            "books_vector": user['books_vector'],
            "movies_vector": [0.0] * len(user['books_vector']),  # Empty for books-only
            "music_vector": [0.0] * len(user['books_vector'])    # Empty for books-only
        }
        react_embeddings_data.append(react_user)
    
    with open(react_embeddings, 'w') as f:
        json.dump(react_embeddings_data, f, indent=2)
    
    # Create visualization plot
    print("🎨 Creating 3D visualization...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    for user in users_3d:
        color = 'red' if user['isCurrentUser'] else 'blue'
        size = 100 if user['isCurrentUser'] else 30
        ax.scatter(user['x'], user['y'], user['z'], c=color, s=size, alpha=0.7)
        
        # Label current user
        if user['isCurrentUser']:
            ax.text(user['x'], user['y'], user['z'], '  Yahya (You)', fontsize=10)
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)")
    ax.set_title(f"3D PCA: Book Taste Space\n({pca_data['explained_variance']['total']:.1%} total variance explained)")
    
    plt.tight_layout()
    plt.savefig('book_taste_pca_3d_200.png', dpi=300, bbox_inches='tight')
    
    print(f"\n🎉 Successfully generated 3D PCA projection!")
    print(f"📁 React data saved to: {react_output}")
    print(f"📁 Embeddings saved to: {react_embeddings}")  
    print(f"📁 Visualization saved to: book_taste_pca_3d_200.png")
    print(f"📊 Explained variance: {pca_data['explained_variance']['total']:.1%}")
    print(f"📊 PC1: {pca_data['explained_variance']['pc1']:.1%}")
    print(f"📊 PC2: {pca_data['explained_variance']['pc2']:.1%}")
    print(f"📊 PC3: {pca_data['explained_variance']['pc3']:.1%}")

if __name__ == "__main__":
    main()