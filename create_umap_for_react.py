#!/usr/bin/env python3
"""
Create UMAP coordinates from demo_population.json for React visualization
"""

import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap

# Load the population data
with open('demo_population.json', 'r', encoding='utf-8') as f:
    population_data = json.load(f)

print(f"📁 Loaded population: {population_data['name']}")
print(f"👥 Total users: {len(population_data['users'])}")

# Extract users with embeddings
users_with_embeddings = [user for user in population_data['users'] if user.get('embedding')]
print(f"🔢 Users with embeddings: {len(users_with_embeddings)}")

if len(users_with_embeddings) == 0:
    print("❌ No users with embeddings found!")
    exit(1)

# Extract embeddings and metadata
embeddings = np.array([user['embedding'] for user in users_with_embeddings])
names = [user['name'] for user in users_with_embeddings]
special_flags = [user.get('special', False) for user in users_with_embeddings]
taste_profiles = [user.get('taste_profile', 'No profile available') for user in users_with_embeddings]

print(f"📊 Embedding dimensions: {embeddings.shape}")

# Standardize embeddings
print("📏 Standardizing embeddings...")
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Apply UMAP 3D
print("🗺️ Applying UMAP 3D transformation...")
n_neighbors = min(15, len(users_with_embeddings) - 1)
umap_3d = umap.UMAP(
    n_components=3,
    n_neighbors=n_neighbors,
    min_dist=0.1,
    random_state=42,
    metric='cosine'
)
umap_coords = umap_3d.fit_transform(embeddings_scaled)

print(f"✅ UMAP completed! Shape: {umap_coords.shape}")

# Create visualization data structure
viz_data = {
    "users": [],
    "statistics": {
        "total_users": len(users_with_embeddings),
        "special_users": sum(special_flags),
        "regular_users": len(users_with_embeddings) - sum(special_flags),
        "dimensions_original": embeddings.shape[1],
        "dimensions_reduced": 3
    },
    "umap_params": {
        "n_neighbors": n_neighbors,
        "min_dist": 0.1,
        "metric": "cosine"
    },
    "explained_variance": {
        "note": "UMAP doesn't compute explained variance like PCA",
        "total": "N/A"
    }
}

# Add user data
for i, (name, special, profile) in enumerate(zip(names, special_flags, taste_profiles)):
    user_data = {
        "name": name,
        "special": special,
        "x": float(umap_coords[i, 0]),
        "y": float(umap_coords[i, 1]), 
        "z": float(umap_coords[i, 2]),
        "taste_profile": profile
    }
    viz_data["users"].append(user_data)

# Save UMAP visualization data
output_file = 'demo_population_umap.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(viz_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ UMAP visualization data saved to: {output_file}")

# Print summary
print(f"\n📊 Summary:")
print(f"👥 Total users: {viz_data['statistics']['total_users']}")
print(f"⭐ Special users: {viz_data['statistics']['special_users']}")
print(f"👤 Regular users: {viz_data['statistics']['regular_users']}")

# Show coordinate ranges
x_coords = [user['x'] for user in viz_data['users']]
y_coords = [user['y'] for user in viz_data['users']]
z_coords = [user['z'] for user in viz_data['users']]

print(f"\n📏 Coordinate ranges:")
print(f"X: {min(x_coords):.3f} to {max(x_coords):.3f}")
print(f"Y: {min(y_coords):.3f} to {max(y_coords):.3f}")  
print(f"Z: {min(z_coords):.3f} to {max(z_coords):.3f}")

# Show special users
special_users = [user for user in viz_data['users'] if user['special']]
print(f"\n⭐ Special users:")
for user in special_users:
    print(f"  - {user['name']}")

print(f"\n🚀 Ready for React visualization!")